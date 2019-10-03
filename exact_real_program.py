from typing import List, Set, Dict, Tuple, Optional, Callable
import operator
from functools import reduce
from copy import copy

import bigfloat as bf
from utils import cast_input


class ExactRealProgram:

    def __init__(self, children: List, lower=None, upper=None):
        super(ExactRealProgram, self).__init__()
        # For interval computation
        self.children = children
        self.lower = lower
        self.upper = upper

        # For evaluation
        self.lower_value = None
        self.upper_value = None

        # For computing ad
        self.ad_lower_children = []
        self.ad_upper_children = []
        self.lower_grad = None
        self.upper_grad = None

    def grad(self) -> float:
        if self.lower_grad is None:
            self.lower_grad = sum(weight * var.grad()[0] for weight, var in self.ad_lower_children)
        if self.upper_grad is None:
            self.upper_grad = sum(weight * var.grad()[1] for weight, var in self.ad_upper_children)
        return self.lower_grad, self.upper_grad

    def print(self):
        print([self.lower, self.upper])

    def __str__(self, level=0):
        ret = "\t" * level + self.operator_string + "\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __add__(self, other: 'ExactRealProgram'):
        other = cast_input(other)
        return ExactAdd([self, other])

    def __sub__(self, other: 'ExactRealProgram'):
        other = cast_input(other)
        return ExactSub([self, other])

    def __mul__(self, other: 'ExactRealProgram'):
        other = cast_input(other)
        return ExactMul([self, other])

    # def __truediv__(self, other: 'ExactRealProgram'):
    #     other = cast_input(other)
    #     return ExactDiv([self, other])

    def __radd__(self, other: 'ExactRealProgram'):
        return cast_input(other) + self

    def __rsub__(self, other: 'ExactRealProgram'):
        return cast_input(other) - self

    def __rmul__(self, other: 'ExactRealProgram'):
        return cast_input(other) * self

    # def __rtruediv__(self, other: 'ExactRealProgram'):
    #     return cast_input(other) / self

    def interval_bf_operation(self,
                              other: 'ExactRealProgram',
                              precision_of_result: int,
                              ad: bool = False) -> 'ExactRealProgram':
        raise NotImplementedError

    def clone_without_grad(self):
        new_self = copy(self)
        new_self.lower_grad = None
        new_self.upper_grad = None
        return new_self

    @property
    @staticmethod
    def bf_operation(self):
        raise NotImplementedError

    def evaluate(self, precision):
        raise NotImplementedError


class ExactConstant(ExactRealProgram):
    def __init__(self, constant: float):
        super(ExactConstant, self).__init__([], constant, constant)

    def __str__(self, level=0):
        return "\t"*level + str([self.lower, self.upper]) + "\n"

    def evaluate(self, precision: int, ad: bool = False):
        return self

    def interval_bf_operation(self,
                              other: 'ExactRealProgram',
                              precision_of_result: int,
                              ad: bool = False) -> 'ExactRealProgram':
        return self


class BinOp(ExactRealProgram):

    def evaluate(self, precison: int, ad: bool = False):
        left, right = self.children
        self.left_value = left.evaluate(precison, ad)
        self.right_value = right.evaluate(precison, ad)
        return self.interval_bf_operation(precison, ad)


class ExactAdd(BinOp):
    operator_string = '+'

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False) -> ExactRealProgram:
        left, right = self.left_value, self.right_value
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        self.lower = bf.add(left.lower, right.lower, context_down)
        self.upper = bf.add(left.upper, right.upper, context_up)

        if ad:
            left, right = self.children

            left.ad_lower_children.append((1.0, self))
            right.ad_lower_children.append((1.0, self))

            left.ad_upper_children.append((1.0, self))
            right.ad_upper_children.append((1.0, self))

        return self


class ExactSub(BinOp):
    operator_string = '-'
    bf_operation = bf.sub

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False) -> ExactRealProgram:
        left, right = self.left_value, self.right_value
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        lower = bf.sub(left.lower, right.lower, context_down)
        upper = bf.sub(left.upper, right.upper, context_up)
        return ExactSub(self.children, lower, upper)


class ExactMul(BinOp):
    operator_string = '*'
    bf_operation = lambda self, l, precision: lambda x, y: bf.mul(x, y, precision)

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False) -> ExactRealProgram:
        left, right = self.left_value, self.right_value
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        
        # Note: super inefficient to compute all pairs, kaucher multiplication in future?
        ll = bf.mul(left.lower, right.lower, context_down)
        lu = bf.mul(left.lower, right.upper, context_up)
        ul = bf.mul(left.upper, right.lower, context_down)
        uu = bf.mul(left.upper, right.upper, context_up)

        # Notes: Make a version that needs only log rather than linear comparisons. Think binary tree.
        self.lower = reduce(lambda x, y: bf.min(x, y, context_down), [ll, lu, ul, uu])
        self.upper = reduce(lambda x, y: bf.max(x, y, context_up), [ll, lu, ul, uu])

        if ad:
            left, right = self.children

            left.ad_lower_children.append((float(right.lower), self))
            right.ad_lower_children.append((float(left.lower), self))

            left.ad_upper_children.append((float(right.upper), self))
            right.ad_upper_children.append((float(left.upper), self))

        return self


# class ExactDiv(BinOp):
#     operator_string = '/'
#     bf_operation = lambda self, l, precision: lambda x, y: bf.div(x, y, precision)


class GenericExactConstant(ExactRealProgram):

    def __init__(self, bf_const: Callable, lower=None, upper=None):
        super(GenericExactConstant, self).__init__([], lower, upper)
        self.bf_const = bf_const

    def evaluate(self, precision_of_result: int, ad: bool = False) -> ExactRealProgram:
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        self.lower = self.bf_const(context_down)
        self.upper = self.bf_const(context_up)
        return self

    def interval_bf_operation(self,
                              other: 'ExactRealProgram',
                              precision_of_result: int,
                              ad: bool = False) -> 'ExactRealProgram':
        return self

    def __str__(self, level=0):
        return "\t"*level + str([self.lower, self.upper]) + "\n"
