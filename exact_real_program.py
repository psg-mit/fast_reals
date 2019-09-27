from typing import List, Set, Dict, Tuple, Optional, Callable
import operator
import bigfloat as bf

from utils import cast_input


class ExactRealProgram:

    def __init__(self, children: List, lower=None, upper=None):
        super(ExactRealProgram, self).__init__()
        self.children = children
        self.lower = lower
        self.upper = upper

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

    def __truediv__(self, other: 'ExactRealProgram'):
        other = cast_input(other)
        return ExactDiv([self, other])

    def __radd__(self, other: 'ExactRealProgram'):
        return cast_input(other) + self

    def __rsub__(self, other: 'ExactRealProgram'):
        return cast_input(other) - self

    def __rmul__(self, other: 'ExactRealProgram'):
        return cast_input(other) * self

    def __rtruediv__(self, other: 'ExactRealProgram'):
        return cast_input(other) / self

    def interval_bf_operation(self):
        raise NotImplementedError

    @property
    @staticmethod
    def bf_operation(self):
        raise NotImplementedError

    def evaluate(self, precision):
        raise NotImplementedError


class ExactConstant(ExactRealProgram):
    def __init__(self, constant: float):
        super(ExactConstant, self).__init__([])
        self.lower = constant
        self.upper = constant

    def __str__(self, level=0):
        return "\t"*level + str([self.lower, self.upper]) + "\n"

    def evaluate(self, precision):
        return self

    def interval_bf_operation(self, args, precison):
        return self


class BinOp(ExactRealProgram):

    def evaluate(self, precison):
        left, right = self.children[0].evaluate(precison), self.children[1].evaluate(precison)
        return self.interval_bf_operation([left, right], precison)


class ExactAdd(BinOp):
    operator_string = '+'

    def interval_bf_operation(self, args, precision_of_result):
        left, right = args
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        lower = bf.add(left.lower, right.lower, context_down)
        upper = bf.add(left.lower, right.lower, context_up)
        return ExactAdd(self.children, lower, upper)


class ExactSub(BinOp):
    operator_string = '-'
    bf_operation = bf.sub

    def interval_bf_operation(self, args, precision_of_result):
        left, right = args
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        lower = bf.sub(left.lower, right.lower, context_down)
        upper = bf.sub(left.lower, right.lower, context_up)
        return ExactAdd(self.children, lower, upper)


class ExactMul(BinOp):
    operator_string = '*'
    bf_operation = lambda self, l, precision: lambda x, y: bf.mul(x, y, precision)


class ExactDiv(BinOp):
    operator_string = '/'
    bf_operation = lambda self, l, precision: lambda x, y: bf.div(x, y, precision)


class GenericExactConstant(ExactRealProgram):

    def __init__(self, bf_const: Callable, lower=None, upper=None):
        super(GenericExactConstant, self).__init__([])
        self.bf_const = bf_const
        self.lower = lower
        self.upper = upper

    def evaluate(self, precision_of_result):
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        lower = self.bf_const(context_down)
        upper = self.bf_const(context_up)
        return GenericExactConstant(self.bf_const, lower, upper)

    def __str__(self, level=0):
        return "\t"*level + str([self.lower, self.upper]) + "\n"
