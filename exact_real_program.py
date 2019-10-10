from typing import List, Callable, Tuple

from copy import copy
import numpy as np
import bigfloat as bf
from bigfloat import BigFloat

from utils import cast_input


class ExactRealProgram:

    def __init__(self, children: List, lower=None, upper=None):
        super(ExactRealProgram, self).__init__()
        # For interval computation
        self.children = children
        self.lower = lower
        self.upper = upper
        self.parent = None
        for child in children:
            child.parent = self

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

    def apply(self, f: Callable):
        f(self)
        [child.apply(f) for child in self.children]

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

    def evaluate(self, precision: int, ad: bool = False):
        raise NotImplementedError

    def evaluate_at(self, precisions: List[int], ad: bool = False):
        """ Evaluate the subtree with the precisions specified by 
        the in-order traversal of the subtree rooted here. """
        raise NotImplementedError


class BinOp(ExactRealProgram):

    def evaluate(self, precison: int, ad: bool = False):
        left, right = self.children
        left.evaluate(precison, ad)
        right.evaluate(precison, ad)
        self.interval_bf_operation(precison, ad)

    def evaluate_at(self, precisions: List[int], ad: bool = False):
        left, right = self.children
        left_size = left.subtree_size()
        left.evaluate_at(precisions[1: left_size + 1], ad)
        right.evaluate_at(precisions[left_size + 1:], ad)
        self.interval_bf_operation(precisions[0], ad)

    def subtree_size(self):
        left, right = self.children
        return 1 + left.subtree_size() + right.subtree_size()


class ExactAdd(BinOp):
    operator_string = '+'

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False):
        left, right = self.children
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


class ExactSub(BinOp):
    operator_string = '-'
    bf_operation = bf.sub

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False):
        left, right = self.children
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        self.lower = bf.sub(left.lower, right.upper, context_down)
        self.upper = bf.sub(left.upper, right.lower, context_up)

        if ad:
            left, right = self.children

            left.ad_lower_children.append((1.0, self))
            right.ad_lower_children.append((-1.0, self))

            left.ad_upper_children.append((1.0, self))
            right.ad_upper_children.append((-1.0, self))


class ExactMul(BinOp):
    operator_string = '*'

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False) -> ExactRealProgram:
        left, right = self.children

        ll, lu, rl, ru = left.lower, left.upper, right.lower, right.upper
        product = ExactMul.multiply(ll, lu, rl, ru, precision_of_result)
        (self.lower, ll_contrib, lr_contrib), (self.upper, ul_contrib, ur_contrib) = product

        if ad:
            left, right = self.children

            left.ad_lower_children.append((float(lr_contrib), self))
            right.ad_lower_children.append((float(ll_contrib), self))

            left.ad_upper_children.append((float(ur_contrib), self))
            right.ad_upper_children.append((float(ul_contrib), self))

    @staticmethod
    def multiply(left_lower: BigFloat,
                 left_upper: BigFloat,
                 right_lower: BigFloat,
                 right_upper: BigFloat,
                 precision_of_result: int):
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive

        # Note: super inefficient to compute all pairs, kaucher multiplication in future?
        ll_down = bf.mul(left_lower, right_lower, context_down), left_lower, right_lower
        lu_down = bf.mul(left_lower, right_upper, context_down), left_lower, right_upper
        ul_down = bf.mul(left_upper, right_lower, context_down), left_upper, right_lower
        uu_down = bf.mul(left_upper, right_upper, context_down), left_upper, right_upper

        ll_up = bf.mul(left_lower, right_lower, context_up), left_lower, right_lower
        lu_up = bf.mul(left_lower, right_upper, context_up), left_lower, right_upper
        ul_up = bf.mul(left_upper, right_lower, context_up), left_upper, right_lower
        uu_up = bf.mul(left_upper, right_upper, context_up), left_upper, right_upper

        lower_vals = min([ll_down, lu_down, ul_down, uu_down], key=lambda x: x[0])
        upper_vals = max([ll_up, lu_up, ul_up, uu_up], key=lambda x: x[0])
        return lower_vals, upper_vals


class ExactDiv(BinOp):
    operator_string = '/'

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False) -> ExactRealProgram:
        left, right = self.children

        inv_lower, inv_upper = ExactDiv.invert(right.lower, right.upper, precision_of_result)
        product = ExactMul.multiply(left.lower, left.upper, inv_lower, inv_upper, precision_of_result)
        (self.lower, ll_contrib, lr_contrib), (self.upper, ul_contrib, ur_contrib) = product

        if ad:
            left, right = self.children

            # Since the right passes through an inversion, it incurs -1/x^2 factor
            left.ad_lower_children.append((float(lr_contrib), self))
            right.ad_lower_children.append((-float(ll_contrib) / float(right.lower)**2, self))

            left.ad_upper_children.append((float(ur_contrib), self))
            right.ad_upper_children.append((-float(ul_contrib) / float(right.upper)**2, self))

    @staticmethod
    def invert(lower: BigFloat, upper: BigFloat, precision_of_result: int) -> ExactRealProgram:
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive

        # interval doesn't contain zero then invert and flip
        if (lower > 0 and upper > 0) or (lower < 0 and upper < 0):
            inv_lower = bf.div(1, upper, context_down)
            inv_upper = bf.div(1, lower, context_up)

        # [lower, 0] -> [-infty, 1 / y1]
        elif lower < 0 and upper == 0:
            inv_lower = BigFloat('-inf')
            inv_upper = bf.div(1, lower, context_up)

        # [0, upper] -> [1 / y2, infty]
        elif lower == 0 and upper > 0:
            inv_lower = bf.div(1, upper, context_down)
            inv_upper = BigFloat('inf')

        # If the interval includes 0 just give up and return [-infty, infty]
        # Note: an alternative is to split up intervals, but that's too tricky for now
        elif lower < 0 < upper:
            inv_lower = BigFloat('-inf')
            inv_upper = BigFloat('inf')

        # Interval is probably such that lower is greater than upper
        else:
            raise ValueError("Input interval is invalid for division")

        return inv_lower, inv_upper


class ExactLeaf(ExactRealProgram):

    def interval_bf_operation(self,
                              other: 'ExactRealProgram',
                              precision_of_result: int,
                              ad: bool = False) -> 'ExactRealProgram':
        pass

    def __str__(self, level=0):
        return "\t"*level + str([self.lower, self.upper]) + "\n"

    def apply(self, f: Callable):
        f(self)

    def subtree_size(self):
        return 1

    def evaluate_at(self, precisions: List[int], ad: bool = False):
        self.evaluate(precisions[0], ad)


class GenericExactConstant(ExactLeaf):

    def __init__(self, bf_const: Callable, lower=None, upper=None):
        super(GenericExactConstant, self).__init__([], lower, upper)
        self.bf_const = bf_const

    def evaluate(self, precision_of_result: int, ad: bool = False):
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        self.lower = self.bf_const(context_down)
        self.upper = self.bf_const(context_up)


class ExactConstant(ExactLeaf):
    def __init__(self, constant: float):
        super(ExactConstant, self).__init__([], constant, constant)

    def evaluate(self, precision: int, ad: bool = False):
        pass


class ExactVariable(ExactLeaf):

    def __init__(self, var_lower: BigFloat, var_upper: BigFloat, lower=None, upper=None):
        super(ExactVariable, self).__init__([], lower, upper)
        self.var_lower = var_lower
        self.var_upper = var_upper
        self.cache: BigFloat = None
        self.cached_precision: int = 0

    def evaluate(self, precision_of_result: int, ad: bool = False):
        # Randomly sample from the variable range with a prefix consistent with all previous queries
        self.lower = self.variable_at_point(precision_of_result, bf.RoundTowardNegative)
        self.upper = self.variable_at_point(precision_of_result, bf.RoundTowardPositive)

    def sample(self):
        """ Dump randomness cache to force a resampling in the future. """
        self.cache = None
        self.lower = None
        self.upper = None

    def binary_to_range(self, point: str) -> BigFloat:
        """ Bring a point from [0, 1] to a given range. """
        interval_width = self.var_upper - self.var_lower
        bitwidth = len(point) - 2
        full_prec_context = bf.precision(bitwidth) + bf.RoundTowardNegative

        # Map to [0, 1]
        value = bf.mul(int(point, 2), bf.exp2(-bitwidth, full_prec_context), full_prec_context)
        rescaled = bf.add(bf.mul(value, interval_width, full_prec_context), self.var_lower, full_prec_context)
        return rescaled

    def variable_at_point(self, precision: int, round_mode: bf.Context) -> BigFloat:
        assert 0 < precision, "Input precision must be positive. "

        if self.cache is None:
            self.cache = "0b" + str(np.random.randint(2))
            self.cached_precision = 1

        if self.cached_precision >= precision:
            return self.binary_to_range(self.cache)

        else:
            # Add precision bit-by-bit
            for i in range(precision - self.cached_precision):
                bit_of_randomness = np.random.randint(2)
                self.cache += str(bit_of_randomness)
            self.cached_precision = precision
        return self.binary_to_range(self.cache)
