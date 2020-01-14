from typing import List, Callable
from copy import copy
import numpy as np
import bigfloat as bf
from bigfloat import BigFloat
from termcolor import colored

from time_ops import Time
from utils import cast_input


class ExactRealProgram:
    time: Time = Time()

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
        self.grad_precision = 1000

        # For pretty printing on critical_path
        self.color = 'blue'

    def _compute_grad(self, children: List) -> bf.BigFloat:
        context = bf.precision(self.grad_precision) + bf.RoundAwayFromZero
        grad = 0
        for (w1, w2), var in children:
            lower, upper = var.grad()
            grad_term = bf.add(bf.mul(w1, lower, context), bf.mul(w2, upper, context), context)
            grad = bf.add(grad_term, grad, context)
        return grad

    def grad(self) -> float:
        if self.lower_grad is None:
            self.lower_grad = self._compute_grad(self.ad_lower_children)
        if self.upper_grad is None:
            self.upper_grad = self._compute_grad(self.ad_upper_children)
        return self.lower_grad, self.upper_grad

    def apply(self, f: Callable):
        f(self)
        [child.apply(f) for child in self.children]

    def print(self):
        print([self.lower, self.upper])

    def full_string(self, level=0):
        value = str([round(float(self.lower), 2), round(float(self.upper), 2)])
        derivatives = str([self.lower_grad, self.upper_grad])
        ret = "\t"*level + self.operator_string + value + derivatives + "\n"
        for child in self.children:
            ret += child.full_string(level+1)
        return colored(ret, self.color)

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
        # Set weight to zero which sets derivatives to 0 for constant intervals
        if left.lower == left.upper:
            left.ad_lower_children = [([0, 0], adlc[1]) for adlc in left.ad_lower_children]
            left.ad_upper_children = [([0, 0], adlc[1]) for adlc in left.ad_upper_children]

        # Set weight to zero which sets derivatives to 0 for constant intervals
        if right.lower == right.upper:
            right.ad_lower_children = [([0, 0], adlc[1]) for adlc in right.ad_lower_children]
            right.ad_upper_children = [([0, 0], adlc[1]) for adlc in right.ad_upper_children]

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
        self.lower = ExactRealProgram.time.add(left.lower, right.lower, context_down)
        self.upper = ExactRealProgram.time.add(left.upper, right.upper, context_up)

        if ad:
            left, right = self.children

            left.ad_lower_children.append(((1, 0), self))
            right.ad_lower_children.append(((1, 0), self))

            left.ad_upper_children.append(((0, 1), self))
            right.ad_upper_children.append(((0, 1), self))


class ExactSub(BinOp):
    operator_string = '-'

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False):
        left, right = self.children
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive
        self.lower = ExactRealProgram.time.sub(left.lower, right.upper, context_down)
        self.upper = ExactRealProgram.time.sub(left.upper, right.lower, context_up)

        if ad:
            left, right = self.children

            left.ad_lower_children.append(((1,  0), self))
            right.ad_lower_children.append(((0, -1), self))

            left.ad_upper_children.append(((0, 1), self))
            right.ad_upper_children.append(((-1, 0), self))


class ExactMul(BinOp):
    operator_string = '*'

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False) -> ExactRealProgram:
        left, right = self.children

        ll, lu, rl, ru = left.lower, left.upper, right.lower, right.upper
        product = ExactMul.multiply(ll, lu, rl, ru, precision_of_result)
        self.lower, self.upper, ll_weights, lr_weights, ul_weights, ur_weights = product

        if ad:
            left, right = self.children

            left.ad_lower_children.append((ll_weights, self))
            right.ad_lower_children.append((lr_weights, self))

            left.ad_upper_children.append((ul_weights, self))
            right.ad_upper_children.append((ur_weights, self))

    @staticmethod
    def multiply(left_lower: BigFloat,
                 left_upper: BigFloat,
                 right_lower: BigFloat,
                 right_upper: BigFloat,
                 precision_of_result: int):
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive

        # Note: super inefficient to compute all pairs, kaucher multiplication in future?
        ll_down = ExactRealProgram.time.mul(left_lower, right_lower, context_down), (left_lower, 0), (right_lower, 0)
        lu_down = ExactRealProgram.time.mul(left_lower, right_upper, context_down), (left_lower, 0), (right_upper, 1)
        ul_down = ExactRealProgram.time.mul(left_upper, right_lower, context_down), (left_upper, 1), (right_lower, 0)
        uu_down = ExactRealProgram.time.mul(left_upper, right_upper, context_down), (left_upper, 1), (right_upper, 1)

        ll_up = ExactRealProgram.time.mul(left_lower, right_lower, context_up), (left_lower, 0), (right_lower, 0)
        lu_up = ExactRealProgram.time.mul(left_lower, right_upper, context_up), (left_lower, 0), (right_upper, 1)
        ul_up = ExactRealProgram.time.mul(left_upper, right_lower, context_up), (left_upper, 1), (right_lower, 0)
        uu_up = ExactRealProgram.time.mul(left_upper, right_upper, context_up), (left_upper, 1), (right_upper, 1)

        (lower_product, (ll, ll_ind), (lr, lr_ind)) = min([ll_down, lu_down, ul_down, uu_down], key=lambda x: x[0])
        (upper_product, (ul, ul_ind), (ur, ur_ind)) = max([ll_up, lu_up, ul_up, uu_up], key=lambda x: x[0])

        # Assign derivative weights based on partial derivatives in chain rule
        llw, lrw, ulw, urw = [[0, 0], [0, 0], [0, 0], [0, 0]]
        if ll_ind == 0:
            llw[0] = float(lr)
        else:
            ulw[0] = float(lr)

        if lr_ind == 0:
            lrw[0] = float(ll)
        else:
            urw[0] = float(ll)

        if ul_ind == 0:
            llw[1] = float(ur)
        else:
            ulw[1] = float(ur)

        if ur_ind == 0:
            lrw[1] = float(ul)
        else:
            urw[1] = float(ul)
        return lower_product, upper_product, llw, lrw, ulw, urw


class ExactDiv(BinOp):
    operator_string = '/'

    def interval_bf_operation(self,
                              precision_of_result: int,
                              ad: bool = False) -> ExactRealProgram:
        left, right = self.children

        inv_lower, inv_upper, inv_lrw, inv_urw = ExactDiv.invert(right.lower, right.upper, precision_of_result)
        product = ExactMul.multiply(left.lower, left.upper, inv_lower, inv_upper, precision_of_result)
        self.lower, self.upper, llw, lrw, ulw, urw = product

        if ad:
            left, right = self.children

            # Since the right passes through an inversion, it incurs -1/x^2 factor
            left.ad_lower_children.append((llw, self))
            right.ad_lower_children.append(([inv_lrw[1] * urw[0], inv_lrw[1] * urw[1]], self))

            left.ad_upper_children.append((ulw, self))
            right.ad_upper_children.append(([inv_urw[0] * lrw[0], inv_urw[0] * lrw[1]], self))

    @staticmethod
    def invert(lower: BigFloat, upper: BigFloat, precision_of_result: int) -> ExactRealProgram:
        context_down = bf.precision(precision_of_result) + bf.RoundTowardNegative
        context_up = bf.precision(precision_of_result) + bf.RoundTowardPositive

        # interval doesn't contain zero then invert and flip [1 / y2, 1 / y1]
        if (lower > 0 and upper > 0) or (lower < 0 and upper < 0):
            inv_lower = ExactRealProgram.time.div(1, upper, context_down)
            inv_upper = ExactRealProgram.time.div(1, lower, context_up)
            lw = [0, -float(inv_upper)**2]
            uw = [-float(inv_lower)**2, 0]

        # [lower, 0] -> [-infty, 1 / y1]
        elif lower < 0 and upper == 0:
            inv_lower = BigFloat('-inf')
            inv_upper = ExactRealProgram.time.div(1, lower, context_up)
            lw = [0, float('nan')]
            uw = [-float(inv_lower)**2, 0]

        # [0, upper] -> [1 / y2, infty]
        elif lower == 0 and upper > 0:
            inv_lower = ExactRealProgram.time.div(1, upper, context_down)
            inv_upper = BigFloat('inf')
            lw = [0, -float(inv_upper)**2]
            uw = [float('nan'), 0]

        # If the interval includes 0 just give up and return [-infty, infty]
        # Note: an alternative is to split up intervals, but that's too tricky for now
        elif lower < 0 < upper:
            inv_lower = BigFloat('-inf')
            inv_upper = BigFloat('inf')
            lw = [0, float('nan')]
            uw = [float('nan'), 0]

        # Interval is probably such that lower is greater than upper
        else:
            raise ValueError("Input interval is invalid for division")

        return inv_lower, inv_upper, lw, uw


class ExactLeaf(ExactRealProgram):

    def interval_bf_operation(self,
                              other: 'ExactRealProgram',
                              precision_of_result: int,
                              ad: bool = False) -> 'ExactRealProgram':
        pass

    def __str__(self, level=0):
        print("\t"*level + str([self.lower, self.upper]) + "\n")

    def full_string(self, level=0):
        value = str([round(float(self.lower), 2), round(float(self.upper), 2)])
        derivatives = str([self.lower_grad, self.upper_grad])
        return colored("\t"*level + value + derivatives + "\n", self.color)

    def apply(self, f: Callable):
        f(self)

    def subtree_size(self):
        return 1

    def evaluate_at(self, precisions: List[int], ad: bool = False):
        self.evaluate(precisions[0], ad)


class ExactInterval(ExactLeaf):

    def __init__(self, lower, upper):
        super(ExactInterval, self).__init__([], lower, upper)

    def evaluate(self, precision_of_result: int, ad: bool = False):
        pass


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
        context_down = bf.precision(53) + bf.RoundTowardNegative
        context_up = bf.precision(53) + bf.RoundTowardPositive
        super(ExactConstant, self).__init__([], BigFloat(constant, context_down), BigFloat(constant, context_up))

    def evaluate(self, precision: int, ad: bool = False):
        pass

    def full_string(self, level=0):
        # If it is exactly a constant, just display the value.
        if self.lower == self.upper:
            return colored("\t"*level + str(round(float(self.lower), 2)) + "\n", self.color)
        return super().full_string(level)


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

    def binary_to_range(self, point: str, precision) -> BigFloat:
        """ Bring a point from [0, 1] to a given range. """
        point = point[:precision + 2]
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
            return self.binary_to_range(self.cache, precision)

        else:
            # Add precision bit-by-bit
            for i in range(precision - self.cached_precision):
                bit_of_randomness = np.random.randint(2)
                self.cache += str(bit_of_randomness)
            self.cached_precision = precision
        return self.binary_to_range(self.cache, precision)
