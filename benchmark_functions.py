# Each of the benchmarks from https://github.com/soarlab/FPTuner/tree/master/examples/primitives
from typing import List
import bigfloat as bf

from exact_real_program import UnaryOperator


def simplest_test(params: List):
    big_const, e, euler, pi = params
    # return UnaryOperator(UnaryOperator(UnaryOperator(UnaryOperator(UnaryOperator(UnaryOperator(e, bf.exp, bf.exp), bf.log, bf.log), bf.exp, bf.exp), bf.log, bf.log), bf.exp, bf.exp), bf.log, bf.log) + big_const * pi
    return e + big_const * pi

# var_T = IR.RealVE("T", 0, float(300.0)-float(0.01), float(300.0)+float(0.01))
# var_a = IR.RealVE("a", 1, float(0.401)-float(1e-06), float(0.401)+float(1e-06))
# var_b = IR.RealVE("b", 2, float(42.7e-06)-float(1e-10), float(42.7e-06)+float(1e-10))
# var_N = IR.RealVE("N", 3, float(995.0), float(1005.0))
# var_p = IR.FConst(float(3.5e7))
# var_V = IR.RealVE("V", 4, float(0.1)-float(0.005), float(0.5)+float(0.005))
# const_k = IR.FConst(float(1.3806503e-23))
def carbon_gas(params: List):
    var_T, var_a, var_b, var_N, var_V, var_p, const_k = params

    temp0 = var_N / var_V

    sub0 = temp0 * temp0
    sub1 = var_V - var_N * var_b
    sub2 = (var_p + var_a * sub0) * sub1
    sub3 = (const_k * var_N) * var_T

    rel = sub2 - sub3
    return rel


# doppler1
# var_u = IR.RealVE("u", 0, (-100.0 - 0.0000001), (100.0 + 0.0000001))
# var_v = IR.RealVE("v", 1, (20.0 - 0.000000001), (20000.0 + 0.000000001))
# var_T = IR.RealVE("T", 2, (-30.0 - 0.000001), (50.0 + 0.000001))
# const_t = 331.4
# const_r = 0.6
#
# doppler2
# var_u = IR.RealVE("u", 0, (-125.0 - 0.000000000001), (125.0 + 0.000000000001))
# var_v = IR.RealVE("v", 1, (15.0 - 0.001), (25000.0 + 0.001))
# var_T = IR.RealVE("T", 2, (-40.0 - 0.00001), (60.0 + 0.00001))
#
# doppler3
# var_u = IR.RealVE("u", 0, (-30.0 - 0.0001), (120.0 + 0.0001))
# var_v = IR.RealVE("v", 1, (320.0 - 0.00001), (20300.0 + 0.00001))
# var_T = IR.RealVE("T", 2, (-50.0 - 0.000000001), (30.0 + 0.000000001))
def doppler(params: List):
    var_u, var_v, var_T = params
    t1 = 331.4 + 0.6 * var_T
    temp = t1 + var_u
    temp = temp * temp
    r = ((6 - t1) * var_v) / temp
    return r


# var_x1 = IR.RealVE("x1", 0, -5.0, 5.0)
# var_x2 = IR.RealVE("x2", 1, -20.0, 5.0)
def jet(params: List):
    var_x1, var_x2 = params
    temp0 = var_x1 * var_x1
    temp1 = temp0 * var_x1

    sub2_0 = (3 * temp0 + 2 * var_x2) - var_x1
    sub2_1 = temp0 + 1
    sub2 = sub2_0 / sub2_1

    sub3_0_0_0 = 2 * var_x1 * sub2
    sub3_0_0_1 = sub2 - 3
    sub3_0_0 = sub3_0_0_0 * sub3_0_0_1
    sub3_0_1 = temp0 * (4 * sub2 - 6)
    sub3_0 = sub3_0_0 + sub3_0_1
    sub3_1 = temp0 + 1
    sub3 = sub3_0 * sub3_1

    rel_temp0 = (sub3 + 3 * temp0 * sub2 + temp1) + var_x1
    rel_temp1 = 3 * sub2
    rel = var_x1 + rel_temp0 + rel_temp1
    return rel


# const_r = IR.FConst(4.0)
# const_k = IR.FConst(1.11)
# var_x = IR.RealVE("x", 0, 0.1, 0.3)
def predator_prey(params: List):
    var_x, const_r, const_k = params

    temp0 = const_r * var_x * var_x
    temp1 = var_x / const_k
    temp2 = temp1 + temp1
    temp3 = 1 + temp2

    rel = temp0 / temp3
    return rel


# Rigid body 1
# var_x1 = IR.RealVE("x1", 0, (-15.0 - delta), (15.0 + delta))
# var_x2 = IR.RealVE("x2", 1, (-15.0 - delta), (15.0 + delta))
# var_x3 = IR.RealVE("x3", 2, (-15.0 - delta), (15.0 + delta))
# delta = 1e-08
def rigid_body1(params: List):
    var_x1, var_x2, var_x3 = params

    sub_x1x2 = var_x1 * var_x2
    sub_x2x3 = var_x2 * var_x3

    # TODO: I'm not sure about what to do.
    # what I did is based on:
    # https://github.com/soarlab/FPTuner/blob/29898c96741db341716a20ebfc4d12138cac2c99/src/tft_ir_api.py#L243
    r1_sub0 = 0 - sub_x1x2
    r1_sub1 = 2 * sub_x2x3
    r1 = (r1_sub0 - r1_sub1 - var_x1) - var_x3
    return r1


# Rigid body 2
# var_x1 = IR.RealVE("x1", 0, (-15.0 - delta), (15.0 + delta))
# var_x2 = IR.RealVE("x2", 1, (-15.0 - delta), (15.0 + delta))
# var_x3 = IR.RealVE("x3", 2, (-15.0 - delta), (15.0 + delta))
def rigid_body2(params: List):
    var_x1, var_x2, var_x3 = params

    sub_x1x2 = var_x1 * var_x2
    sub_x1x2x3 = sub_x1x2 * var_x3
    sub_x3x3 = var_x3 * var_x3

    r2_sub0 = 2 * sub_x1x2x3
    r2_sub1 = 3 * sub_x3x3
    r2_sub2 = var_x2 * sub_x1x2x3
    r2 = ((r2_sub0 + r2_sub1) - r2_sub2) + r2_sub1 - var_x2
    return r2


# x = IR.RealVE("x", 0, -1.57079632679, 1.57079632679)
def sine(params: List):
    x = params[0]
    x2 = x * x
    x3 = x2 * x

    x5 = x2 * x3
    x7 = x2 * x5

    rel_1 = x3 / 6
    rel_2 = x5 / 120
    rel_3 = x7 / 5040

    rel = (x - rel_1) + rel_2 - rel_3
    return rel


# x = IR.RealVE("x", 0, -2.0, 2.0)
# c1 = 0.954929658551372
# c2 = 0.12900613773279798
def sine_order3(params: List):
    x, c1, c2 = params
    x2 = x * x
    x3 = x2 * x

    rel_1 = c1 * x
    rel_2 = c2 * x3

    rel = rel_1 - rel_2
    return rel


# x = IR.RealVE("x", 0, 0.0, 1.0)
# c1 = 0.5
# c2 = 0.125
# c3 = 0.0625
# c4 = 0.0390625
def sqroot(params: List):
    x, c1, c2, c3, c4 = params
    x2 = x * x
    x3 = x2 * x

    x4 = x2 * x2

    rel_1 = c1 * x
    rel_2 = c2 * x2
    rel_3 = c3 * x3
    rel_4 = c4 * x4

    rel = (((1 + rel_1) - rel_2) + rel_3) - rel_4
    return rel


# var_v = IR.RealVE("v", 0, (-4.5 - 0.0000001), (-0.3 + 0.0000001))
# var_w = IR.RealVE("w", 1, (0.4 - 0.000000000001), (0.9 + 0.000000000001))
# var_r = IR.RealVE("r", 2, (3.8 - 0.00000001), (7.8 + 0.00000001))
# c1 = 0.125
# c2 = 4.5
def turbine1(params: List):
    var_v, var_w, var_r, c1, c2 = params
    sub_1v = 1 - var_v
    sub_ww = var_w * var_w
    sub_rr = var_r * var_r
    sub_2v = 2 * var_v
    sub_wwrr = sub_ww * sub_rr
    sub_wwrr1v = sub_wwrr / sub_1v
    sub_2rr = 2 / sub_rr

    r1_sub0 = c1 * (3 - sub_2v) * sub_wwrr1v

    r1 = (3 + sub_2rr - r1_sub0) - c2
    return r1


# var_v = IR.RealVE("v", 0, (-4.5 - 0.0000001), (-0.3 + 0.0000001))
# var_w = IR.RealVE("w", 1, (0.4 - 0.000000000001), (0.9 + 0.000000000001))
# var_r = IR.RealVE("r", 2, (3.8 - 0.00000001), (7.8 + 0.00000001))
# c1 = 0.5
# c2 = 2.5
def turbine2(params: List):
    var_v, var_w, var_r, c1, c2 = params
    sub_1v = 1 - var_v
    sub_ww = var_w * var_w
    sub_rr = var_r * var_r
    sub_wwrr = sub_ww * sub_rr
    sub_wwrr1v = sub_wwrr / sub_1v

    r2_sub0 = 6 * var_v
    r2_sub1 = c1 * var_v * sub_wwrr1v
    r2 = r2_sub0 - r2_sub1 - c2
    return r2


# var_v = IR.RealVE("v", 0, (-4.5 - 0.0000001), (-0.3 + 0.0000001))
# var_w = IR.RealVE("w", 1, (0.4 - 0.000000000001), (0.9 + 0.000000000001))
# var_r = IR.RealVE("r", 2, (3.8 - 0.00000001), (7.8 + 0.00000001))
# c1 = 0.125
# c2 = 4.5
def turbine3(params: List):
    var_v, var_w, var_r, c1, c2 = params
    sub_1v = 1 - var_v
    sub_ww = var_w * var_w
    sub_rr = var_r * var_r
    sub_2v = 2 * var_v
    sub_wwrr = sub_ww * sub_rr
    sub_wwrr1v = sub_wwrr / sub_1v
    sub_2rr = 2 / sub_rr

    r3_sub0 = c1 * (1 + sub_2v) * sub_wwrr1v

    r3 = (3 - sub_2rr - r3_sub0) - c2
    return r3


# var_x = IR.RealVE("x", 0, (0.1 - 1e-06), (0.3 + 1e-06))
# const_r = IR.FConst(4.0)
# const_k = IR.FConst(1.11)
def verlhulst(params: List):
    x, r, k = params
    temp0 = r * x
    temp1 = x / k
    temp2 = 1 + temp1
    rel = temp0 / temp2
    return rel
