from typing import Iterable, Callable, List

from unary_ops import (interval_sqrt,
                       interval_log,
                       interval_sin,
                       interval_cuberoot,
                       interval_fifthroot,
                       interval_cos,
                       interval_atan,
                       interval_exp,
                       interval_pow10)
from exact_real_program import ExactVariable, ExactConstant, GenericExactConstant, UnaryOperator
import benchmark_functions
import bigfloat as bf


class Benchmark:
    """A benchmark in the form of a function parameterized by distributions and constants."""

    def __init__(self,
                 benchmark: Callable,
                 distributions: Iterable = [],
                 constants: Iterable[float] = [],
                 exact_constants: Iterable[Callable] = [],
                 unary_ops: Iterable[Callable] = []):
        self.variables: List[ExactVariable] = [ExactVariable(*d) for d in distributions]
        self.constants: List[ExactConstant] = [ExactConstant(c) for c in constants]
        self.exact_constants: List[GenericExactConstant] = [GenericExactConstant(c) for c in exact_constants]
        self.unary_ops = [(lambda c: (lambda child: UnaryOperator(child, c)))(c) for c in unary_ops]
        self.benchmark_function = benchmark

    def benchmark(self):
        return self.benchmark_function(self.variables + self.constants + self.exact_constants + self.unary_ops)


# Misc benchmarks
simplest_test = Benchmark(
    benchmark=benchmark_functions.simplest_test,
    distributions=(),
    constants=(
        bf.exp2(100000, bf.RoundTowardZero + bf.precision(1)),  # big_const
    ),
    exact_constants=(
        lambda context: bf.exp(1, context),  # e
        bf.const_pi,  # pi
    ),
)

# CCA Benchmarks
sqrt_pi = Benchmark(
    benchmark=benchmark_functions.sqrt_pi,
    exact_constants=(
        bf.const_pi,  # pi
    ),
    unary_ops=(interval_sqrt, ),
)

log_pi = Benchmark(
    benchmark=benchmark_functions.log_pi,
    exact_constants=(
        bf.const_pi,  # pi
    ),
    unary_ops=(interval_log, ),
)

sin_e = Benchmark(
    benchmark=benchmark_functions.sin_e,
    exact_constants=(
        lambda context: bf.exp(1, context),  # e
    ),
    unary_ops=(interval_sin, ),
)

cos_e = Benchmark(
    benchmark=benchmark_functions.cos_e,
    exact_constants=(
        lambda context: bf.exp(1, context),  # e
    ),
    unary_ops=(interval_cos, ),
)

sin_sin_sin_1 = Benchmark(
    benchmark=benchmark_functions.sin_sin_sin_1,
    constants=(1,),
    unary_ops=(interval_sin, ),
)

cos_cos_cos_1 = Benchmark(
    benchmark=benchmark_functions.cos_cos_cos_1,
    constants=(1,),
    unary_ops=(interval_cos, ),
)

e_e_e = Benchmark(
    benchmark=benchmark_functions.e_e_e,
    exact_constants=(
        lambda context: bf.exp(1, context),  # e
    ),
    unary_ops=(interval_exp, ),
)

log_log_log_pi = Benchmark(
    benchmark=benchmark_functions.log_log_log_pi,
    exact_constants=(
        bf.const_pi,  # pi
    ),
    unary_ops=(interval_log, ),
)

log_log_log_e = Benchmark(
    benchmark=benchmark_functions.log_log_log_e,
    exact_constants=(
        lambda context: bf.exp(1, context),  # e
    ),
    unary_ops=(interval_log, ),
)


log_log_log_log_pi = Benchmark(
    benchmark=benchmark_functions.log_log_log_log_pi,
    exact_constants=(
        bf.const_pi,  # pi
    ),
    unary_ops=(interval_log, ),
)

log_log_log_log_e = Benchmark(
    benchmark=benchmark_functions.log_log_log_log_e,
    exact_constants=(
        lambda context: bf.exp(1, context),  # e
    ),
    unary_ops=(interval_log, ),
)


sin_10_50 = Benchmark(
    benchmark=benchmark_functions.sin_10_50,
    constants=(50, ),
    unary_ops=(interval_sin, interval_pow10, ),
)

cos_10_50 = Benchmark(
    benchmark=benchmark_functions.cos_10_50,
    constants=(50, ),
    unary_ops=(interval_cos, interval_pow10, ),
)


e_1000 = Benchmark(
    benchmark=benchmark_functions.e_1000,
    constants=(1000, ),
    unary_ops=(interval_exp, ),
)

arctan_10_50 = Benchmark(
    benchmark=benchmark_functions.arctan_10_50,
    constants=(50, ),
    unary_ops=(interval_atan, interval_pow10, ),
)

e_pi_sqrt_163 = Benchmark(
    benchmark=benchmark_functions.e_pi_sqrt_163,
    constants=(163, ),
    exact_constants=(
        bf.const_pi,  # pi
    ),
    unary_ops=(interval_exp, interval_sqrt, ),
)

many_roots = Benchmark(
    benchmark=benchmark_functions.many_roots,
    constants=(32, 5, 27, 1, 3, 9, 25),
    unary_ops=(interval_cuberoot, interval_fifthroot, ),
)

sin_log_sqrt = Benchmark(
    benchmark=benchmark_functions.sin_log_sqrt,
    constants=(3, 640320, 163),
    unary_ops=(interval_log, interval_sin, interval_sqrt, ),
)

logistic_map_1000_steps = Benchmark(
    benchmark=benchmark_functions.logistic_map_1000_steps,
    exact_constants=(
        bf.const_pi,  # pi
    ),
)

# FPBench benchmarks
carbon_gas = Benchmark(
    benchmark=benchmark_functions.carbon_gas,
    distributions=(
        (float(300.0)-float(0.01), float(300.0)+float(0.01)),  # var_T
        (float(0.401)-float(1e-06), float(0.401)+float(1e-06)),  # var_a
        (float(42.7e-06)-float(1e-10), float(42.7e-06)+float(1e-10)),  # var_b
        (float(995.0), float(1005.0)),  # var_N
        (float(0.1)-float(0.005), float(0.5)+float(0.005))  # var_V
    ),
    constants=(
        float(3.5e7),  # var_p
        float(1.3806503e-23)  # const_k
    ),
)


doppler1 = Benchmark(
    benchmark=benchmark_functions.doppler,
    distributions=(
        ((-100.0 - 0.0000001), (100.0 + 0.0000001)),  # var_u
        ((20.0 - 0.000000001), (20000.0 + 0.000000001)),  # var_v
        ((-30.0 - 0.000001), (50.0 + 0.000001)),  # var_T
    ),
)


doppler2 = Benchmark(
    benchmark=benchmark_functions.doppler,
    distributions=(
        ((-125.0 - 0.000000000001), (125.0 + 0.000000000001)),  # var_u
        ((15.0 - 0.001), (25000.0 + 0.001)),  # var_v
        ((-40.0 - 0.00001), (60.0 + 0.00001)),  # var_T
    ),
)


doppler3 = Benchmark(
    benchmark=benchmark_functions.doppler,
    distributions=(
        ((-30.0 - 0.0001), (120.0 + 0.0001)),  # var_u
        ((320.0 - 0.00001), (20300.0 + 0.00001)),  # var_v
        ((-50.0 - 0.000000001), (30.0 + 0.000000001)),  # var_T
    ),
)


jet = Benchmark(
    benchmark=benchmark_functions.jet,
    distributions=(
        (-5.0, 5.0),  # var_x1
        (-20.0, 5.0),  # var_x2
    ),
)


predator_prey = Benchmark(
    benchmark=benchmark_functions.predator_prey,
    distributions=(
        (0.1, 0.3),  # var_x
    ),
    constants=(
        4,  # const_r
        1.11,  # const_k
    ),
)


rigid_body1 = Benchmark(
    benchmark=benchmark_functions.rigid_body1,
    distributions=(
        ((-15.0 - 1e-08), (15.0 + 1e-08)),  # var_x1
        ((-15.0 - 1e-08), (15.0 + 1e-08)),  # var_x2
        ((-15.0 - 1e-08), (15.0 + 1e-08))   # var_x3
    ),
)


rigid_body2 = Benchmark(
    benchmark=benchmark_functions.rigid_body2,
    distributions=(
        ((-15.0 - 1e-08), (15.0 + 1e-08)),  # var_x1
        ((-15.0 - 1e-08), (15.0 + 1e-08)),  # var_x2
        ((-15.0 - 1e-08), (15.0 + 1e-08))   # var_x3
    ),
)


sine = Benchmark(
    benchmark=benchmark_functions.sine,
    distributions=((-1.57079632679, 1.57079632679), ),  # x
)


sine_order3 = Benchmark(
    benchmark=benchmark_functions.sine_order3,
    distributions=(
        (-2.0, 2.0),  # x
        ),
    constants=(
        0.954929658551372,  # c1
        0.12900613773279798,  # c2
    ),
)


sqroot = Benchmark(
    benchmark=benchmark_functions.sqroot,
    distributions=(
        (0.0, 1.0),  # x
    ),
    constants=(
        0.5,        # c1
        0.125,      # c2
        0.0625,     # c3
        0.0390625,  # c4
    ),
)


turbine1 = Benchmark(
    benchmark=benchmark_functions.turbine1,
    distributions=(
        ((-4.5 - 0.0000001), (-0.3 + 0.0000001)),  # var_v
        ((0.4 - 0.000000000001), (0.9 + 0.000000000001)),  # var_w
        ((3.8 - 0.00000001), (7.8 + 0.00000001))   # var_r
    ),
    constants=(
        0.125,  # c1
        4.5,    # c2
    ),
)


turbine2 = Benchmark(
    benchmark=benchmark_functions.turbine2,
    distributions=(
        ((-4.5 - 0.0000001), (-0.3 + 0.0000001)),  # var_v
        ((0.4 - 0.000000000001), (0.9 + 0.000000000001)),  # var_w
        ((3.8 - 0.00000001), (7.8 + 0.00000001))   # var_r
    ),
    constants=(
        0.5,  # c1
        2.5,  # c2
    ),
)


turbine3 = Benchmark(
    benchmark=benchmark_functions.turbine3,
    distributions=(
        ((-4.5 - 0.0000001), (-0.3 + 0.0000001)),  # var_v
        ((0.4 - 0.000000000001), (0.9 + 0.000000000001)),  # var_w
        ((3.8 - 0.00000001), (7.8 + 0.00000001))   # var_r
    ),
    constants=(
        0.125,  # c1
        0.5,    # c2
    ),
)


verlhulst = Benchmark(
    benchmark=benchmark_functions.verlhulst,
    distributions=(((0.1 - 1e-06), (0.3 + 1e-06)), ),  # x
    constants=(
        4.0,      # r
        1.11,     # k
    ),
)

jesse_benchmarks = {
    "simplest test": simplest_test,
}

fpbench_benchmarks = {
    "carbon gas": carbon_gas,
    "doppler1": doppler1,
    "doppler2": doppler2,
    "doppler3": doppler3,
    "jet": jet,
    "predator prey": predator_prey,
    "rigidbody1": rigid_body1,
    "rigidbody2": rigid_body2,
    "sine": sine,
    "sine_order3": sine_order3,
    "sqroot": sqroot,
    "turbine1": turbine1,
    "turbine2": turbine2,
    "turbine3": turbine3,
    "verlhulst": verlhulst
}

cca_benchmarks = {
    "sqrt_pi": sqrt_pi,
    "log_pi": log_pi,
    "sin_e": sin_e,
    "cos_e": cos_e,
    "sin_sin_sin_1": sin_sin_sin_1,
    "cos_cos_cos_1": cos_cos_cos_1,
    "e_e_e": e_e_e,
    "log_log_log_pi": log_log_log_pi,
    "log_log_log_e": log_log_log_e,
    "log_log_log_log_pi": log_log_log_log_pi,
    "log_log_log_log_e": log_log_log_log_e,
    "sin_10_50": sin_10_50,
    "cos_10_50": cos_10_50,
    "e_1000": e_1000,
    "arctan_10_50": arctan_10_50,
    "e_pi_sqrt_163": e_pi_sqrt_163,
    "many_roots": many_roots,
    "sin_log_sqrt": sin_log_sqrt,
    # "logistic_map_1000_steps": logistic_map_1000_steps,
}
