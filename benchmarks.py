from typing import Iterable, Callable, List

from exact_real_program import ExactVariable, ExactConstant, GenericExactConstant
import benchmark_functions
import bigfloat as bf


class Benchmark:
    """A benchmark in the form of a function parameterized by distributions and constants."""

    def __init__(self,
                 benchmark: Callable,
                 distributions: Iterable = [],
                 constants: Iterable[float] = [],
                 exact_constants: Iterable[Callable] = []):
        self.variables: List[ExactVariable] = [ExactVariable(*d) for d in distributions]
        self.constants: List[ExactConstant] = [ExactConstant(c) for c in constants]
        self.exact_constants: List[GenericExactConstant] = [GenericExactConstant(c) for c in exact_constants]
        self.benchmark_function = benchmark

    def benchmark(self):
        return self.benchmark_function(self.variables + self.constants + self.exact_constants)


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


all_benchmarks = {
    "simplest test": simplest_test,
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
