from typing import Iterable, Callable, List

from exact_real_program import ExactVariable, ExactConstant
import benchmark_functions


class Benchmark:
    """A benchmark in the form of a function parameterized by distributions and constants."""

    def __init__(self,
                 distributions: Iterable,
                 constants: Iterable[float],
                 benchmark: Callable):
        self.variables: List[ExactVariable] = [ExactVariable(*d) for d in distributions]
        self.constants: List[ExactConstant] = [ExactConstant(c) for c in constants]
        self.benchmark_function = benchmark

    def benchmark(self):
        return self.benchmark_function(self.variables + self.constants)


carbon_gas = Benchmark(
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
    benchmark=benchmark_functions.carbon_gas
)


doppler1 = Benchmark(
    distributions=(
        ((-100.0 - 0.0000001), (100.0 + 0.0000001)),  # var_u
        ((20.0 - 0.000000001), (20000.0 + 0.000000001)),  # var_v
        ((-30.0 - 0.000001), (50.0 + 0.000001)),  # var_T
    ),
    constants=(),
    benchmark=benchmark_functions.doppler
)


doppler2 = Benchmark(
    distributions=(
        ((-125.0 - 0.000000000001), (125.0 + 0.000000000001)),  # var_u
        ((15.0 - 0.001), (25000.0 + 0.001)),  # var_v
        ((-40.0 - 0.00001), (60.0 + 0.00001)),  # var_T
    ),
    constants=(),
    benchmark=benchmark_functions.doppler
)


doppler3 = Benchmark(
    distributions=(
        ((-30.0 - 0.0001), (120.0 + 0.0001)),  # var_u
        ((320.0 - 0.00001), (20300.0 + 0.00001)),  # var_v
        ((-50.0 - 0.000000001), (30.0 + 0.000000001)),  # var_T
    ),
    constants=(),
    benchmark=benchmark_functions.doppler
)


jet = Benchmark(
    distributions=(
        (-5.0, 5.0),  # var_x1
        (-20.0, 5.0),  # var_x2
    ),
    constants=(),
    benchmark=benchmark_functions.jet
)


predator_prey = Benchmark(
    distributions=(
        (0.1, 0.3),  # var_x
    ),
    constants=(
        4,  # const_r
        1.11,  # const_k
    ),
    benchmark=benchmark_functions.predator_prey
)


rigid_body1 = Benchmark(
    distributions=(
        ((-15.0 - 1e-08), (15.0 + 1e-08)),  # var_x1
        ((-15.0 - 1e-08), (15.0 + 1e-08)),  # var_x2
        ((-15.0 - 1e-08), (15.0 + 1e-08))   # var_x3
    ),
    constants=(
    ),
    benchmark=benchmark_functions.rigid_body1
)


rigid_body2 = Benchmark(
    distributions=(
        ((-15.0 - 1e-08), (15.0 + 1e-08)),  # var_x1
        ((-15.0 - 1e-08), (15.0 + 1e-08)),  # var_x2
        ((-15.0 - 1e-08), (15.0 + 1e-08))   # var_x3
    ),
    constants=(
    ),
    benchmark=benchmark_functions.rigid_body2
)


sine = Benchmark(
    distributions=((-1.57079632679, 1.57079632679), ),  # x
    constants=(),
    benchmark=benchmark_functions.sine
)


sine_order3 = Benchmark(
    distributions=((-2.0, 2.0), ),  # x
    constants=(
        0.954929658551372,  # c1
        0.12900613773279798,  # c2
    ),
    benchmark=benchmark_functions.sine_order3
)


sqroot = Benchmark(
    distributions=((0.0, 1.0), ),  # x
    constants=(
        0.5,        # c1
        0.125,      # c2
        0.0625,     # c3
        0.0390625,  # c4
    ),
    benchmark=benchmark_functions.sqroot
)


turbine1 = Benchmark(
    distributions=(
        ((-4.5 - 0.0000001), (-0.3 + 0.0000001)),  # var_v
        ((0.4 - 0.000000000001), (0.9 + 0.000000000001)),  # var_w
        ((3.8 - 0.00000001), (7.8 + 0.00000001))   # var_r
    ),
    constants=(
        0.125,  # c1
        4.5,    # c2
    ),
    benchmark=benchmark_functions.turbine1
)


turbine2 = Benchmark(
    distributions=(
        ((-4.5 - 0.0000001), (-0.3 + 0.0000001)),  # var_v
        ((0.4 - 0.000000000001), (0.9 + 0.000000000001)),  # var_w
        ((3.8 - 0.00000001), (7.8 + 0.00000001))   # var_r
    ),
    constants=(
        0.5,  # c1
        2.5,  # c2
    ),
    benchmark=benchmark_functions.turbine2
)


turbine3 = Benchmark(
    distributions=(
        ((-4.5 - 0.0000001), (-0.3 + 0.0000001)),  # var_v
        ((0.4 - 0.000000000001), (0.9 + 0.000000000001)),  # var_w
        ((3.8 - 0.00000001), (7.8 + 0.00000001))   # var_r
    ),
    constants=(
        0.125,  # c1
        0.5,    # c2
    ),
    benchmark=benchmark_functions.turbine3
)


verlhulst = Benchmark(
    distributions=(((0.1 - 1e-06), (0.3 + 1e-06)), ),  # x
    constants=(
        4.0,      # r
        1.11,     # k
    ),
    benchmark=benchmark_functions.verlhulst
)


all_benchmarks = {
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
