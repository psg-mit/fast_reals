from typing import Iterable

import bigfloat as bf

from exact_real_program import ExactConstant, GenericExactConstant, ExactVariable
from evaluate import evaluate, evaluate_using_derivatives
from benchmarks import all_benchmarks


def run_benchmarks(names: Iterable = None):
    """Run given Iterable of benchmark names if None, run all benchmarks. """
    if names is None:
        names = all_benchmarks.keys()
        print(names)

    for name in names:
        print(name)
        benchmark = all_benchmarks[name]
        exact_program, variables = benchmark.benchmark(), benchmark.variables

        error_bound: float = 1e-20
        steps = evaluate(exact_program, error_bound, [30] * exact_program.subtree_size())
        steps = evaluate_using_derivatives(exact_program, error_bound, [30] * exact_program.subtree_size())
        print("It took", steps, "refinement steps to achieve the", error_bound, "error bound. \n")


if __name__ == '__main__':
    run_benchmarks(None)

    # # Tests for ExactVariable
    # bits = 10
    # a = ExactVariable(1, 3)
    # a.evaluate(bits)
    # print(a.lower, a.upper)
    # a.sample()
    # print(a.lower, a.upper)
    # a.evaluate(bits)
    # print(a.lower, a.upper)

    # Simple example
    # big_number = ExactConstant(1e100)
    # e = GenericExactConstant(lambda context: bf.exp(1, context))
    # pi = GenericExactConstant(bf.const_pi)
    # a = pi + big_number * e
    # program = a

    # error_bound: int = 1
    # steps = evaluate_using_derivatives(a, error_bound, [30] * a.subtree_size())
    # print("It took", steps, "refinement steps to achieve the", error_bound, "error bound. \n")
    # program.print()

    # print(pi.grad())
    # print(e.grad())
