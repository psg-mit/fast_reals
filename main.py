from typing import Iterable, List

import matplotlib.pyplot as plt
from tqdm import trange
import pickle
import numpy as np

from exact_real_program import ExactConstant, GenericExactConstant, ExactVariable
from evaluate import evaluate, evaluate_using_derivatives
from benchmarks import all_benchmarks


def run_benchmarks(outfile: str, names: Iterable = None):
    """Run given Iterable of benchmark names if None, run all benchmarks. """
    if names is None:
        names = all_benchmarks.keys()
        print(names)

    name_step_counts = {}
    for name in names:
        print(name)
        benchmark = all_benchmarks[name]
        exact_program, variables = benchmark.benchmark(), benchmark.variables
        init_prec = 10
        num_samples = 100
        error_bounds: List[float] = [1e-30, 1e-25, 1e-20, 1e-15, 1e-10]
        step_counts, base_counts = [], []
        for error_bound in error_bounds:
            steps_for_error_bound = []
            normal_steps = []
            for _ in trange(num_samples):
                [var.sample() for var in variables]
                normal_steps.append(evaluate(exact_program, error_bound,
                                             ad=False, initial_precision=init_prec))
                steps_for_error_bound.append(evaluate_using_derivatives(exact_program, error_bound,
                                                                        [init_prec] * exact_program.subtree_size()))
            base_counts.append(np.mean(np.array(normal_steps)))
            step_counts.append(np.mean(np.array(steps_for_error_bound)))

            # print("It took", steps, "refinement steps to achieve the", error_bound, "error bound. \n")
        name_step_counts[name] = base_counts, step_counts
    with open(outfile, 'wb') as f:
        pickle.dump((name_step_counts, error_bounds), f)


def load_results(f: str):
    with open(f, 'rb') as out:
        name_step_counts, error_bounds = pickle.load(out)
    print(error_bounds)
    print(name_step_counts)
    for name, (base_counts, deriv_counts) in name_step_counts.items():
        plt.plot(np.log(np.array(error_bounds)), base_counts, 'b--')
        plt.plot(np.log(np.array(error_bounds)), deriv_counts, 'r')
        plt.legend(('Base', 'With derivatives'), loc='upper right')
        plt.xlabel('Error Bound', fontsize=14)
        plt.ylabel('Refinement Steps', fontsize=14)
        plt.title(name)
        plt.show()


if __name__ == '__main__':
    outfile = "init10_samples100.pkl"
    # outfile = "test"
    run_benchmarks(outfile, None)
    load_results(outfile)

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
