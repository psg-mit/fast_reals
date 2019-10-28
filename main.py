from typing import Iterable, List

import matplotlib.pyplot as plt
from tqdm import trange
import pickle
import numpy as np
import os
import re

from exact_real_program import ExactConstant, GenericExactConstant, ExactVariable
from evaluate import evaluate, evaluate_using_derivatives
from benchmarks import all_benchmarks
from utils import time_wrap


def run_benchmarks(names: Iterable = None, filename: str = ''):
    """Run given Iterable of benchmark names if None, run all benchmarks. """
    if names is None:
        names = all_benchmarks.keys()
        print(names)

    name_data = {}
    for name in names:
        print(name)
        benchmark = all_benchmarks[name]
        exact_program, variables = benchmark.benchmark(), benchmark.variables
        init_prec = 10
        num_samples = 1
        error_bounds: List[float] = [1e-300]#, 1e-25, 1e-20, 1e-15, 1e-10]
        step_counts, base_counts = [], []
        ad_times, base_times = [], []
        for error_bound in error_bounds:
            normal_steps, normal_times = [], []
            with_ad_steps, with_ad_times = [], []
            benchmark_iters = []
            for _ in trange(num_samples):
                [var.sample() for var in variables]
                time, (no_ad_steps, iterations) = time_wrap(evaluate, [exact_program, error_bound, False, init_prec])
                normal_steps.append(no_ad_steps)
                normal_times.append(time)
                # benchmark_iters.append(iterations)
                time, with_ad_step = time_wrap(evaluate_using_derivatives,
                                                [exact_program, error_bound, [init_prec]*exact_program.subtree_size()])
                with_ad_steps.append(with_ad_step)
                with_ad_times.append(time)

                plt.plot([i for i in range(len(iterations))], [t.total_seconds() for t in iterations])
                plt.show()
            base_counts.append(np.mean(normal_steps))
            step_counts.append(np.mean(with_ad_steps))
            base_times.append(np.mean(normal_times))
            ad_times.append(np.mean(with_ad_times))
        name_data[name] = {"refinements": (base_counts, step_counts), "times": (base_times, ad_times)}
        print(exact_program.full_string())

    if filename:
        # Prevent overwriting
        i = 0
        while os.path.exists(filename + '.pkl'):
            if not i:
                filename += '_version' + str(i)
            else:
                filename = re.sub('[0-9]*$', '', filename) + str(i)
            i += 1

        filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump((name_data, error_bounds), f)

        print('Wrote file', filename, '\n')
    return filename


def load_results(f: str):
    with open(f, 'rb') as f:
        name_data, error_bounds = pickle.load(f)
    times, refinements = [], []
    for name, data in name_data.items():
        # if name != "verlhulst":
        base_counts, ad_counts = data["refinements"]
        base_times, ad_times = data["times"]
        base_seconds = [time.total_seconds() for time in base_times]
        ad_seconds = [time.total_seconds() for time in ad_times]

        # plt.plot(np.log(error_bounds), base_seconds, 'b--')
        # plt.plot(np.log(error_bounds), ad_seconds, 'r')
        # plt.legend(('Base', 'With derivatives'), loc='upper right')
        # plt.xlabel('Error Bound', fontsize=14)
        # plt.ylabel('Time', fontsize=14)
        # plt.title(name)
        # plt.show()

        # plt.plot(np.log(error_bounds), base_counts, 'b--')
        # plt.plot(np.log(error_bounds), ad_counts, 'r')
        # plt.legend(('Base', 'With derivatives'), loc='upper right')
        # plt.xlabel('Error Bound', fontsize=14)
        # plt.ylabel('Refinement Steps', fontsize=14)
        # plt.title(name)
        # plt.show()

        times.append(np.mean([(bas - ads) / bas * 100 for bas, ads in zip(base_seconds, ad_seconds)]))
        refinements.append(np.mean([(bas - ads) / bas * 100 for bas, ads in zip(base_counts, ad_counts)]))
    print("time", round(np.mean(times), 2))
    print("refinements", round(np.mean(refinements), 2))


if __name__ == '__main__':
    np.random.seed(0)
    filename = "results/init10_samples100_logsqr_deriv"
    # outfile = "test"
    # run_benchmarks(None)
    filename = run_benchmarks(None)#, filename)
    # if filename:
        # load_results(filename + ".pkl")

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
    # import bigfloat as bf
    # big_number = ExactConstant(1e100)
    # e = GenericExactConstant(lambda context: bf.exp(1, context))
    # pi = GenericExactConstant(bf.const_pi)
    # a = pi + big_number * e
    # program = a

    # error_bound: int = 1e-10
    # steps = evaluate_using_derivatives(a, error_bound, [30] * a.subtree_size())
    # # print("It took", steps, "refinement steps to achieve the", error_bound, "error bound. \n")
    # program.print()

    # print(pi.grad())
    # print(e.grad())
