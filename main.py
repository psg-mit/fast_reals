from typing import Iterable, List, Dict

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import bigfloat as bf

from evaluate import evaluate, evaluate_using_derivatives
from benchmarks import all_benchmarks
from utils import time_wrap, multiprocess


@multiprocess
def run_benchmark(log_error_bound: int,
                  name: str,
                  use_ad: bool = False):
    benchmark = all_benchmarks[name]
    exact_program, variables = benchmark.benchmark(), benchmark.variables
    init_prec = 50

    context = bf.RoundTowardZero + bf.precision(100000)
    assert log_error_bound < 0
    error_bound = bf.BigFloat(f'0.{"0"*(-log_error_bound-1)}1', context)
    [var.sample() for var in variables]
    if use_ad:
        params = [exact_program, error_bound, [init_prec]*exact_program.subtree_size()]
        time, (num_refinements, precision_configuration) = time_wrap(evaluate_using_derivatives, params)
    else:
        params = [exact_program, error_bound, False, init_prec]
        time, (num_refinements, precision_configuration) = time_wrap(evaluate, params)
    data = {
        "time": time.total_seconds(),
        "num_refinements": num_refinements,
        "error_bound": float(error_bound),
        "last_prec_config": precision_configuration,
        }
    print(exact_program.full_string())
    return data


def run_benchmarks(log_error_bounds: List[int], benchmark_names: List[str]):
    benchmark_results = defaultdict(list)
    for benchmark_name in benchmark_names:
        for log_error_bound in log_error_bounds:
            benchmark_results[f'{benchmark_name}_ad'].append(run_benchmark(log_error_bound, benchmark_name, True))
            benchmark_results[f'{benchmark_name}_uniform'].append(run_benchmark(log_error_bound, benchmark_name, False))
    return benchmark_results


if __name__ == '__main__':
    np.random.seed(0)

    benchmarks = ['simplest test']
    log10_error_bounds: List[int] = [-10000]
    results = run_benchmarks(log10_error_bounds, benchmarks)
