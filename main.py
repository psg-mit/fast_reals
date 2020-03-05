from typing import Iterable, List, Dict, Optional, Set, Callable

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import bigfloat as bf

from evaluate import evaluate, evaluate_using_derivatives
from benchmarks import fpbench_benchmarks, cca_benchmarks, jesse_benchmarks, Benchmark
from utils import time_wrap, multiprocess


@multiprocess
def run_benchmark(log_error_bound: int,
                  benchmark: Benchmark,
                  configuaration_increment: Optional[Callable] = None,
                  use_ad: bool = False):
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
        params = [exact_program, error_bound, configuaration_increment, True, init_prec]
        time, (num_refinements, precision_configuration) = time_wrap(evaluate, params)
    data = {
        "time": time.total_seconds(),
        "num_refinements": num_refinements,
        "log_error_bound": float(log_error_bound),
        "last_prec_config": precision_configuration,
        }
    # print(exact_program.full_string())
    return data


def run_benchmarks(log_error_bounds: List[int],
                   benchmarks: Dict[str, Benchmark],
                   benchmarks_to_run: Optional[Set[str]] = None):
    benchmark_queue = {k: v for k, v in benchmarks.items() if k in benchmarks_to_run} if benchmarks_to_run else benchmarks
    benchmark_results = defaultdict(list)
    for benchmark_name, benchmark in benchmark_queue.items():
        print(benchmark_name)
        for log_error_bound in log_error_bounds:
            # Warmup run!
            # benchmark_results[f'{benchmark_name} ad'].append(run_benchmark(log_error_bound, benchmark, use_ad=True))
            # benchmark_results[f'{benchmark_name} linear'].append(run_benchmark(log_error_bound, benchmark, use_ad=True))
            # benchmark_results[f'{benchmark_name} exp'].append(run_benchmark(log_error_bound, benchmark, use_ad=True))
            # benchmark_results[f'{benchmark_name} superexp'].append(run_benchmark(log_error_bound, benchmark, use_ad=True))
            # actual runs
            benchmark_results[f'{benchmark_name} ad'].append(run_benchmark(log_error_bound, benchmark, use_ad=True))
            benchmark_results[f'{benchmark_name} linear'].append(run_benchmark(log_error_bound, benchmark, lambda x: int(50 * x), False))
            benchmark_results[f'{benchmark_name} exp'].append(run_benchmark(log_error_bound, benchmark, lambda x: int(50 * 1.25**x), False))
            benchmark_results[f'{benchmark_name} superexp'].append(run_benchmark(log_error_bound, benchmark, lambda x: int(50 * 1.25**(1.25**x)), False))
    return benchmark_results


def plot_results(results):
    times = {}
    for auto_diff, linear, exp, superexp in zip(list(results.keys())[::4], list(results.keys())[1::4], list(results.keys())[2::4], list(results.keys())[3::4]):
        log_error_bounds = []
        adt, lt, et, st = [], [], [], []
        for ad, l, e, s in zip(results[auto_diff], results[linear], results[exp], results[superexp]):
            log_error_bounds.append(l['log_error_bound'])
            adt.append(ad['time'])
            lt.append(l['time'])
            et.append(e['time'])
            st.append(s['time'])
        benchmark_name = exp[:-3]
        print(benchmark_name)
        log_error_bounds = log_error_bounds
        times[benchmark_name] = (adt, lt, et, st)
    fig, axs = plt.subplots(5, 4)
    fig.suptitle('Comparison of Precision Schedules', fontsize=26)
    for k, (name, (ad, linear, exp, superexp)) in enumerate(times.items()):
        i, j = k // 4, k % 4
        axs[i, j].plot(log_error_bounds, ad, 'o--', color='orange')
        axs[i, j].plot(log_error_bounds, linear, 'g:')
        axs[i, j].plot(log_error_bounds, exp, 'b--')
        axs[i, j].plot(log_error_bounds, superexp, 'r')
        axs[i, j].legend(('Ad Exp', 'Linear', 'Exp', 'Super-exp'), loc='upper right')
        axs[i, j].set()
        axs[i, j].set_title(name, fontsize=18)
        axs[i, j].set(xlabel='Log10 Error Bound', ylabel='Time')
    plt.show()


if __name__ == '__main__':
    import pickle

    def run():
        np.random.seed(0)

        benchmarks_name = None
        log10_error_bounds: List[int] = [-5000,
                                         -10000,
                                         -15000,
                                         -20000]
        results = run_benchmarks(log10_error_bounds, cca_benchmarks, benchmarks_name)

        pickle.dump(results, open("cca_test.p", "wb"))

    def load():
        results = pickle.load(open("cca_test.p", "rb"))
        plot_results(results)

    run()
    load()

    # for a, b in zip(list(x.keys())[::2], list(x.keys())[1::2]):
    #     print(f"{a[:-3]} {round((x[b][0]['time'] / x[a][0]['time'] - 1) * 100, 3)}")
    # for k in x.keys():
    #     print(k, x[k][0])
