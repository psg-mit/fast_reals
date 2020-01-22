from typing import Iterable, List

import matplotlib.pyplot as plt
from tqdm import trange
import pickle
import numpy as np
import os
import re
import bigfloat as bf
from tap import Tap

from evaluate import evaluate, evaluate_using_derivatives
from benchmarks import all_benchmarks
from utils import time_wrap


def run_benchmarks(names: Iterable = None, filename: str = '', use_ad=False):
    """Run given Iterable of benchmark names if None, run all benchmarks. """
    if names is None:
        names = all_benchmarks.keys()
        print(names)
    ad = "_ad" if use_ad else ""

    name_data = {}
    for name in names:
        print(name)
        benchmark = all_benchmarks[name]
        exact_program, variables = benchmark.benchmark(), benchmark.variables
        init_prec = 50
        num_samples = 1
        error_bounds: List = [bf.BigFloat('0.' + '0' * 50000 + '1'),
                              bf.BigFloat('0.' + '0' * 100000 + '1'),
                              bf.BigFloat('0.' + '0' * 150000 + '1'),]
                            #   bf.BigFloat('0.' + '0' * 10000 + '1'),]
        refinements, times = [], []
        for error_bound in error_bounds:
            sample_refinements, sample_times = [], []
            for _ in trange(num_samples):
                [var.sample() for var in variables]
                if use_ad:
                    params = [exact_program, error_bound, [init_prec]*exact_program.subtree_size()]
                    time, num_refinements = time_wrap(evaluate_using_derivatives, params)
                else:
                    params = [exact_program, error_bound, False, init_prec]
                    time, (num_refinements, iterations) = time_wrap(evaluate, params)
                sample_refinements.append(num_refinements)
                sample_times.append(time)
            refinements.append(np.mean(sample_refinements))
            times.append(np.mean(sample_times))
        name_data[name] = {"refinements": refinements, "times": times}
        print(exact_program.full_string())

    if filename:
        # Prevent overwriting
        i = 0
        while os.path.exists(filename + ad + '.pkl'):
            if not i:
                filename += '_version' + str(i)
            else:
                filename = re.sub('[0-9]*$', '', filename) + str(i)
            i += 1

        if not os.path.exists(filename + ad + '.pkl'):
            filename += ad
        filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump((name_data, [float(bf.log10(error_bound)) for error_bound in error_bounds]), f)

        print('Wrote file', filename, '\n')
    return filename


def load_results(filename: str):
    with open(filename, 'rb') as f:
        name_data, log_error_bounds = pickle.load(f)
    if len(re.findall("_version", filename)) > 0:
        ad_filename = filename.strip(".pkl") + "_ad" + ".pkl"
    else:
        ad_filename = filename.strip(".pkl") + "_ad" + ".pkl"
    with open(ad_filename, 'rb') as f:
        ad_name_data, ad_log_error_bounds = pickle.load(f)
    fig, axs = plt.subplots(3, 5)
    fig.suptitle('Derivative', fontsize=26)
    times, refinements = [], []
    for k, (name, data) in enumerate(name_data.items()):
        base_counts, ad_counts = data['refinements'], ad_name_data[name]['refinements']
        base_times, ad_times = data['times'], ad_name_data[name]['times']
        base_seconds = [time.total_seconds() for time in base_times]
        ad_seconds = [time.total_seconds() for time in ad_times]
        i, j = k % 3, k % 5
        axs[i, j].plot(log_error_bounds, base_seconds, 'b--')
        axs[i, j].plot(log_error_bounds, ad_seconds, 'r')
        axs[i, j].legend(('Base', 'With derivatives'), loc='upper right')
        axs[i, j].set()
        axs[i, j].set_title(name, fontsize=18)
        axs[i, j].set(xlabel='Error Bound', ylabel='Time')

    #     axs[i, j].plot(np.log(error_bounds), base_counts, 'b--')
    #     axs[i, j].plot(np.log(error_bounds), ad_counts, 'r')
    #     axs[i, j].legend(('Base', 'With derivatives'), loc='upper right')
    #     axs[i, j].set(xlabel='Error Bound', ylabel='Refinement Steps')
    #     axs[i, j].set_title(name, fontsize=18)

        times.append(np.mean([(bas - ads) / bas * 100 for bas, ads in zip(base_seconds, ad_seconds)]))
        refinements.append(np.mean([(bas - ads) / bas * 100 for bas, ads in zip(base_counts, ad_counts)]))
    plot_filename = filename + '_plot' + '.png'
    plt.savefig(plot_filename)
    print('Wrote file', plot_filename)
    plt.show()

    # # print("time", round(np.mean(times), 2))
    # # print("refinements", round(np.mean(refinements), 2))


class ArgumentParser(Tap):
    filename: str
    use_ad: bool = False
    load_results: bool = False
    run: bool = True

if __name__ == '__main__':
    np.random.seed(0)

    args = ArgumentParser().parse_args()
    if args.run:
        filename = run_benchmarks(['simplest test'], args.filename, args.use_ad)
    else:
        filename = args.filename + '.pkl'

    if args.load_results:
        load_results(filename)
