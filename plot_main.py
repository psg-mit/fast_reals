import pickle
import matplotlib.pyplot as plt
# import numpy as np


def load_results(filename_uniform: str, filename_nonuniform: str):
    with open(filename_uniform, 'rb') as f:
        data = pickle.load(f)
    with open(filename_nonuniform, 'rb') as f:
        ad_data = pickle.load(f)

    base_times, ad_times = data['time'], ad_data['time']
    import ipdb; ipdb.set_trace()
    print(base_times, ad_times)
    # plt.plot(log_error_bounds, base_times, 'b--', label='Baseline')
    # plt.plot(log_error_bounds, ad_times, 'r', label='With Derivatives')
    # plt.legend(loc='upper right')
    # plt.xlabel('Log Error Bounds')
    # plt.ylabel('Times')
    # plt.show()


load_results("single_bound_uniform_0.p", "single_bound_0.p")
