# import pickle
import matplotlib.pyplot as plt
# import numpy as np

with open("durations.txt") as f:
    timing_results = f.read()

iterations_times = [float(t.strip()) for t in timing_results.split('\n')[:-1]]

uniform_refinement = iterations_times[:29]
nonuniform_refinement = iterations_times[29:]

print(sum(uniform_refinement), sum(nonuniform_refinement))

# all_durations = []
# for i in range(5):
#     durations = pickle.load(open(f"durations_{i}.p", "rb" ))
#     all_durations.append(durations)

# all_durations_uniform = []
# for i in range(5):
#     durations = pickle.load(open(f"durations_uniform_{i}.p", "rb" ))
#     all_durations_uniform.append(durations)

# mean_durations = np.mean(np.array(all_durations), axis=0)
# var_durations = np.var(np.array(all_durations), axis=0)

# mean_durations_uniform = np.mean(np.array(all_durations_uniform), axis=0)
# var_durations_uniform = np.var(np.array(all_durations_uniform), axis=0)

# x = [i for i in range(1, 1000000, 10000)]
# plt.errorbar(x, mean_durations, yerr=var_durations, fmt='o', color='black',
#              ecolor='orange', elinewidth=3, capsize=0)
# plt.errorbar(x, mean_durations_uniform, yerr=var_durations_uniform, fmt='x', color='blue',
#              ecolor='red', elinewidth=3, capsize=0)
# plt.xlabel('x') #, font=18)
# plt.ylabel('Duration') #, font=18)
# plt.show()
