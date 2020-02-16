from typing import List


error_bounds: List = [
    # '0.' + '0' * 10 + '1',
                      '0.' + '0' * 10000 + '1',
                    #   '0.' + '0' * 1000 + '1'
                      ]

filename = "results/junk"
uniform_lines = "\n".join([f"python3 main.py --label single_bound_{i} --error_bound {error_bounds[i]} --use_ad" for i in range(len(error_bounds))])

non_uniform_lines = "\n".join([f"python3 main.py --label single_bound_uniform_{i} --error_bound {error_bounds[i]}" for i in range(len(error_bounds))])


bash_file = f"""echo Running with AD
{uniform_lines}
echo "\n"Running without AD
{non_uniform_lines}
"""

with open("generated_main.sh", "w") as f:
    f.write(bash_file)
