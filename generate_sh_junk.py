from typing import List


def C(t):
    if t == 0:
        return 50
    c = C(t - 1) + 50 * 1.25**t
    return c


def D(t):
    if t == 0:
        return 50
    d = D(t - 1) + 50 * 1.33**t
    return d


def make_argparsable(l: List):
    return str(l).replace(',', "")[1:-1]


cs = [int(C(t)) for t in range(29)]
ds = [int(D(t)) for t in range(24)]

uniform_lines = "\n".join([f"python3 main.py --uniform --label uniform_{i} --precisions {make_argparsable(cs[i: i+1]*5)}" for i in range(len(cs))])

non_uniform_lines = "\n".join([f"python3 main.py --label {i} --precisions {make_argparsable([ds[i], cs[i], ds[i], cs[i], ds[i]])}" for i in range(len(ds))])


bash_file = f"""rm durations.txt
echo Uniform
{uniform_lines}
echo "\n"Nonuniform
{non_uniform_lines}
"""

with open("generated_main_junk.sh", "w") as f:
    f.write(bash_file)
