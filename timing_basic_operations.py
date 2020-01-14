from timeit import time
import bigfloat as bf
import gmpy2 as gp
import numpy as np
import matplotlib.pyplot as plt
from tap import Tap
import sys


def compute_timings(i):
    bf_operations = [bf.add, bf.mul, bf.sub, bf.div]
    gp_operations = [gp.add, gp.mul, gp.sub, gp.div]
    labels = ["add", "mul", "sub", "div"]

    use_bf = False
    # for label, operation in zip(labels, bf_operations if use_bf else gp_operations):
    operation = bf_operations[i] if use_bf else gp_operations[i]
    label = labels[i]
    precisions, times = [], []
    for i in range(3, 18):
        p = int(2**i)
        precisions.append(p)
        if use_bf:
            context = bf.precision(p) + bf.RoundTowardZero
            start = time.time()
            pi = bf.const_pi(context)
            e = bf.exp(1, context)
            a = operation(e, pi, context)
            end = time.time()
        else:
            gp.set_context(gp.context(precision=p))
            start = time.time()
            pi = gp.const_pi()
            e = gp.exp(1)
            a = operation(e, pi)
            end = time.time()
        times.append((end-start))
        # plt.plot(precisions, times, label=label)
        # plt.legend()
        # plt.xlabel('Precision')
        # plt.ylabel('Time')
    return precisions, times


def gmpy2_results():
    precisions = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    all_times = [[0.0001704692840576172, 2.6226043701171875e-05, 1.9788742065429688e-05, 4.100799560546875e-05, 0.0001068115234375, 6.175041198730469e-05, 0.0001742839813232422, 0.00023245811462402344, 0.00038814544677734375, 0.0007314682006835938, 0.0017549991607666016, 0.0041196346282958984, 0.0033495426177978516, 0.010641336441040039, 0.029802560806274414],
    [5.1021575927734375e-05, 9.5367431640625e-06, 8.58306884765625e-06, 1.2874603271484375e-05, 3.0040740966796875e-05, 3.0279159545898438e-05, 4.76837158203125e-05, 8.606910705566406e-05, 0.00019550323486328125, 0.0005719661712646484, 0.0012898445129394531, 0.004693746566772461, 0.0034704208374023438, 0.009855270385742188, 0.02844071388244629],
    [0.00010251998901367188, 1.430511474609375e-05, 1.2159347534179688e-05, 2.0503997802734375e-05, 5.841255187988281e-05, 4.2438507080078125e-05, 5.7220458984375e-05, 9.417533874511719e-05, 0.00026726722717285156, 0.0005404949188232422, 0.0014770030975341797, 0.004302024841308594, 0.003881692886352539, 0.010071992874145508, 0.030391454696655273],
    [0.00011777877807617188, 1.7404556274414062e-05, 1.7404556274414062e-05, 2.4080276489257812e-05, 6.008148193359375e-05, 5.078315734863281e-05, 6.4849853515625e-05, 0.00011873245239257812, 0.0002815723419189453, 0.0005567073822021484, 0.0013151168823242188, 0.004609346389770508, 0.003298521041870117, 0.011148929595947266, 0.02996516227722168]]
    labels = ["add", "mul", "sub", "div"]
    for times, label in zip(all_times, labels):
        plt.plot(precisions, times, label=label)
        plt.legend()
        plt.xlabel('Precision')
        plt.ylabel('Time')
        plt.title('gmpy2')
    plt.show()


def bigfloat_results():
    precisions = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    all_times = [[0.00020551681518554688, 0.00010371208190917969, 9.632110595703125e-05, 9.870529174804688e-05, 0.00011658668518066406, 0.00011968612670898438, 0.00019359588623046875, 0.0001785755157470703, 0.00028705596923828125, 0.0005626678466796875, 0.0013585090637207031, 0.00400853157043457, 0.0031507015228271484, 0.009463787078857422, 0.0257875919342041],
    [0.0002124309539794922, 0.00012254714965820312, 9.489059448242188e-05, 9.894371032714844e-05, 0.00013375282287597656, 0.00012230873107910156, 0.0001385211944580078, 0.00017714500427246094, 0.00030422210693359375, 0.0005593299865722656, 0.0017592906951904297, 0.004185199737548828, 0.0035414695739746094, 0.009912490844726562, 0.02675318717956543],
    [0.0002086162567138672, 0.00011777877807617188, 0.00010156631469726562, 0.0001049041748046875, 0.000125885009765625, 0.0001251697540283203, 0.00014638900756835938, 0.00018405914306640625, 0.00029015541076660156, 0.0006339550018310547, 0.0014774799346923828, 0.003882169723510742, 0.003314495086669922, 0.009437799453735352, 0.027207136154174805],
    [0.0002186298370361328, 0.00010609626770019531, 9.894371032714844e-05, 0.00010275840759277344, 0.00015783309936523438, 0.00012254714965820312, 0.0001513957977294922, 0.00018835067749023438, 0.0002903938293457031, 0.0005626678466796875, 0.0015044212341308594, 0.004424333572387695, 0.0034782886505126953, 0.009825468063354492, 0.02861332893371582]]

    labels = ["add", "mul", "sub", "div"]
    for times, label in zip(all_times, labels):
        plt.plot(precisions, times, label=label)
        plt.legend()
        plt.xlabel('Precision')
        plt.ylabel('Time')
        plt.title('bigfloat')
    plt.show()


def load_file(filename: str):
    with open(filename, 'r') as f:
        text = f.read()
    op_texts = text.split("\n\n***===***\n")
    labels = ["add", "sub", "mul", "div"]
    for op_text, label in zip(op_texts, labels):
        times, precisions = [], []
        for line in op_text.split("\n"):
            tokens = line.split()
            if tokens:
                times.append(float(tokens[0]))
                precisions.append(int(tokens[-1]))
        label = label
        plt.plot(precisions, times, label=label)
        plt.legend()
        plt.xlabel('Precision')
        plt.ylabel('Time')
        plt.title('MPFR C++')
    plt.show()


if __name__ == "__main__":
    # p, t = compute_timings(3)
    # print(t)
    bigfloat_results()
    gmpy2_results()
    class ArgParse(Tap):
        filename: str
    args = ArgParse().parse_args()
    load_file(args.filename)
# for label, operation in zip(labels, operations):
#     precision = 10
#     precisions = []
#     ratios = []
#     ratio = 0
#     print(label)
#     while ratio < 2:
#         test_precisions = [precision, 2 * precision]
#         times = []
#         for p in test_precisions:
#             context = bf.precision(p) + bf.RoundTowardZero
#             start = time.time()
#             pi = bf.const_pi(context)
#             e = bf.exp(1, context)
#             a = operation(e, pi, context)
#             end = time.time()
#             ts.append(end - start)
#             times.append(sum(ts) / len(ts))
#         ratio = times[1] / times[0]
#         ratios.append(ratio)
#         precisions.append(precision)
#         precision *= 2
#         print(precision)

#     print("Precision at which doubling precision takes twice the time for", label, "is:", precision)

#     plt.plot(precisions, ratios, label=label)
#     plt.legend()
#     plt.xlabel('Precision')
#     plt.ylabel('Ratio between time at precision at 2p and p')
# plt.show()

# 1) super high precision
# 2) throw out overhead
# 3) performance estimation is too hard (maybe some surrogate method)
