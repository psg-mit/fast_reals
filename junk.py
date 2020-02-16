import bigfloat as bf
from tap import Tap
from time import time


def error(precs, a):
    context100000 = bf.RoundTowardZero + bf.precision(1000000)

    contexts = [bf.RoundTowardZero + bf.precision(prec) for prec in precs]

    start = time()
    approx = bf.add(bf.const_pi(contexts[0]),
                    bf.mul(bf.exp2(a, contexts[1]),
                           bf.exp(1, contexts[2]),
                           contexts[3]),
                    contexts[4])
    duration = time() - start
    exact = bf.add(bf.const_pi(context100000),
                   bf.mul(bf.exp2(100, context100000),
                          bf.exp(1, context100000),
                          context100000),
                   context100000)
    return bf.abs(bf.sub(approx, exact, context100000), context100000), duration


class Arguments(Tap):
    uniform: bool = False


if __name__ == '__main__':
    args = Arguments().parse_args()

    a = 100000
    x = 100000
    # precs = pi, 2^a, e, mul, add
    if args.uniform:
        error, duration = error([a+x]*5, a)
        print(duration)
        # print(error)
        with open('f', 'w') as f:
            f.write(str(error))
    else:
        error, duration = error([x, x, a+x, a+x, a+x], a)
        print(duration)
        # print(error)
        with open('g', 'w') as f:
            f.write(str(error))
