import bigfloat as bf

from exact_real_program import ExactConstant, GenericExactConstant


if __name__ == '__main__':
    big_number = ExactConstant(1e100)
    e = GenericExactConstant(lambda context: bf.exp(1, context))
    pi = GenericExactConstant(bf.const_pi)
    a = pi + big_number * e

    # x = [1, 2, 3]
    # x, y, z = [ExactConstant(i) for i in x]

    # a = (x * (y + z))
    a.lower_grad, a.upper_grad = 1, 1
    # a.evaluate(10, True).print()
    # print(a)
    # print(x.grad())
    # print(y.grad())
    # print(z.grad())

    a.evaluate(10, True).print()
    print(a)
    print(pi.grad())
    print(e.grad())
