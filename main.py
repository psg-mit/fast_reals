import bigfloat as bf

from exact_real_program import ExactConstant, GenericExactConstant
from evaluate import evaluate, evaluate_using_derivatives


if __name__ == '__main__':
    big_number = ExactConstant(1e100)
    e = GenericExactConstant(lambda context: bf.exp(1, context))
    pi = GenericExactConstant(bf.const_pi)
    a = pi + big_number * e

    # x = [1, 2, 3]
    # x, y, z = [ExactConstant(i) for i in x]

    # a = (x * (y + z))
    # a.lower_grad, a.upper_grad = 1, 1
    # a.evaluate(10, True).print()
    # print(a)
    # print(x.grad())
    # print(y.grad())
    # print(z.grad())

    # evaluate
    # a.evaluate(10, True).print()
    # print(a)
    # print(pi.grad())
    # print(e.grad())
    program = a
    error_bound: int = 1
    # steps = evaluate(a, error_bound, True)

    steps = evaluate_using_derivatives(a, error_bound, [30] * a.subtree_size())
    print("It took", steps, "refinement steps to achieve the", error_bound, "error bound. \n")
    program.print()

    print(pi.grad())
    print(e.grad())
