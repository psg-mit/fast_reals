import bigfloat as bf

from exact_real_program import ExactConstant, GenericExactConstant


if __name__ == '__main__':
    e = GenericExactConstant(lambda context: bf.exp(1, context))
    pi = GenericExactConstant(bf.const_pi)
    a = pi + e
    a.evaluate(100).print()
