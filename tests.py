import unittest

from exact_real_program import ExactInterval


class MultiplyTest(unittest.TestCase):

    def test_symmetric(self):
        x, y = ExactInterval(1, 2), ExactInterval(2, 3)
        program = x * y
        program.lower_grad, program.upper_grad = 1, 1
        program.evaluate(10, ad=True)
        print(x.grad(), y.grad())
        self.assertEqual(x.grad(), (2, 3))
        self.assertEqual(y.grad(), (1, 2))

    def test_wrt_right(self):
        x, y = ExactInterval(-1, 1), ExactInterval(-3, 2)
        program = x * y
        program.lower_grad, program.upper_grad = 0, 1
        program.evaluate(10, ad=True)
        print(x.grad(), y.grad())
        self.assertEqual(x.grad(), (-3, 0))
        self.assertEqual(y.grad(), (-1, 0))

    def test_wrt_left(self):
        x, y = ExactInterval(-1, 1), ExactInterval(-3, 2)
        program = x * y
        program.lower_grad, program.upper_grad = 1, 0
        program.evaluate(10, ad=True)
        print(x.grad(), y.grad())
        self.assertEqual(x.grad(), (0, -3))
        self.assertEqual(y.grad(), (1, 0))

    def test_wrt_both(self):
        x, y = ExactInterval(-1, 1), ExactInterval(-3, 2)
        program = x * y
        program.lower_grad, program.upper_grad = 1, 1
        program.evaluate(10, ad=True)
        print(x.grad(), y.grad())
        self.assertEqual(x.grad(), (-3, -3))
        self.assertEqual(y.grad(), (0, 0))


if __name__ == '__main__':
    unittest.main()
