import unittest

from exact_real_program import ExactInterval


class AddTest(unittest.TestCase):

    def test_simple(self):
        x, y = ExactInterval(1, 2), ExactInterval(2, 3)
        program = x + y
        program.lower_grad, program.upper_grad = 1, 1
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (1, 1))
        self.assertEqual(y.grad(), (1, 1))

    def test_left(self):
        x, y = ExactInterval(1, 2), ExactInterval(2, 3)
        program = x + y
        program.lower_grad, program.upper_grad = 1, 0
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (1, 0))
        self.assertEqual(y.grad(), (1, 0))

    def test_right(self):
        x, y = ExactInterval(1, 2), ExactInterval(2, 3)
        program = x + y
        program.lower_grad, program.upper_grad = 0, 1
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (0, 1))
        self.assertEqual(y.grad(), (0, 1))


class SubTest(unittest.TestCase):

    def test_simple(self):
        x, y = ExactInterval(1, 2), ExactInterval(2, 3)
        program = x - y
        program.lower_grad, program.upper_grad = 1, 1
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (1, 1))
        self.assertEqual(y.grad(), (-1, -1))

    def test_left(self):
        x, y = ExactInterval(1, 2), ExactInterval(2, 3)
        program = x - y
        program.lower_grad, program.upper_grad = 1, 0
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (1, 0))
        self.assertEqual(y.grad(), (0, -1))

    def test_right(self):
        x, y = ExactInterval(1, 2), ExactInterval(2, 3)
        program = x - y
        program.lower_grad, program.upper_grad = 0, 1
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (0, 1))
        self.assertEqual(y.grad(), (-1, 0))


class MulTest(unittest.TestCase):

    def test_simple(self):
        x, y = ExactInterval(1, 2), ExactInterval(3, 4)
        program = x * y
        program.lower_grad, program.upper_grad = 1, 1
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (3, 4))
        self.assertEqual(y.grad(), (1, 2))

    def test_left(self):
        x, y = ExactInterval(1, 2), ExactInterval(3, 4)
        program = x * y
        program.lower_grad, program.upper_grad = 1, 0
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (3, 0))
        self.assertEqual(y.grad(), (1, 0))

    def test_right(self):
        x, y = ExactInterval(1, 2), ExactInterval(3, 4)
        program = x * y
        program.lower_grad, program.upper_grad = 0, 1
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (0, 4))
        self.assertEqual(y.grad(), (0, 2))

    def test_tricky(self):
        x, y = ExactInterval(-1, 1), ExactInterval(-3, 2)
        program = x * y
        program.lower_grad, program.upper_grad = 1, 1
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (-3, -3))
        self.assertEqual(y.grad(), (0, 0))

    def test_tricky_left(self):
        x, y = ExactInterval(-1, 1), ExactInterval(-3, 2)
        program = x * y
        program.lower_grad, program.upper_grad = 1, 0
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (0, -3))
        self.assertEqual(y.grad(), (1, 0))

    def test_tricky_right(self):
        x, y = ExactInterval(-1, 1), ExactInterval(-3, 2)
        program = x * y
        program.lower_grad, program.upper_grad = 0, 1
        program.evaluate(10, ad=True)
        self.assertEqual(x.grad(), (-3, 0))
        self.assertEqual(y.grad(), (-1, 0))


if __name__ == '__main__':
    unittest.main()
