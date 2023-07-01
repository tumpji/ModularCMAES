import numpy as np
import numpy.testing as npt

import unittest


def unittest_get_global_verbosity():
    return 2


class NumpyUnitTest:
    def assertIsNumpyArray(self, a):
        if not isinstance(a, np.ndarray):
            raise AssertionError('Argument "a" is not an instance of ndarray')

    def assertArrayComparable(self, a, b):
        if not isinstance(a, np.ndarray):
            raise AssertionError('Argument "a" is not an instance of ndarray')
        if not isinstance(b, np.ndarray):
            raise AssertionError('Argument "b" is not an instance of ndarray')

        if len(a.shape) != len(b.shape) or a.shape != b.shape:
            raise AssertionError(f'"a" and "b" have different number of \
                                 dimensions, a has shape {a.shape} while b has {b.shape}')

    def assertArrayEqual(self, a, b, tol=0.):
        self.assertArrayComparable(a, b)
        if not np.all(a - b <= tol):
            raise AssertionError('"a" and "b" did not match')

    def assertArrayNotEqual(self, a, b, tol=0.):
        self.assertArrayComparable(a, b)
        if np.all(a - b <= tol):
            raise AssertionError('"a" and "b" did match')

    def assertArraySymetric(self, a, tol=0.):
        self.assertIsNumpyArray(a)
        if not np.all(np.abs(a-a.T) <= tol):
            raise AssertionError('"a" is not symetric array')


class TestA(unittest.TestCase, NumpyUnitTest):
    def testSym(self):
        a = np.random.rand(4,4)
        b = a + a.T

        self.assertArraySymetric(b)

        with self.assertRaises(AssertionError):
            self.assertArraySymetric(a)

    def testNE(self):
        a = np.random.rand(4,3)
        b = np.random.rand(4,3)

        self.assertArrayNotEqual(a, b)
        with self.assertRaises(AssertionError):
            self.assertArrayNotEqual(a, a)

    def testEQ(self):
        a = np.random.rand(4,3)
        b = np.random.rand(4,3)

        self.assertArrayEqual(a, a)
        with self.assertRaises(AssertionError):
            self.assertArrayEqual(a, b)

    def testArrComp(self):
        a = np.random.rand(4,3)
        b = np.random.rand(4,2)

        self.assertArrayComparable(a, a)
        with self.assertRaises(AssertionError):
            self.assertArrayComparable(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=unittest_get_global_verbosity())

