import numpy as np

from modcma.surrogate.data import *
import unittest
import numpy.testing as npt

class Test_SurrogateData_V1_push_pop_only(unittest.TestCase):
    def setUp(self):
        self.S = Parameters(5)
        self.A = SurrogateData_V1(self.S)

    def fill_in_pseudorandom(self, l):
        self.A.push_many(
            np.arange(l*5).reshape(-1, 5),
            np.arange(l).reshape(-1, 1)
        )

    def test_empty_len(self):
        self.assertEqual(0, len(self.A))

    def test_empty_pop(self):
        self.assertEqual(0, len(self.A))
        x, f = self.A.pop()
        self.assertEqual(0, len(x))
        self.assertEqual(0, len(f))

        x, f = self.A.pop(10)
        self.assertEqual(0, len(self.A))
        self.assertEqual(0, len(x))
        self.assertEqual(0, len(f))

    def test_over_pop(self):
        self.fill_in_pseudorandom(10)
        x, f = self.A.pop(15)
        self.assertEqual(10, len(x))
        self.assertEqual(10, len(f))
        self.assertEqual(0, len(self.A))

    def test_1_insert_5(self):
        self.A.push(np.arange(5), 3)
        self.assertEqual(1, len(self.A))

        x, f = self.A.pop(1)
        npt.assert_array_almost_equal(f, 3)
        npt.assert_array_almost_equal(np.arange(5)[np.newaxis, :], x)

    def test_1_insert_5_1(self):
        self.A.push(np.arange(5).reshape(5, 1), 3)
        self.assertEqual(1, len(self.A))

        x, f = self.A.pop(1)
        npt.assert_array_almost_equal(f, 3)
        npt.assert_array_almost_equal(np.arange(5)[np.newaxis, :], x)

    def test_1_insert_1_5(self):
        self.A.push(np.arange(5).reshape(1, 5), 3)
        self.assertEqual(1, len(self.A))

        x, f = self.A.pop(1)
        npt.assert_array_almost_equal(f, 3)
        npt.assert_array_almost_equal(np.arange(5)[np.newaxis, :], x)

    def test_2_insert_5(self):
        self.A.push(np.arange(5), 3)
        self.A.push(np.arange(5) + 5, 2)
        self.assertEqual(2, len(self.A))

        x, f = self.A.pop(2)
        npt.assert_array_almost_equal(f, np.array([3,2]).reshape(-1, 1))
        npt.assert_array_almost_equal(x, np.array([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]
        ]))

    def test_insert_many(self):
        self.A.push_many(
            np.arange(3*5).reshape(-1, 5),
            np.arange(3).reshape(-1, 1)
        )

        x, f = self.A.pop(3)
        npt.assert_array_almost_equal(f, np.arange(3).reshape(3, 1))
        npt.assert_array_almost_equal(x, np.arange(3*5).reshape(3, 5))

    def test_insert_x_wrong_shape(self):
        self.assertRaises(ValueError, self.A.push, np.arange(4), 3)
        self.assertRaises(ValueError, self.A.push, np.arange(10), 3)

    def test_insert_f_wrong_shape(self):
        self.assertRaises(ValueError, self.A.push, np.arange(5), np.array([1,2]))
        self.assertRaises(ValueError, self.A.push, np.arange(5), np.array([]))

    def test_insert_xf_wrong_shape(self):
        self.assertRaises(ValueError, self.A.push, np.arange(7), np.array([[1,2]]).T)

    def test_insert_many_x_wrong_shape(self):
        self.assertRaises(ValueError, self.A.push_many,
                          np.arange(6*2).reshape(-1, 6),
                          np.array([1, 2]))
        self.A.push_many(np.arange(5*2).reshape(-1, 5), np.array([[1, 2]]).T)

    def test_insert_many_f_wrong_shape(self):
        self.assertRaises(ValueError, self.A.push_many,
                          np.arange(5*2).reshape(-1, 5),
                          np.array([1, 2, 3]))
        self.A.push_many(np.arange(5*2).reshape(-1, 5), np.array([[1, 2]]).T)


class Test_SurrogateData_V1_sort(unittest.TestCase):
    def setUp(self):
        self.S = Parameters(5)
        self.S.surrogate_data_max_size_absolute = None
        self.S.surrogate_data_mahalanobis_space = None
        self.S.surrogate_data_mahalanobis_space_max_value = np.inf

        self.A = SurrogateData_V1(self.S)

    def fill_in_pseudorandom(self, l):
        self.A.push_many(
            np.arange(l*5).reshape(-1, 5),
            np.arange(l).reshape(-1, 5)
        )

    def test_sort_default(self):
        self.A.push_many(
            np.arange(20*5).reshape(-1, 5),
            np.arange(20).reshape(-1, 1),
            time_ordered=True
        )
        self.A.sort_all()
        npt.assert_array_almost_equal(self.A.X, np.arange(20 * 5).reshape(-1, 5))
        npt.assert_array_almost_equal(self.A.F, np.arange(20).reshape(-1, 1))

    def test_sort_euclidean(self):
        self.S.surrogate_data_sorting = 'euclidean'
        rng = np.random.default_rng(seed=42)
        self.A.push_many(
            rng.normal(0, 1, size=(20, 5)),
            rng.normal(0, 1, size=(20, 1)),
            time_ordered=True
        )

        self.A.sort_all()
        best = np.inf
        for x in self.A.X[::1]:
            assert x.shape == (5,)
            act = np.sum(np.square(x - self.A.parameters.m.ravel()))
            self.assertLessEqual(act, best)
            best = act

        self.S.surrogate_data_max_size_absolute = 2
        self.assertEqual(len(self.A.X), 2)
        self.assertEqual(len(self.A.F), 2)

        self.assertAlmostEqual(
            np.sum(np.square(self.A.X[-1] - self.A.parameters.m.ravel())), best)



'''

class TestSurrogateData_V1(unittest.TestCase):
    def assertEqual(self, first: Any, second: Any, msg: Any = ...) -> None:
        if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            if first.shape == second.shape:
                if np.equal(first, second).all():
                    return
        return super().assertEqual(first, second, msg)

    def setUp(self):
        self.S = Parameters(5)
        self.A = SurrogateData_V1(self.S)

    def fillin(self, n=1):
        for _ in range(n):
            x = np.random.randn(3, 1)
            y = np.random.randn(1, 1)
            self.A.push(x, y)

    def test_voidlen(self):
        self.assertEqual(0, len(self.A))

    def test_voidPop(self):
        self.A.pop()
        self.assertEqual(0, len(self.A))
        self.A.pop(10)
        self.assertEqual(0, len(self.A))

    def test_overPop(self):
        self.fillin(10)
        self.A.pop(15)
        self.assertEqual(0, len(self.A))

    def test_push31(self):
        xs, ys = [], []
        for i in range(10):
            x = np.random.randn(3, 1)
            y = np.random.randn(1, 1)
            self.A.push(x, y)
            self.assertEqual(i + 1, len(self.A))
            xs.append(x)
            ys.append(y)

        self.assertEqual(self.A.F.shape[0], 10)

        for i in range(10):
            x, y = self.A.pop()
            self.assertEqual(x, xs[i].T)
            self.assertEqual(y, ys[i])
            self.assertEqual(10 - i - 1, len(self.A))

    def test_push13(self):
        xs, ys = [], []
        for i in range(10):
            x = np.random.randn(1, 3)
            y = np.random.randn(1, 1)
            self.A.push(x, y)
            self.assertEqual(i + 1, len(self.A))
            xs.append(x)
            ys.append(y)

        self.assertEqual(self.A.F.shape[0], 10)

        for i in range(10):
            x, y = self.A.pop()
            self.assertEqual(x, xs[i])
            self.assertEqual(y, ys[i])
            self.assertEqual(10 - i - 1, len(self.A))

    def test_pushmany(self):
        self.S = Parameters(5, surrogate_data_sorting='lq', surrogate_data_mahalanobis_space=False)
        self.A = SurrogateData_V1(self.S)
        L = [random.randint(1, 10) for _ in range(10)]
        G = [(np.random.rand(length, 5), np.random.rand(length, 1)) for length in L]
        for (x, f) in G:
            self.A.push_many(x, f)
        self.assertEqual(len(self.A), sum(L))

        target = np.concatenate([x for (x, f) in G], axis=0)
        self.assertArrayEqual(target, self.A.X)

        target = np.concatenate([f for (x, f) in G], axis=0)
        self.assertEqual(target, self.A.F)

    def test_sort(self):
        self.S = Parameters(4, surrogate_data_sorting='lq', surrogate_data_mahalanobis_space=False)
        self.A = SurrogateData_V1(self.S)
        x = np.array([
            [1, 1, 3, 5],
            [2, 3, 2, 3],
            [1, 1, 1, 2],
            [2, 9, 1, 2],
            [1, 8, 1, 2]
        ])
        y = np.array([[4, 3, 1, 5, 2]]).T
        xok = np.array([
            [2, 9, 1, 2],
            [1, 1, 3, 5],
            [2, 3, 2, 3],
            [1, 8, 1, 2],
            [1, 1, 1, 2],
        ])

        for i in range(len(y)):
            self.A.push(x[i], y[i])

        for i in [0, 1]:
            self.A.sort(i)
            self.assertArrayEqual(self.A.X, x)
            self.assertArrayEqual(self.A.F, y)

        for i in range(2, 5 + 1):
            target = list(y[:-i]) + \
                     list(reversed(sorted(y[-i:])))
            target = np.array(target)

            self.A.sort(i)
            self.assertArrayEqual(self.A.F, target)

        self.assertArrayEqual(self.A.F, np.array([[5, 4, 3, 2, 1.]]).T)
        self.assertArrayEqual(self.A.X, xok)

    def test_max_size(self):
        S = Parameters(5)

        for size in [3, 64, 101, 200]:
            S.surrogate_data_max_size_absolute = size
            A = SurrogateData_V1(S)

            X = np.random.rand(size + 101, 5)
            F = np.random.rand(size + 101, 1)

            A.push_many(X, F)

            self.assertEqual(len(A.F), size)
            self.assertEqual(len(A.X), size)
            self.assertEqual(A.X.shape[1], 5)

    def test_weight(self):
        S = Parameters(5)

        # FULL
        for size in [3, 64, 101, 200]:
            S.surrogate_data_max_size_absolute = size
            S.surrogate_data_min_weight = 2.5
            S.surrogate_data_max_weight = 100.
            A = SurrogateData_V1(S)

            X = np.random.rand(size + 101, 5)
            F = np.random.rand(size + 101, 1)
            A.push_many(X, F)

            self.assertEqual(len(A.W), size)
            self.assertEqual(A.W[0], S.surrogate_data_min_weight)
            self.assertEqual(A.W[-1], S.surrogate_data_max_weight)
            self.assertAlmostEqual(A.W[1] - A.W[0], A.W[-1] - A.W[-2])

        # NOT FILLED
        A = SurrogateData_V1(S)
        A.push(np.random.rand(1, 5), np.random.rand(1, 1))

        self.assertEqual(len(A.W), 1)
        self.assertEqual(A.W[0], S.surrogate_data_min_weight)

        A.push(np.random.rand(1, 5), np.random.rand(1, 1))
        self.assertEqual(len(A.W), 2)
        self.assertEqual(A.W[0], S.surrogate_data_min_weight)
        self.assertEqual(A.W[-1], S.surrogate_data_max_weight)

        A.push(np.random.rand(1, 5), np.random.rand(1, 1))
        self.assertEqual(len(A.W), 3)
        self.assertEqual(A.W[0], S.surrogate_data_min_weight)
        self.assertAlmostEqual(A.W[1], (S.surrogate_data_min_weight + S.surrogate_data_max_weight) / 2)
        self.assertEqual(A.W[-1], S.surrogate_data_max_weight)

    def test_getitem(self):
        S = Parameters(5, surrogate_data_mahalanobis_space=False)

        InX = np.random.rand(30, 5)
        InY = np.random.rand(30, 1)

        A = SurrogateData_V1(S)
        A.push_many(InX, InY)

        for i in range(30):
            x, y = A[i]
            self.assertArrayEqual(x, InX[i])
            self.assertArrayEqual(y, InY[i])

    @unittest.skip('TODO')
    def test_parameters(self):
        pass
'''

if __name__ == '__main__':
    unittest.main(verbosity=2)
