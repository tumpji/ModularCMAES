import numpy as np

from modcma.surrogate.data import *
import unittest
import numpy.testing as npt
import scipy
import math


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
        x = rng.normal(0, 1, size=(20, 5))
        f = rng.normal(0, 1, size=(20, 1))
        self.A.push_many(x, f, time_ordered=True)

        self.A.sort_all()
        best = np.inf
        for xa in self.A.X[::1]:
            assert xa.shape == (5,)
            act = np.sum(np.square(xa - self.A.parameters.m.ravel()))
            self.assertLessEqual(act, best)
            best = act

        self.S.surrogate_data_max_size_absolute = 2
        self.assertEqual(len(self.A.X), 2)
        self.assertEqual(len(self.A.F), 2)

        self.assertAlmostEqual(
            np.sum(np.square(self.A.X[-1] - self.A.parameters.m.ravel())), best)
        self.sort_match(x,f,self.A.X,self.A.F)

    def test_sort_lq(self):
        self.S.surrogate_data_sorting = 'lq'
        rng = np.random.default_rng(seed=42)
        x = rng.normal(0, 1, size=(20, 5))
        f = rng.normal(0, 1, size=(20, 1))
        self.A.push_many(x, f, time_ordered=True)
        self.A.sort_all()

        npt.assert_array_almost_equal(-np.sort(-f, axis=0), self.A.F)
        npt.assert_array_almost_equal(x[np.argsort(-f, axis=0)[:, 0]], self.A.X)
        self.sort_match(x,f,self.A.X,self.A.F)

    def sort_match(self, origX, origF, X, F):
        for x, f in zip(X, F):
            boolmap = np.all(origX == x[np.newaxis, :], axis=1)
            self.assertTrue(np.any(boolmap))
            self.assertTrue(np.all(origF[boolmap, :] == f))

    def test_sort_mahalanobis(self):
        self.S.surrogate_data_sorting = 'mahalanobis'
        self.S.inv_root_C = np.array([
            [0.25, 0   , 0   , 0   , 0  ],
            [0   , 0.33, 0   , 0   , 0  ],
            [0   , 0   , 0.5 , 0   , 0.1],
            [0   , 0   , 0   , 0.66, 0.2],
            [0   , 0   , 0.1 , 0.2 , 1.00],
        ])
        rng = np.random.default_rng(seed=42)
        x = rng.normal(0, 1, size=(40, 5))
        f = rng.normal(0, 1, size=(40, 1))
        self.A.push_many(x, f, time_ordered=True)
        self.A.sort_all()

        best = np.inf
        for x_act in self.A.X:
            VI = self.S.inv_root_C @ self.S.inv_root_C.T
            distance = scipy.spatial.distance.mahalanobis(self.S.m.ravel(), x_act, VI)
            self.assertLessEqual(distance, best)
            best = distance

        self.S.surrogate_data_max_size_absolute = 1
        npt.assert_array_almost_equal(x_act[np.newaxis, :], self.A.X)
        self.sort_match(x,f,self.A.X,self.A.F)


class Test_SurrogateData_V1_weighting(unittest.TestCase):
    def setUp(self):
        self.S = Parameters(5)
        self.S.surrogate_data_max_size_absolute = None
        self.S.surrogate_data_mahalanobis_space = None
        self.S.surrogate_data_mahalanobis_space_max_value = np.inf

        self.A = SurrogateData_V1(self.S)

        surrogate_data_weighting: ('constant', 'linear', 'logarithmic') = 'linear'

        self.A.push_many(
            np.arange(20*5).reshape(-1, 5),
            np.arange(20).reshape(-1, 1),
            time_ordered=True
        )
        self.A.sort_all()

    def test_weight_constant(self):
        self.S.surrogate_data_weighting = 'constant'
        npt.assert_array_almost_equal(self.A.W, np.ones((20,)))

    def test_weight_linear(self):
        self.S.surrogate_data_weighting = 'linear'
        self.S.surrogate_data_min_weight = 1.
        self.S.surrogate_data_max_weight = 6.
        self.assertAlmostEqual(1, self.A.W[0])
        self.assertAlmostEqual(6, self.A.W[-1])
        self.assertAlmostEqual(
            self.A.W[2] - self.A.W[1],
            self.A.W[12] - self.A.W[11]
        )

    def test_weight_logarithmic(self):
        self.S.surrogate_data_weighting = 'logarithmic'
        self.S.surrogate_data_min_weight = 1.
        self.S.surrogate_data_max_weight = 6.
        self.assertAlmostEqual(1, self.A.W[0])
        self.assertAlmostEqual(6, self.A.W[-1])
        for i in range(18):
            self.assertLess(
                self.A.W[i+1] - self.A.W[i],
                self.A.W[i+2] - self.A.W[i+1]
            )
            self.assertAlmostEqual(
                math.log(self.A.W[i + 1]) - math.log(self.A.W[i]),
                math.log(self.A.W[i + 2]) - math.log(self.A.W[i + 1])
            )

    def test_weight_checking(self):
        self.S.surrogate_data_min_weight = -1.
        self.S.surrogate_data_max_weight = 5.
        self.assertRaises(ValueError, lambda: self.A.W)

        self.S.surrogate_data_min_weight = 1.
        self.S.surrogate_data_max_weight = -5.
        self.assertRaises(ValueError, lambda: self.A.W)

        self.S.surrogate_data_min_weight = 5.
        self.S.surrogate_data_max_weight = 1.
        self.assertRaises(ValueError, lambda: self.A.W)


if __name__ == '__main__':
    unittest.main(verbosity=2)
