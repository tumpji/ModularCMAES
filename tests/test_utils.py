"""Module containing tests for ModularCMA-ES Utilities."""

import io
import unittest
import unittest.mock

import numpy as np
import numpy.testing as npt

from modcma import utils


class TestUtils(unittest.TestCase):
    """Test case for utilities of Modular CMA-ES package."""

    def setUp(self):
        """Test setup method."""
        class Foo(utils.AnnotatedStruct):
            x: int
            y: float = 0.0
            z: np.ndarray = np.ones(5)
            c: (None, "x", "y", 1) = None

        self.fooclass = Foo

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_timeit(self, mock_stdout):
        """Test timit method."""
        @utils.timeit
        def f():
            pass

        f()
        self.assertIn("Time elapsed", mock_stdout.getvalue())

    def test_anyof(self):
        """Test AnyOf descriptor."""
        foo = self.fooclass(1)
        self.assertEqual(foo.c, None)
        with self.assertRaises(ValueError):
            foo.c = "z"
            foo.c = 10
            foo.c = 1.0
        foo.c = "x"
        self.assertEqual(foo.c, "x")

    def test_instanceof(self):
        """Test InstanceOf descriptor."""
        foo = self.fooclass(1)
        self.assertEqual(int, type(foo.x))
        self.assertEqual(float, type(foo.y))
        self.assertEqual(np.ndarray, type(foo.z))

        x = np.zeros(1)
        foo.z = x
        self.assertListEqual(foo.z.tolist(), x.tolist())
        self.assertNotEqual(id(foo.z), id(x))

        with self.assertRaises(TypeError):
            _ = self.fooclass(None)
            _ = self.fooclass("")
            _ = self.fooclass("x")
            _ = self.fooclass(1.0)

            foo.y = 1
            foo.y = "z"
            foo.z = 1
            foo.z = "z"

    def test_metaclass_raises(self):
        """Test metaclass raises correct error."""
        with self.assertRaises(TypeError):
            class Foo(utils.AnnotatedStruct):
                x: "x"
            _ = Foo()

    def test_repr(self):
        """Test representation."""
        self.assertEqual(type(repr(self.fooclass(1))), str)

    def test_descriptor(self):
        """Test descriptor."""
        class Foo:
            x = utils.Descriptor()

        self.assertIsInstance(Foo.x, utils.Descriptor)
        foo = Foo()
        foo.x = 1
        self.assertEqual(foo.x, 1)
        del foo.x
        self.assertNotIn("x", foo.__dict__)

    def test_ert(self):
        """Test ert method."""
        evals = [5000, 45000, 1000, 100, 10]
        budget = 10000
        ert, ert_sd, n_succ = utils.ert(evals, budget)
        self.assertEqual(n_succ, 4)
        self.assertAlmostEqual(ert, 12777.5)
        self.assertAlmostEqual(ert_sd, 17484.642861665)

        for evals in ([50000], [], [int(1e10)]):
            ert, ert_sd, n_succ = utils.ert(evals, budget)
            self.assertEqual(ert, float("inf"))
            self.assertEqual(np.isnan(ert_sd), True)
            self.assertEqual(n_succ, 0)


class TestUtilsResultCollector(unittest.TestCase):
    def setUp(self) -> None:
        self.obj = utils.ResultCollector()

    def test_one_element(self):
        self.obj.provide_result('a', 2)
        self.assertEqual(float(self.obj.get_result('a')), 2.)

    def test_multiple_elements(self):
        self.obj.provide_result('a', 2)
        self.obj.provide_result('b', 3)
        self.assertEqual(float(self.obj.get_result('a')), 2.)
        self.obj.provide_result('cc', 4)
        self.assertEqual(float(self.obj.get_result('b')), 3.)
        self.assertEqual(float(self.obj.get_result('cc')), 4.)

    def test_no_provide(self):
        self.obj.provide_result('a', 2)
        self.assertRaises(KeyError, self.obj.get_result, 'b')

    def test_mean_aggregation_default(self):
        self.obj.provide_result('a', 2)
        self.obj.provide_result('a', 3)
        self.assertAlmostEqual(self.obj.get_result('a'), 2.5)

    def test_mean_aggregation_explicit_mean(self):
        self.obj.provide_result('a', 10)
        self.obj.provide_result('a', 20)
        self.assertAlmostEqual(self.obj.get_result('a', aggregation_method=utils.AggregationMethod.MEAN), 15)

    def test_mean_aggregation_explicit_median(self):
        self.obj.provide_result('a', 10)
        self.obj.provide_result('a', 10)
        self.obj.provide_result('a', 20)
        self.assertAlmostEqual(self.obj.get_result('a', aggregation_method=utils.AggregationMethod.MEDIAN), 10)

        self.obj.provide_result('a', 20)
        self.assertAlmostEqual(self.obj.get_result('a', aggregation_method=utils.AggregationMethod.MEDIAN), 15)

        self.obj.provide_result('a', 20)
        self.assertAlmostEqual(self.obj.get_result('a', aggregation_method=utils.AggregationMethod.MEDIAN), 20)

    def test_multiple_outputs(self):
        self.obj.provide_result('a', 10)
        self.obj.provide_result('a', 10)
        self.obj.provide_result('a', 20)

        self.obj.provide_result('b', 10)
        self.obj.provide_result('b', 20)
        self.obj.provide_result('b', 20)

        a = self.obj.get_results()
        b = self.obj.get_results(aggregation_method=utils.AggregationMethod.MEAN)
        c = self.obj.get_results(aggregation_method=utils.AggregationMethod.MEDIAN)

        self.assertEqual(a,b)
        self.assertNotEqual(a,c)
        self.assertEqual(len(a), 2)
        self.assertEqual(len(c), 2)
        self.assertAlmostEqual(float(a['a']), 40/3)
        self.assertAlmostEqual(float(a['b']), 50/3)
        self.assertEqual(float(c['a']), 10)
        self.assertEqual(float(c['b']), 20)

    def test_multidimensional(self):
        aaa = np.array([[1, 2, 3]])
        bbb = np.array([3, 1, 2])
        ccc = np.array([[[3, 1, 2], [1, 1, 2]]])

        for i in [0, 1, 2, 2]:
            self.obj.provide_result('aaa', aaa + i)
            self.obj.provide_result('bbb', bbb + i)
            self.obj.provide_result('ccc', ccc + i)

        mean_aaa = self.obj.get_result('aaa')
        mean_bbb = self.obj.get_result('bbb')
        mean_ccc = self.obj.get_result('ccc')

        median_aaa = self.obj.get_result('aaa', aggregation_method=utils.AggregationMethod.MEDIAN)
        median_bbb = self.obj.get_result('bbb', aggregation_method=utils.AggregationMethod.MEDIAN)
        median_ccc = self.obj.get_result('ccc', aggregation_method=utils.AggregationMethod.MEDIAN)

        self.assertEqual(mean_aaa.shape, aaa.shape)
        self.assertEqual(mean_bbb.shape, bbb.shape)
        self.assertEqual(mean_ccc.shape, ccc.shape)

        npt.assert_array_almost_equal(aaa + 5/4, mean_aaa)
        npt.assert_array_almost_equal(bbb + 5/4, mean_bbb)
        npt.assert_array_almost_equal(ccc + 5/4, mean_ccc)

        npt.assert_array_almost_equal(aaa + 1.5, median_aaa)
        npt.assert_array_almost_equal(bbb + 1.5, median_bbb)
        npt.assert_array_almost_equal(ccc + 1.5, median_ccc)


if __name__ == "__main__":
    unittest.main()
