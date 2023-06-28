import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np

from modcma.parameters import Parameters
from modcma.surrogate.regression_models import get_model, SurrogateModelBase, \
    Linear_SurrogateModel, Quadratic_SurrogateModel, LQ_SurrogateModel

class Test_get_model(unittest.TestCase):
    def test_empty(self):
        p = Parameters(2)
        m = get_model(p.surrogate_model)(p)
        self.assertIsInstance(m, Linear_SurrogateModel)

    def test_Quandratic(self):
        p = Parameters(2)
        p.surrogate_model = 'Quadratic'
        m = get_model(p.surrogate_model)(p)
        self.assertIsInstance(m, Quadratic_SurrogateModel)

    def test_LQ(self):
        p = Parameters(2)
        p.surrogate_model = 'LQ'
        m = get_model(p.surrogate_model)(p)
        self.assertIsInstance(m, LQ_SurrogateModel)


class TestModelsBase(unittest.TestCase):
    def train_model(self, X, Y):
        self.model: SurrogateModelBase
        self.model.fit(X, Y, None)

    def train_try_model(self, X, Y):
        self.train_model(X, Y)
        self.try_model(X, Y)

    def try_model(self, X, Y):
        Yt = self.model.predict(X)
        self.assertIsNone(assert_array_almost_equal(Y, Yt))

    def try_ne_model(self, X, Y):
        Yt = self.model.predict(X)
        self.assertFalse(np.allclose(Y, Yt))


if __name__ == '__main__':
    unittest.main(verbosity=2)
