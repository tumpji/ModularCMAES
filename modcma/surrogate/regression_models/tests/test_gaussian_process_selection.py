import math
import unittest

import numpy

from modcma import Parameters
from modcma.surrogate.regression_models import GaussianProcessBasicSelection, GaussianProcessBasicAdditiveSelection, \
    GaussianProcessBasicMultiplicativeSelection, GaussianProcessBasicBinarySelection
from modcma.surrogate.regression_models.model_gp import Quadratic, Linear, ExpSinSquared


class Test_GaussianProcessBasicSelection(unittest.TestCase):
    def test_quadratic(self):
        data = np.random.rand(50, 1)
        target = data[:,0]**2
        target -= np.mean(target)

        parameters = Parameters(1)
        parameters.surrogate_model_gp_noisy_samples = False
        model = GaussianProcessBasicSelection(parameters)
        model.fit(data, target)

        self.assertIs(model.best_model.KERNEL_CLS, Quadratic)

    def test_linear(self):
        data = np.random.rand(50, 2)
        target = data[:,0] + data[:,1] * 10

        parameters = Parameters(2)
        parameters.surrogate_model_gp_noisy_samples = False
        model = GaussianProcessBasicSelection(parameters)
        model.fit(data, target)

        self.assertIs(model.best_model.KERNEL_CLS, Linear)

    def test_sin(self):
        data = np.random.rand(75, 1) * 3.14 * 100
        #data = np.random.rand(272, 1) * 3.14 * 100

        target = np.sin(data/4 + 0.14)

        parameters = Parameters(1)
        model = GaussianProcessBasicSelection(parameters)
        model.fit(data, target)

        self.assertIs(model.best_model.KERNEL_CLS, ExpSinSquared)

    def test_generator(self):
        parameters = Parameters(1)
        model = GaussianProcessBasicSelection(parameters)
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks)
        )


class Test_GaussianProcessExtendedSelection(unittest.TestCase):
    def combination_number(self, a):
        # combinations with replacement (n over 2)
        return math.factorial(a + 2 - 1) // math.factorial(2) // math.factorial(a - 1)

    def test_generator_additive(self):
        parameters = Parameters(1)
        model = GaussianProcessBasicAdditiveSelection(parameters)
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks) +
            self.combination_number(len(model._building_blocks)) - 2
            # Lin + Lin == Lin
            # Qua + Qua == Qua
        )

    def test_generator_multiplicative(self):
        parameters = Parameters(1)
        model = GaussianProcessBasicMultiplicativeSelection(parameters)
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks) +
            self.combination_number(len(model._building_blocks)) - 1
            # Lin + Lin == Qua
        )

    def test_generator_both(self):
        parameters = Parameters(1)
        model = GaussianProcessBasicBinarySelection(parameters)
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks) +
            2*self.combination_number(len(model._building_blocks)) - 3
        )
