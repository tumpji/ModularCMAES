import math
import unittest

import numpy as np

from modcma import Parameters
from modcma.surrogate.regression_models import GaussianProcessBasicSelection, GaussianProcessBasicAdditiveSelection, \
    GaussianProcessBasicMultiplicativeSelection, GaussianProcessBasicBinarySelection
from modcma.surrogate.regression_models.model_gp import Quadratic, Linear, ExpSinSquared


class Test_GaussianProcessBasicSelection(unittest.TestCase):
    @staticmethod
    def get_parameters(d: int) -> Parameters:
        return Parameters(d,
                          seed=42,
                          surrogate_data_weighting='constant',
                          surrogate_data_mahalanobis_space=False,
                          surrogate_model_gp_noisy_samples=False
                          )

    def setUp(self) -> None:
        np.random.seed(42)

    def test_quadratic(self):
        data = np.random.rand(50, 1)
        data -= np.mean(data, axis=0)
        target = data[:, 0] ** 2
        target -= np.mean(target)

        model = GaussianProcessBasicSelection(self.get_parameters(1))
        model.fit(data, target)

        self.assertIs(model.best_model.KERNEL_CLS, Quadratic)

    def test_linear(self):
        data = np.random.rand(50, 2)
        data -= np.mean(data, axis=0)
        target = data[:, 0] + data[:, 1] * 10
        target -= np.mean(target)

        model = GaussianProcessBasicSelection(self.get_parameters(2))
        model.fit(data, target)

        self.assertIs(model.best_model.KERNEL_CLS, Linear)

    def test_sin(self):
        data = np.random.rand(75, 1) * 3.14 * 100
        data -= np.mean(data, axis=0)
        target = np.sin(data / 4 + 0.14)
        target -= np.mean(target)

        model = GaussianProcessBasicSelection(self.get_parameters(1))
        model.fit(data, target)

        self.assertIs(model.best_model.KERNEL_CLS, ExpSinSquared)

    def test_generator(self):
        parameters = Parameters(1)
        model = GaussianProcessBasicSelection(self.get_parameters(1))
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks)
        )


class Test_GaussianProcessExtendedSelection(unittest.TestCase):
    @staticmethod
    def get_parameters(d: int) -> Parameters:
        return Parameters(d,
                          seed=42,
                          surrogate_data_weighting='constant',
                          surrogate_data_mahalanobis_space=False,
                          surrogate_model_gp_noisy_samples=False
                          )

    @staticmethod
    def combination_number(a):
        # combinations with replacement (n over 2)
        return math.factorial(a + 2 - 1) // math.factorial(2) // math.factorial(a - 1)

    def test_generator_additive(self):
        model = GaussianProcessBasicAdditiveSelection(self.get_parameters(1))
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks) +
            self.combination_number(len(model._building_blocks)) - 2
            # Lin + Lin == Lin
            # Qua + Qua == Qua
        )

    def test_generator_multiplicative(self):
        model = GaussianProcessBasicMultiplicativeSelection(self.get_parameters(1))
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks) +
            self.combination_number(len(model._building_blocks)) - 1
            # Lin + Lin == Qua
        )

    def test_generator_both(self):
        model = GaussianProcessBasicBinarySelection(self.get_parameters(1))
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks) +
            2 * self.combination_number(len(model._building_blocks)) - 3
        )
