from abc import abstractmethod, ABCMeta
from functools import cache
from typing import Tuple, Optional, Type, List, Generator
import operator
import itertools
import time
import copy
from collections import defaultdict

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.model_selection import KFold, LeaveOneOut

from modcma.typing_utils import XType, YType
from modcma.surrogate.model import SurrogateModelBase
from modcma.parameters import Parameters

import modcma.surrogate.losses as losses


# import kernels
from modcma.surrogate.gp_kernels import basic_kernels, functor_kernels, GP_kernel_concrete_base
for k in basic_kernels + functor_kernels:
    locals()[k.__name__] = k

# Stuff for statitc typing
MaternFiveHalves: Type[GP_kernel_concrete_base]
MaternOneHalf: Type[GP_kernel_concrete_base]
MaternThreeHalves: Type[GP_kernel_concrete_base]
RationalQuadratic: Type[GP_kernel_concrete_base]
ExponentiatedQuadratic: Type[GP_kernel_concrete_base]
ExpSinSquared: Type[GP_kernel_concrete_base]
Linear: Type[GP_kernel_concrete_base]
Quadratic: Type[GP_kernel_concrete_base]
Cubic: Type[GP_kernel_concrete_base]
Parabolic: Type[GP_kernel_concrete_base]
ExponentialCurve: Type[GP_kernel_concrete_base]
Constant: Type[GP_kernel_concrete_base]

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


class _GaussianProcessModelMixtureBase:
    # TODO: RENAME

    def __init__(self, parameters: Parameters) -> None:
        self.parameters = parameters
        self.MAX_MODELS = self.parameters.surrogate_model_selection_max_models
        self.MAX_TIME = self.parameters.surrogate_model_selection_max_seconds

        self.random_state = np.random.RandomState(
            self.parameters.surrogate_model_selection_random_state)

        # the selection ...
        self._building_blocks: List[Type[GP_kernel_concrete_base]] = [
            MaternFiveHalves,
            MaternOneHalf,
            MaternThreeHalves,
            RationalQuadratic,
            ExponentiatedQuadratic,
            ExpSinSquared,
            Linear,
            Quadratic,
            #### #Cubic,
            #Parabolic,
            #ExponentialCurve,
            #Constant,
        ]

    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from self._building_blocks

    def _shuffle_kernel_space(self):
        space = np.array(list(self._generate_kernel_space()))
        if self.parameters.surrogate_model_selection_randomization:
            self.random_state.shuffle(space)
        yield from space

    def _fit(self, X: XType, F: YType, W: YType):
        time_start = time.time()
        models, losses = [], []

        for i, kernel in enumerate(self._shuffle_kernel_space()):
            if self.MAX_MODELS and i >= self.MAX_MODELS:
                break

            model = _GaussianProcessModel(self.parameters, kernel)
            loss = model._loss(X, F, W)
            models.append(model)
            losses.append(loss)

            if self.MAX_TIME and (time.time() - time_start) >= self.MAX_TIME:
                break

        if np.all(np.isnan(losses)):
            self.best_model = None
        else:
            best_index = np.nanargmin(losses)
            self.best_model = models[best_index]
            self.best_model._fit(X, F, W)
        return self

    def _predict(self, X: XType) -> Optional[YType]:
        if self.best_model is None:
            return None
        return self.best_model._predict(X)

    def _predict_with_confidence(self, X: XType) -> Optional[Tuple[YType, YType]]:
        if self.best_model is None:
            return None
        return self.best_model._predict_with_confidence(X)



class GaussianProcessBasicSelection(_GaussianProcessModelMixtureBase, SurrogateModelBase):
    def __init__(self, parameters: Parameters):
        SurrogateModelBase.__init__(self, parameters)
        _GaussianProcessModelMixtureBase.__init__(self, parameters)

    def df(self):
        return super().df

    def _fit(self, X: XType, F: YType, W: YType) -> None:
        return super()._fit(X, F, W)

    def _predict(self, X: XType) -> YType:
        return super(_GaussianProcessModelMixtureBase, self)._predict(X)


class GaussianProcessBasicAdditiveSelection(GaussianProcessBasicSelection):
    ''' <model> Gaussian Process model that chooses the best addition of two kernels'''

    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) in [(Linear, Linear), (Quadratic, Quadratic), ]:
                continue
            yield a + b


class GaussianProcessBasicMultiplicativeSelection(GaussianProcessBasicSelection):
    ''' <model> Gaussian Process model that chooses the best multiplication of two kernels'''

    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) in [(Linear, Linear), ]:
                continue
            yield a * b


class GaussianProcessBasicBinarySelection(GaussianProcessBasicSelection):
    ''' <model> Gaussian Process model that chooses the best addition or multiplication of two kernels'''

    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) not in [(Linear, Linear), ]:
                yield a * b
            if (a, b) not in [(Linear, Linear), (Quadratic, Quadratic), ]:
                yield a + b


class _GaussianProcessModelSearchBase(_GaussianProcessModelMixtureBase, SurrogateModelBase):
    def __init__(self, parameters, random_state=None):
        self.parameters = parameters
        self.best_model = None

        self.random_state = np.random.RandomState(
            self.parameters.surrogate_model_selection_random_state)

    def _expand_node(self, base_kernel):
        order = np.arange(2 * len(self._basis))
        indices = np.repeat(np.arange(len(self._basis), dtype=np.int_), 2)
        operations = np.tile(['+', '*'], len(self._basis))

        if self.parameters.surrogate_model_selection_randomization:
            self.random_state.shuffle(order)

        for i in order:
            kernel, operation = self._basis[indices[i]], operations[i]
            if operation == '+':
                yield base_kernel + kernel
            elif operation == '*':
                yield base_kernel * kernel
            else:
                raise NotImplementedError(
                    f'Unknown operation between two kernels {operation} is not implemented')

    def _init_time(self):
        self.time_start = time.time()

    def _check_timeup(self):
        if self.TRAIN_MAX_TIME_S:
            if time.time() - self.time_start > self.TRAIN_MAX_TIME_S:
                return True
        return False

    def _search_method(self, X: XType, F: YType, W: YType):
        # implements greedy search
        self._init_time()
        best_state = Constant
        best_value = self._evaluate_kernel(best_state)

        timeup = False

        while not timeup:
            new_states, new_values = [], []

            for new_state in self._expand_node(best_state):
                new_values.append(self._evaluate_kernel(new_state))
                new_states.append(new_state)

                if timeup := self._check_timeup():
                    break

            if len(new_values) == 0:
                break

            best_new_index = np.argmin(new_values)

            if new_values[best_new_index] < best_value:
                best_value = new_values[best_new_index]
                best_state = new_states[best_new_index]
            else:
                break
        return best_state

    def _evaluate_kernel(self, kernel):
        # TODO
        return value
        pass

    def _fit(self, X: XType, F: YType, W: YType):
        best_kernel = self._search_method(X, F, W)

        self.time_start = time.time()
        self.best_model = Constant
        self.best_model = 

        for major_iteration in itertools.count():
            if major_iteration == 0:
                for kernel in self._basis:
                    pass

            pass

        models = []
        losses = []

        for model in itertools.islice(self._generate_model_space(), self.TRAIN_MAX_MODELS):
            loss = model._loss(X, F)

            models.append(model)
            losses.append(loss)

            if self.TRAIN_MAX_TIME_S:
                if time.time() - time_start > self.TRAIN_MAX_TIME_S:
                    break

        best_index = np.nanargmin(losses)
        self.best_model = models[best_index]
        return self

    def _predict(self, X: XType) -> YType:
        return self.best_model._predict(X)

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return self.best_model._predict_with_confidence(X)


'''
class GaussianProcessPenalizedAdditiveSelection(GaussianProcessBasicSelection):
    def penalize_kernel(self, loss, kernel_obj):
        return super().penalize_kernel(loss, kernel_obj)

    def _predict(self, X: XType) -> YType:
        return self.best_model._predict(X)

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return self.best_model._predict_with_confidence(X)

'''
