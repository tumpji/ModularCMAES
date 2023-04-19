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

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def create_positive_variable(default, dtype=tf.float64, name=None):
    if isinstance(default, (float, int)):
        default = tf.constant(default, dtype=dtype)

    bijector = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    return tfp.util.TransformedVariable(
        initial_value=default,
        bijector=bijector,
        dtype=dtype,
        name=name,
    )


def create_constant(default, dtype=tf.float64, name: Optional[str] = None):
    return tf.constant(default, dtype=dtype, name=name)

# ###############################################################################
# ### MODEL BUILDING COMPOSITION MODELS
# ###############################################################################


class _ModelBuildingBase(metaclass=ABCMeta):
    def __init__(self, parameters, kernel_cls):
        self.parameters = parameters
        self.kernel_cls = kernel_cls
        self.mean_fn = None

        # default
        self.observation_index_points = None
        self.observations = None
        self.kernel = None

    @abstractmethod
    def build_for_training(self,
                           observation_index_points=None,
                           observations=None) -> tfp.distributions.GaussianProcess:
        self.observation_index_points = observation_index_points
        self.observations = observations
        self.kernel = self.kernel_cls(self.parameters)

    @abstractmethod
    def build_for_regression(self,
                             X,
                             observation_index_points=None,
                             observations=None
                             ) -> tfp.distributions.GaussianProcessRegressionModel:
        pass

    @staticmethod
    def create_class(parameters: Parameters):
        if parameters.surrogate_model_gp_noisy_samples:
            return _ModelBuilding_Noisy
        else:
            return _ModelBuilding_Noiseless

    def copy(self):
        a = copy.copy(self)
        a.observation_index_points = None
        a.observations = None
        a.kernel = None
        return a

class _ModelBuilding_Noiseless(_ModelBuildingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_noise_variance = create_constant(0.)

    def build_for_training(self,
                           observation_index_points=None,
                           observations=None):
        super().build_for_training(observation_index_points, observations)

        return tfd.GaussianProcess(
            kernel=self.kernel.kernel(),
            mean_fn=self.mean_fn,
            index_points=self.observation_index_points,
            observation_noise_variance=self.observation_noise_variance
        )

    def build_for_regression(self,
                             X,
                             observation_index_points=None,
                             observations=None):
        if observation_index_points is None:
            assert(observations is None)

            observation_index_points = self.observation_index_points
            observations = self.observations

        return tfd.GaussianProcessRegressionModel(
            kernel=self.kernel.kernel(),
            mean_fn=self.mean_fn,
            index_points=X,
            observation_index_points=observation_index_points,
            observations=observations,
            observation_noise_variance=self.observation_noise_variance,
            predictive_noise_variance=self.observation_noise_variance,
        )


class _ModelBuilding_Noisy(_ModelBuilding_Noiseless):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_noise_variance = \
            create_positive_variable(1., name='observation_noise_variance')


# ###############################################################################
# ### MODEL TRAINING COMPOSITION MODELS
# ###############################################################################

class NoFold:
    def split(self, X):
        h = np.arange(len(X))
        return h, h

class _ModelTrainingBase(metaclass=ABCMeta):
    ''' provides training and loss prediction '''

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.training_cache = defaultdict(list)

        self.FOLDS: int = self.parameters.surrogate_model_selection_cross_validation_folds
        self.RANDOM_STATE: Optional[int] = \
            self.parameters.surrogate_model_selection_cross_validation_random_state

    @staticmethod
    def create_class(parameters: Parameters):
        return _ModelTraining_MaximumLikelihood

    @abstractmethod
    def compute_loss(self,
             model: _ModelBuildingBase,
             observation_test_points,
             observations_test,
             observation_index_points,
             observations) -> float:
        ''' predicts loss of the model (the model may be changed) '''
        pass

    def fit_model(self,
            model: _ModelBuildingBase,
            observation_index_points,
            observations) -> _ModelBuildingBase:
        ''' fits the model '''
        if model.observation_index_points is not None and \
           model.observations is not None and \
           np.array_equal(model.observation_index_points, observation_index_points) and \
           np.array_equal(model.observations, observations):
            return model
        # cache miss ...
        return self._fit_model(model, observation_index_points, observations)

    def _compute_loss(self, observations, predictions, stddev=None):
        ''' computes loss given prediction and target values '''
        loss_name = self.parameters.surrogate_model_selection_criteria
        loss = losses.get_cls_by_name(loss_name, self.parameters)
        return loss(predictions, observations, stddev=stddev)

    def _loss_aggregation_method(self, loss_history) -> float:
        ''' aggregate losses into one value '''
        if len(loss_history) == 1:
            return loss_history[0]
        return float(np.nanmean(loss_history))

    def _setup_folding(self, X):
        ''' return folding object with .split(X) -> ixd, idx interface '''
        if self.FOLDS is None or self.FOLDS <= 0:
            return NoFold()
        return KFold(
            n_splits=min(self.FOLDS, len(X)),
            shuffle=True,
            random_state=self.RANDOM_STATE
        )

    @abstractmethod
    def _fit_model(self,
             model: _ModelBuildingBase,
             observation_index_points,
             observations) -> _ModelBuildingBase:
        ''' fits the model, the model is changed '''
        pass


class _ModelTraining_MaximumLikelihood(_ModelTrainingBase):
    ''' implements maximum likelihood training for GP '''

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        self.LEARNING_RATE = self.parameters.surrogate_model_gp_learning_rate
        self.MAX_ITERATIONS = self.parameters.surrogate_model_gp_max_iterations
        self.EARLY_STOPPING_DELTA = self.parameters.surrogate_model_gp_early_stopping_delta
        self.EARLY_STOPPING_PATIENCE = self.parameters.surrogate_model_gp_early_stopping_patience

    def _fit_gp(self, model, observations):
        ''' fits one gp, returns the gp '''
        optimizer = tf.optimizers.Adam(learning_rate=self.LEARNING_RATE)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                loss = -model.log_prob(observations)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        minimal_neg_log_likelihood = np.inf
        minimal_index = 0

        neg_log_likelihood = np.nan
        for i in range(self.MAX_ITERATIONS):
            neg_log_likelihood = step()
            neg_log_likelihood = neg_log_likelihood.numpy()
            # nan
            if np.isnan(neg_log_likelihood):
                break

            if minimal_neg_log_likelihood - self.EARLY_STOPPING_DELTA > neg_log_likelihood:
                minimal_neg_log_likelihood = neg_log_likelihood
                minimal_index = i
            elif minimal_index + self.EARLY_STOPPING_PATIENCE < i:
                break
        return model

    def _loss_of_one_model(self,
                           model,
                           observation_index_points,
                           observations,
                           observation_test_points,
                           observations_test):
        ''' computes loss of one model '''
        gp = model.build_for_training(observation_index_points, observations)
        self._fit_gp(gp, observations)

        train_loss = np.array_equal(observation_index_points,
                                    observation_test_points)\
            and np.array_equal(observations, observations_test)

        if not train_loss:
            gp = model.build_for_regression(
                observation_test_points,
                observation_index_points,
                observations)

        mean = gp.mean().numpy()
        stddev = gp.stddev().numpy()
        return self._compute_loss(observations_test, mean, stddev=stddev)

    def compute_loss(self,
             model: _ModelBuildingBase,
             observation_index_points,
             observations) -> float:
        loss_history = []
        folding_method = self._setup_folding(observation_index_points)

        for train_idx, test_idx in folding_method.split(observation_index_points):
            partial_model = model.copy()

            # data selection
            act_observation_index_points = observation_index_points[train_idx]
            act_observations = observations[train_idx]
            act_test_observation_index_points = observation_index_points[test_idx]
            act_test_observations = observations[test_idx]

            loss = self._loss_of_one_model(partial_model,
                                           act_observation_index_points,
                                           act_observations,
                                           act_test_observation_index_points,
                                           act_test_observations)
            loss_history.append(loss)

        return self._loss_aggregation_method(loss_history)

    def _fit_model(self,
             model: _ModelBuildingBase,
             observation_index_points,
             observations) -> _ModelBuildingBase:
        ''' fits the model, the model is changed '''
        gp = model.build_for_training(observation_index_points, observations)
        self._fit_gp(gp, observations)
        return model


# ###############################################################################
# ### MODELS
# ###############################################################################


class _GaussianProcessModel:
    ''' gaussian process wihtout known kernel '''

    def __init__(self, parameters: Parameters, kernel_cls):
        self.parameters = parameters
        self.KERNEL_CLS = kernel_cls

        self.MODEL_GENERATION_CLS = _ModelBuildingBase.create_class(self.parameters)
        self.MODEL_TRAINING_CLS = _ModelTrainingBase.create_class(self.parameters)

        self.model = self.MODEL_GENERATION_CLS(self.parameters, self.KERNEL_CLS)
        self.model_training = self.MODEL_TRAINING_CLS(self.parameters)

    def _fit(self, X: XType, F: YType, W: YType):
        self.model = self.model_training.fit_model(self.model, X, F)
        return self

    def _predict(self, X: XType) -> YType:
        gprm = self.model.build_for_regression(X)
        return gprm.mean().numpy()

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        gprm = self.model.build_for_regression(X)
        mean = gprm.mean().numpy()
        stddev = gprm.stddev().numpy()
        return mean, stddev

    def _loss(self, X, F):
        return self.model_training.compute_loss(self.model, X, F)

    def df(self) -> int:
        return 0


class GaussianProcess(_GaussianProcessModel, SurrogateModelBase):
    def __init__(self, parameters: Parameters):
        SurrogateModelBase.__init__(self, parameters)
        # loads the kernel form settings
        KERNEL_CLS = eval(self.parameters.surrogate_model_gp_kernel)
        _GaussianProcessModel.__init__(self, parameters, KERNEL_CLS)


class _GaussianProcessModelMixtureBase:
    TRAIN_MAX_MODELS: Optional[int] = None
    TRAIN_MAX_TIME_S: Optional[int] = None

    def __init__(self, parameters: Parameters) -> None:
        self.parameters = parameters

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
        return self._building_blocks

    def _generate_model_space(self):
        for kernel_cls in self._generate_kernel_space():
            model = _GaussianProcessModel(self.parameters, kernel_cls)
            yield model

    def _fit(self, X: XType, F: YType, W: YType):
        time_start = time.time()

        models = []
        losses = []

        for model in itertools.islice(
                self._generate_model_space(), self.TRAIN_MAX_MODELS):
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
    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) in [(Linear, Linear), (Quadratic, Quadratic), ]:
                continue
            yield a + b


class GaussianProcessBasicMultiplicativeSelection(GaussianProcessBasicSelection):
    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) in [(Linear, Linear), ]:
                continue
            yield a * b


class GaussianProcessBasicBinarySelection(GaussianProcessBasicSelection):
    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) not in [(Linear, Linear), ]:
                yield a * b
            if (a, b) not in [(Linear, Linear), (Quadratic, Quadratic), ]:
                yield a + b


'''
class GaussianProcessPenalizedAdditiveSelection(GaussianProcessBasicSelection):
    def penalize_kernel(self, loss, kernel_obj):
        return super().penalize_kernel(loss, kernel_obj)

    def _predict(self, X: XType) -> YType:
        return self.best_model._predict(X)

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return self.best_model._predict_with_confidence(X)

'''
