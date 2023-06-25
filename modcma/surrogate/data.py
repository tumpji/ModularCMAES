import numpy as np
import math

from typing import Callable, List, Union, Optional, Any
from numpy.typing import NDArray
from scipy.stats import kendalltau
from abc import ABCMeta, abstractmethod, abstractproperty

from modcma.parameters import Parameters
from modcma.typing_utils import XType, YType, xType, yType, NDArrayInt, NDArrayBool

from modcma.utils import normalize_str_eq


class SurrogateDataBase(metaclass=ABCMeta):
    FIELDS = ['_X', '_F', '_TIME']

    def __init__(self, settings: Parameters):
        self.settings = settings

        # (#Samples, Dimensions)
        self._X: XType = np.empty(0, settings.d, dtype=np.float64)
        self._F: YType = np.empty(0, 1, dtype=np.float64)
        self._TIME: NDArrayInt = np.empty(0, 1, dtype=np.int_)
        self._act_time = 0

    def push(self, x, f: Union[YType, float]):
        """ Push element to the archive """
        self._act_time += 1
        x = np.array(x).reshape(1, self.settings.d)
        f = np.array(f).reshape(1, 1)

        # checks for equality
        if self.settings.surrogate_data_equality_removal:
            if np.any(np.all(np.equal(self._X, x), axis=1)):
                return

        self._X = np.vstack([self._X, x])
        self._F = np.vstack([self._F, f])
        self._TIME = np.vstack([self._TIME, self._act_time])

    def push_many(self, X, F, time_ordered=False):
        """ Push multiple elements into the archive """
        self._act_time += 1
        F = np.array(F).reshape(-1, 1)
        assert (X.shape[1] == self.settings.d)
        assert (X.shape[0] == F.shape[0])

        # checks for equality
        if self.settings.surrogate_data_equality_removal:
            selection = [np.any(np.all(np.equal(self._X, x), axis=1)) for x in X]
            F = F[selection]
            X = X[selection]

        self._X = np.vstack([self._X, X])
        self._F = np.vstack([self._F, F])
        if time_ordered:
            time_order = np.arange(self._act_time, self._act_time + len(F)).reshape(-1, 1)
            self._act_time += len(F) - 1
        else:
            time_order = np.repeat(self._act_time, len(F)).reshape(-1, 1)
        self._TIME = np.vstack([self._TIME, time_order])

    def pop(self, number: int = 1):
        """ Removes @number of elements from the beginning of the stack (default=1) and returns them.
            If ordered, it corresponds to the least relevant
        """
        x = self._X[:number]
        f = self._F[:number]
        self._X = self._X[number:]
        self._F = self._F[number:]
        self._TIME = self._TIME[number:]
        return x, f

    def __len__(self) -> int:
        """ Returns the number of saved samples (not necessary equals to the training size) """
        return self._F.shape[0]

    def _to_mahalanobis(self, X):
        return (X - self.settings.m.T) @ self.settings.inv_root_C.T

    def _compute_order_measure(self, selection: slice = slice(None)) -> YType:
        """ returns the preference for the samples """
        sort_method = self.settings.surrogate_data_sorting

        if normalize_str_eq(sort_method, 'time'):
            measure = -self._TIME[selection]
        elif normalize_str_eq(sort_method, 'lq'):
            measure = self._F[selection]
        elif normalize_str_eq(sort_method, 'mahalanobis'):
            measure = np.sum(np.square(self._to_mahalanobis(self._X[selection])), axis=1, keepdims=True)
        else:
            raise NotImplementedError(f'The sorting method {sort_method} is not implemented.')

        assert measure.shape[1] == 1
        return measure

    def sort_all(self):
        """ sorts all elements based on surrogate_data_sorting """
        if len(self) <= 1:
            return

        order = np.argsort(self._compute_order_measure(), axis=0)

        for name in self.FIELDS:
            data = getattr(self, name)
            if data is not None:
                setattr(self, name, data[order])

    def sort(self, n: Optional[int] = None) -> None:
        """ sorts latest n elements based on surrogate_data_sorting """

        if n is None:
            return self.sort_all()

        if n <= 1 or len(self) <= 1:
            return

        n = min(len(self), n)
        select: slice = slice(-n, None)
        other: slice = slice(None, -n)

        order = np.argsort(self._compute_order_measure(select), axis=0)

        for name in self.FIELDS:
            data = getattr(self, name)
            if data is not None:
                new_data = [data[other], data[select][order]]
                setattr(self, name, np.vstack(new_data))


class SurrogateData_V1(SurrogateDataBase):
    """ In this version of surrogate data only the maximum number of samples are considered. No other filtering is
    present."""

    @property
    def _max_training_size(self) -> int:
        """ number of samples selected for training a surrogate model """
        # absolute max
        if self.settings.surrogate_data_max_size_absolute is not None:
            return min(len(self), self.settings.surrogate_data_max_size_absolute)
        return len(self)

    @property
    def X(self) -> Optional[XType]:  # Covariates
        """ returns all training data """
        return self._X[-self._max_training_size:]

    @property
    def F(self) -> Optional[YType]:  # Target Values
        return self._F[-self._max_training_size:]

    @property
    def W(self):  # Weight
        if self.settings.surrogate_data_weighting == 'constant':
            return np.ones(self._max_training_size)
        elif self.settings.surrogate_data_weighting == 'logarithmic':
            assert self.settings.surrogate_data_min_weight > 0.
            assert self.settings.surrogate_data_max_weight > 0.
            return np.logspace(np.log10(self.settings.surrogate_data_max_weight),
                               np.log10(self.settings.surrogate_data_min_weight),
                               num=self._max_training_size)
            pass
        elif self.settings.surrogate_data_weighting == 'linear':
            return np.linspace(self.settings.surrogate_data_min_weight,
                               self.settings.surrogate_data_max_weight,
                               num=self._max_training_size)
        else:
            raise NotImplementedError("Couldn't interpret the weight_function")

''' 
class SurrogateData_V2(SurrogateDataBase):
    """ """
    # MODEL BUILDING BUSINESS
    @property
    def X(self) -> Optional[XType]:  # Covariates
        # TODO: return mahalanobis
        if self._X is None:
            return None

        #
        X = self._X[-self._max_model_size:]
        if self.settings.surrogate_data_mahalanobis_space:
            # TODO: check/fix dimensions self.settings.inv_root_C @ (X - self.settings.m.T).T ???
            X = (self.settings.inv_root_C @ (X.T - self.settings.m)).T
        return X

    @property
    def F(self) -> Optional[YType]:  # Target Values
        if self._F is None:
            return None
        return self._F[-self._max_model_size:]

    @property
    def W(self):  # Weight
        if self.settings.surrogate_data_weighting == 'constant':
            return np.ones(self._max_model_size)
        elif self.settings.surrogate_data_weighting == 'logarithmic':
            assert self.settings.surrogate_data_min_weight > 0.
            assert self.settings.surrogate_data_max_weight > 0.
            return np.logspace(np.log10(self.settings.surrogate_data_max_weight),
                               np.log10(self.settings.surrogate_data_min_weight),
                               num=self._max_model_size)
            pass
        elif self.settings.surrogate_data_weighting == 'linear':
            return np.linspace(self.settings.surrogate_data_min_weight,
                               self.settings.surrogate_data_max_weight,
                               num=self._max_model_size)
        else:
            raise NotImplementedError("Couldn't interpret the weight_function")
