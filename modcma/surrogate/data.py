import numpy as np
import math

from typing import Callable, List, Union, Optional, Any
from numpy.typing import NDArray
from scipy.stats import kendalltau
from abc import ABCMeta, abstractmethod, abstractproperty

from modcma.parameters import Parameters
from modcma.typing_utils import XType, YType, xType, yType, NDArrayInt

from modcma.utils import normalize_str_eq


class SurrogateDataBase(metaclass=ABCMeta):
    FIELDS = ['_X', '_X_mahal', '_F', '_TIME']

    def __init__(self, settings: Parameters):
        self.settings = settings

        # (#Samples, Dimensions)
        self._X: XType = np.empty(0, settings.d, dtype=np.float64)
        self._F: YType = np.empty(0, 1, dtype=np.float64)
        self._TIME: NDArrayInt = np.empty(0, 1, dtype=np.int_)
        self._act_time = 0

        # mahalanobis cache
        self._X_mahal: Optional[XType] = None
        self._converted_m = None
        self._converted_inv_root_C = None

    def push(self, x, f: Union[YType, float]):
        """ Push element to the archive """
        self._act_time += 1
        x = np.array(x).reshape(1, self.settings.d)
        f = np.array(f).reshape(1, 1)

        self._X = np.vstack([self._X, x])
        self._X_mahal = None
        self._F = np.vstack([self._F, f])
        self._TIME = np.vstack([self._TIME, self._act_time])

    def push_many(self, X, F, time_ordered=False):
        """ Push multiple elements into the archive """
        self._act_time += 1
        F = np.array(F).reshape(-1, 1)
        assert (X.shape[1] == self.settings.d)
        assert (X.shape[0] == F.shape[0])

        self._X = np.vstack([self._X, X])
        self._X_mahal = None
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
        self._X_mahal = None
        self._F = self._F[number:]
        self._TIME = self._TIME[number:]
        return x, f

    def __len__(self) -> int:
        """ Returns the number of saved samples (not necessary equals to the training size) """
        return self._F.shape[0]

    def _update_X_mahal(self):
        """ checks if the conversion to the mahalanobis space is up-to-date"""
        if self._X_mahal is None \
                or len(self._X) != len(self._X_mahal) \
                or self._converted_m != self.settings.m \
                or self._converted_inv_root_C != self.settings.inv_root_C:
            self._converted_m = self.settings.m
            self._converted_inv_root_C = self.settings.inv_root_C
            self._X_mahal = (self._X - self.settings.m.T) @ self.settings.inv_root_C.T

    def _compute_order_measure(self, selection: slice = slice(None)):
        """ returns the preference of the samples """
        sort_method = self.settings.surrogate_data_sorting

        if normalize_str_eq(sort_method, 'time'):
            measure = -self._TIME[selection]
        elif normalize_str_eq(sort_method, 'lq'):
            measure = self._F[selection]
        elif normalize_str_eq(sort_method, 'mahalanobis'):
            self._update_X_mahal()
            measure = np.sum(np.square(self._X_mahal[selection]), axis=1, keepdims=True)
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

    @property
    def model_size(self) -> int:
        """ number of samples selected for training a surrogate model """
        size = len(self)

        # absolute max
        if self.settings.surrogate_data_max_size_absolute is not None:
            size = min(size, self.settings.surrogate_data_max_size_absolute)

        # relative max
        if self.settings.surrogate_data_max_size_relative_dof is not None:
            if self.settings.surrogate_model_instance is not None:
                if self.settings.surrogate_model_instance.dof > 0:
                    t = self.settings.surrogate_data_max_size_relative_dof * self.settings.surrogate_model_instance.dof
                    size = min(size, int(math.ceil(t)))
        return size

        # truncation ratio
        # if self.settings.surrogate_data_truncation_ratio is not None:
        #    size = int(math.ceil(size * self.settings.surrogate_data_truncation_ratio))
        # return size

    # MODEL BUILDING BUSINESS
    @property
    def X(self) -> Optional[XType]:  # Covariates
        # TODO: return mahalanobis
        if self._X is None:
            return None

        #
        X = self._X[-self.model_size:]
        if self.settings.surrogate_data_mahalanobis_space:
            # TODO: check/fix dimensions self.settings.inv_root_C @ (X - self.settings.m.T).T ???
            X = (self.settings.inv_root_C @ (X.T - self.settings.m)).T
        return X

    @property
    def F(self) -> Optional[YType]:  # Target Values
        if self._F is None:
            return None
        return self._F[-self.model_size:]

    @property
    def W(self):  # Weight
        if self.settings.surrogate_data_weighting == 'constant':
            return np.ones(self.model_size)
        elif self.settings.surrogate_data_weighting == 'logarithmic':
            assert self.settings.surrogate_data_min_weight > 0.
            assert self.settings.surrogate_data_max_weight > 0.
            return np.logspace(np.log10(self.settings.surrogate_data_max_weight),
                               np.log10(self.settings.surrogate_data_min_weight),
                               num=self.model_size)
            pass
        elif self.settings.surrogate_data_weighting == 'linear':
            return np.linspace(self.settings.surrogate_data_min_weight,
                               self.settings.surrogate_data_max_weight,
                               num=self.model_size)
        else:
            raise NotImplementedError("Couldn't interpret the weight_function")


'''
#####################
# Population Storage Management

class FilterUnique(Filter):
    def __call__(self, pop: PopHistory) -> PopHistory:
        _, ind = np.unique(pop.x, axis=1, return_index=True)
        return pop[ind]


class FilterDistance(Filter):
    def __init__(self, parameters: Parameters, distance: float):
        self.max_distance = distance
        self.parameters = parameters

    @abstractmethod
    def _compute_distance(self, pop: PopHistory) -> npt.NDArray[np.float32]:
        pass

    def _get_mask(self, pop: PopHistory) -> npt.NDArray[np.bool_]:
        distance = self._compute_distance(pop)
        return distance <= self.max_distance

    def __call__(self, pop: PopHistory) -> PopHistory:
        mask = self._get_mask(pop)
        return pop[mask]


class FilterDistanceMahalanobis(FilterDistance):
    def __init__(self, parameters: Parameters, distance: float):
        super().__init__(parameters, distance)
        B = self.parameters.B
        sigma = self.parameters.sigma
        D = self.parameters.D

        self.transformation = np.linalg.inv(B) @ np.diag((1./sigma)*(1./D))

    def _compute_distance(self, pop: PopHistory) -> npt.NDArray[np.float32]:
        center_x = pop.x - self.parameters.m
        return np.sqrt(self.transformation @ center_x)


FILTER_TYPE = Union[
    FilterRealEvaluation,
    FilterUnique,
    FilterDistanceMahalanobis
]

'''
