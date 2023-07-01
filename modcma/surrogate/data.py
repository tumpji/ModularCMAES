import collections.abc
import math
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
import sklearn.model_selection

from modcma.parameters import Parameters
from modcma.typing_utils import XType, YType, NDArrayInt, NDArrayBool
from modcma.utils import normalize_str_eq


class SurrogateDataBase:
    FIELDS = ['_X', '_F', '_TIME']

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

        # (#Samples, Dimensions)
        self._X: XType = np.empty((0, parameters.d), dtype=np.float64)
        self._F: YType = np.empty((0, 1), dtype=np.float64)
        self._TIME: NDArrayInt = np.empty((0, 1), dtype=np.int_)
        self._act_time = 0

    def push(self, x, f: Union[YType, float]):
        """ Push element to the archive """
        self._act_time += 1
        x = np.array(x).reshape(1, self.parameters.d)
        f = np.array(f).reshape(1, 1)

        # checks for equality
        if self.parameters.surrogate_data_equality_removal:
            if np.any(np.all(np.equal(self._X, x), axis=1)):
                return

        self._X = np.vstack([self._X, x])
        self._F = np.vstack([self._F, f])
        self._TIME = np.vstack([self._TIME, self._act_time])

    def push_many(self, X, F, time_ordered=False):
        """ Push multiple elements into the archive """
        if len(F.shape) != 2 or F.shape[1] != 1:
            raise ValueError(f'Values have wrong dimensions {F.shape} expected {(len(X), 1)}')
        if len(X.shape) != 2 or X.shape[1] != self.parameters.d:
            raise ValueError(f'Features have wrong dimensions {X.shape} expected {(len(X), self.parameters.d)}')
        if len(X) != len(F):
            raise ValueError(f'Number of features and values does not match X: {X.shape}, F: {F.shape}')

        self._act_time += 1

        # checks for equality
        if self.parameters.surrogate_data_equality_removal:
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
        return (X - self.parameters.m.T) @ self.parameters.inv_root_C.T

    def _compute_order_measure(self, selection: slice = slice(None)) -> YType:
        """ returns the preference for the samples - less is better """
        sort_method = self.parameters.surrogate_data_sorting

        if normalize_str_eq(sort_method, 'time'):
            measure = -self._TIME[selection, 0]
        elif normalize_str_eq(sort_method, 'lq'):
            measure = self._F[selection, 0]
        elif normalize_str_eq(sort_method, 'euclidean'):
            measure = np.sum(np.square(self._X[selection] - self.parameters.m.T), axis=1)
        elif normalize_str_eq(sort_method, 'mahalanobis'):
            measure = np.sum(np.square(self._to_mahalanobis(self._X[selection])), axis=1)
        else:
            raise NotImplementedError(f'The sorting method {sort_method} is not implemented.')
        return measure

    def sort_all(self):
        """ sorts all elements based on surrogate_data_sorting, 'best' are in the last indexes """
        if len(self) <= 1:
            return

        order = np.argsort(-self._compute_order_measure(), axis=0)

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
        if self.parameters.surrogate_data_max_size_absolute is not None:
            return min(len(self), self.parameters.surrogate_data_max_size_absolute)
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
        if self.parameters.surrogate_data_weighting == 'constant':
            return np.ones(self._max_training_size)
        elif self.parameters.surrogate_data_weighting == 'logarithmic':
            assert self.parameters.surrogate_data_min_weight > 0.
            assert self.parameters.surrogate_data_max_weight > 0.
            return np.logspace(np.log10(self.parameters.surrogate_data_max_weight),
                               np.log10(self.parameters.surrogate_data_min_weight),
                               num=self._max_training_size)
            pass
        elif self.parameters.surrogate_data_weighting == 'linear':
            return np.linspace(self.parameters.surrogate_data_min_weight,
                               self.parameters.surrogate_data_max_weight,
                               num=self._max_training_size)
        else:
            raise NotImplementedError("Couldn't interpret the weight_function")


class SurrogateData_V2(SurrogateData_V1):
    """ Uses boolean selection instead of slices to extract data from the archive """
    FIELDS = ['_X', '_F', '_X_mahal', '_X_mahal_norm', '_TIME']

    def __init__(self, parameters):
        super().__init__(parameters)

        # mahalanobis cache management
        self._X_mahal: Optional[XType] = None
        self._X_mahal_norm = None
        self._converted_m = None
        self._converted_inv_root_C = None

        # mask cache management
        self._selection_cache = None

    def _update_X_mahal(self, compute_norm=False) -> None:
        """ checks if the conversion to the mahalanobis space is up-to-date"""
        if self._X_mahal is None \
                or len(self._X) != len(self._X_mahal) \
                or self._converted_m != self.parameters.m \
                or self._converted_inv_root_C != self.parameters.inv_root_C:
            self._converted_m = self.parameters.m
            self._converted_inv_root_C = self.parameters.inv_root_C
            self._X_mahal = self._to_mahalanobis(self._X)
            self._X_mahal_norm = None

        if compute_norm:
            self._X_mahal_norm = np.sqrt(np.sum(np.square(self._X_mahal), axis=1, keepdims=True))

    def push(self, x, f: Union[YType, float]):
        self._X_mahal = None
        self._selection_cache = None
        return super().push(x, f)

    def push_many(self, X, F, time_ordered=False):
        self._X_mahal = None
        self._selection_cache = None
        return super().push_many(X, F, time_ordered)

    def pop(self, number: int = 1):
        self._X_mahal = None
        self._selection_cache = None
        return super().pop(number)

    def sort_all(self):
        self._selection_cache = None
        return super().sort_all()

    def sort(self, n: Optional[int] = None) -> None:
        self._selection_cache = None
        return super().sort(n)

    def _compute_order_measure(self, selection: slice = slice(None)) -> YType:
        """ returns the preference for the samples """
        sort_method = self.parameters.surrogate_data_sorting

        if normalize_str_eq(sort_method, 'mahalanobis'):
            self._update_X_mahal(compute_norm=True)
            return self._X_mahal_norm[selection, 0]
        else:
            return super()._compute_order_measure(selection)

    @property
    def _selection(self) -> NDArrayBool:
        """ computes indexes of training data """
        if self._selection_cache is None:
            # mahalanobis distance handling ...
            if self.parameters.surrogate_data_mahalanobis_space \
                    and self.parameters.surrogate_data_mahalanobis_space_max_value is not None:
                self._update_X_mahal(compute_norm=True)
                mask = self._X_mahal_norm <= self.parameters.surrogate_data_mahalanobis_space_max_value
            else:
                mask = np.ones(len(self))

            # mask is finished, now apply the maximum number of elements
            assert mask.shape[0] == len(self)
            self._selection_cache, = np.where(mask)

            # now apply the maximum number of elements
            self._selection_cache = self._selection_cache[-self._max_training_size:]
        return self._selection_cache

    @property
    def X(self) -> Optional[XType]:  # Covariates
        if self.parameters.surrogate_data_mahalanobis_space:
            self._update_X_mahal()
            return self._X_mahal[self._selection]
        else:
            return self._X[self._selection]

    @property
    def F(self) -> Optional[YType]:  # Target Values
        return self._F[self._selection]

    @property
    def W(self):  # Weight
        no_elems = len(self._selection)

        if self.parameters.surrogate_data_weighting == 'constant':
            return np.ones(no_elems)
        elif self.parameters.surrogate_data_weighting == 'logarithmic':
            assert self.parameters.surrogate_data_min_weight > 0.
            assert self.parameters.surrogate_data_max_weight > 0.
            return np.logspace(np.log10(self.parameters.surrogate_data_max_weight),
                               np.log10(self.parameters.surrogate_data_min_weight),
                               num=no_elems)
            pass
        elif self.parameters.surrogate_data_weighting == 'linear':
            return np.linspace(self.parameters.surrogate_data_min_weight,
                               self.parameters.surrogate_data_max_weight,
                               num=no_elems)
        else:
            raise NotImplementedError("Couldn't interpret the weight_function")


class SurrogateData_V3(SurrogateData_V2):
    """ Adds model selection to the mixture """

    def set_up_model_selection(self):
        return Folding(
            self.parameters.surrogate_data_cross_validation_folds,
            self.parameters.numpy_rng,
            self.X, self.F, self.W,
        )


class Folding(collections.abc.Iterator):
    @dataclass
    class CrossValidationInstance:
        Xtrain: XType
        Xtest: XType
        Ftrain: YType
        Ftest: YType
        Wtrain: YType
        Wtest: YType

    def __init__(self, parameters: Parameters, X: XType, F: YType, W: YType):
        self.parameters: Parameters = parameters
        self._X, self._F, self._W = X, F, W

        self.folds: Optional[int] = self.parameters.surrogate_data_cross_validation_folds
        self.train_test_ratio: float = self.parameters.surrogate_data_train_test_ratio
        self.numpy_rng = self.parameters.numpy_rng

        self.index: Optional[int] = None

    def __iter__(self):
        if self.folds is not None and self.folds > 1:
            self._iterator = enumerate(sklearn.model_selection.KFold(
                self.folds, shuffle=True, random_state=self.numpy_rng
            ).split(self._X, self._F))
        elif self.folds < 0:
            raise ValueError("Negative values are not allowed to set number of folds")
        else:
            # Train-test split
            h = np.arange(len(self._X))
            self.numpy_rng.shuffle(h)
            len_test = math.ceil(self.train_test_ratio * len(h))
            self._iterator = iter(enumerate([(h[len_test:], h[:len_test]), ]))

    def __next__(self):
        self.index, (train_ind, test_ind) = next(self._iterator)
        return Folding.CrossValidationInstance(
            Xtrain=self._X[train_ind],
            Xtest=self._X[test_ind],
            Ytrain=self._F[train_ind],
            Ytest=self._F[test_ind],
            Wtrain=self._W[train_ind],
            Wtest=self._W[test_ind]
        )
