import numpy as np

from sklearn.pipeline import Pipeline

from abc import abstractmethod, ABCMeta
from typing import Optional, Tuple

from typing_extensions import override
from modcma.typing_utils import XType, YType
from modcma.utils import normalize_string

from modcma.parameters import Parameters

####################
# Helper functions

def normalize_X(X: XType, d):
    assert X.shape[1] == d
    return X


def normalize_F(Y: YType):
    return Y.ravel()


normalize_W = normalize_F


class SurrogateModelBase(metaclass=ABCMeta):
    ModelName = "Base"

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self._fitted = False

    def fit(self,
            X: Optional[XType],
            F: Optional[YType],
            W: Optional[YType] = None):
        ''' fit the surrogate '''
        if X is None or F is None:
            self._fitted = False
            return self

        X = normalize_X(X, self.parameters.d)
        F = normalize_F(F)

        if W is None:
            W = np.ones_like(F)
        else:
            W = normalize_W(W)
        self._fit(X, F, W)
        self._fitted = True
        return self

    @abstractmethod
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        pass

    def predict(self, X: XType) -> YType:
        F = self._predict(X)
        return normalize_F(F)

    @abstractmethod
    def _predict(self, X: XType) -> YType:
        return np.tile(np.nan, (len(X), 1))

    def predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        F, variance = self._predict_with_confidence(X)
        F = normalize_F(F)
        variance = normalize_F(variance)
        return F, variance

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        F = self.predict(X)
        return F, np.tile(np.nan, F.shape)

    @property
    def fitted(self) -> bool:
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool):
        self._fitted = value

    @property
    def df(self) -> int:
        return 0

    @property
    def max_df(self) -> int:
        return self.df

    @classmethod
    def name(cls) -> str:
        return normalize_string(cls.ModelName)


####################
# Other Surrogate Models


class SklearnSurrogateModelBase(SurrogateModelBase):
    @override
    def _predict(self, X: XType) -> YType:
        self.model: Pipeline
        return self.model.predict(X)

    '''
    def predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return super().predict_with_confidence(X)
    '''