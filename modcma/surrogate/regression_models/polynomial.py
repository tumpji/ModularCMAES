from typing import Optional

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from typing_extensions import override

from modcma.surrogate.regression_models import SklearnSurrogateModelBase, SurrogateModelBase
from modcma.typing_utils import XType, YType


# TODO: add lasso and ridge regression
# TODO: Adjusted R^2 to switch between models
# TODO: Interaction only PolynomialFeatures
# TODO: Cubic model ?

class PureQuadraticFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X: XType, y: YType = None):
        return self

    def transform(self, X: XType) -> XType:
        return np.hstack((X, np.square(X)))


class Linear_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'Linear'

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('linearregression', LinearRegression())
        ]).fit(X, F, linearregression__sample_weight=W)

    @property
    def df(self) -> int:
        return self.parameters.d + 1


class QuadraticPure_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'QuadraticPure'

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('quad.f.', PureQuadraticFeatures()),
            ('linearregression', LinearRegression())
        ]).fit(X, F, linearregression__sample_weight=W)

    @property
    def df(self) -> int:
        return 2 * self.parameters.d + 1


class QuadraticInteraction_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'QuadraticInteraction'

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           interaction_only=True,
                                           include_bias=False)),
            ('linearregression', LinearRegression())
        ]).fit(X, F, linearregression__sample_weight=W)

    @property
    def df(self) -> int:
        return (self.parameters.d * (self.parameters.d + 1) + 2) // 2


class Quadratic_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'Quadratic'

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           include_bias=False)),
            ('linearregression', LinearRegression())
        ]).fit(X, F, linearregression__sample_weight=W)

    @property
    def df(self):
        return (self.parameters.d + 2) * (self.parameters.d + 1) // 2


class LQ_SurrogateModel(SurrogateModelBase):
    ModelName = 'LQ'
    SAFETY_MARGIN: float = 1.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: Optional[Pipeline] = None
        self._dof: int = self.parameters.d + 1
        self.i_model: int = 0

    def _select_model(self, N: int, D: int) -> Pipeline:
        # model             degree of freedom
        # linear            D + 1
        # quadratic         2D + 1
        # full-quadratic    C_r(D, 1) + C_r(D, 2) = (D^2 + 3D)/2 + 1

        margin = self.parameters.surrogate_model_lq_margin

        if N >= margin * ((D ** 2 + 3 * D) / 2 + 1):
            ppl = [('full-quadratic',
                    PolynomialFeatures(degree=2, include_bias=False))]
            self._dof = (self.parameters.d ** 2 + 3 * self.parameters.d + 2) // 2
            self.i_model = 2
        elif N >= margin * (2 * D + 1):
            ppl = [('pure-quadratic', PureQuadraticFeatures())]
            self._dof = 2 * self.parameters.d + 1
            self.i_model = 1
        else:
            ppl = []
            self._dof = self.parameters.d + 1
            self.i_model = 0
        return Pipeline(ppl + [('linearregression', LinearRegression())])

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        (N, D) = X.shape
        self.model = self._select_model(N, D)
        self.model = self.model.fit(X, F, linearregression__sample_weight=W)

    @override
    def _predict(self, X: XType) -> YType:
        if self.model is None:
            return super().predict(X)
        return self.model.predict(X)

    @property
    def df(self) -> int:
        return self._dof

    @property
    def max_df(self) -> int:
        return (self.parameters.d ** 2 + 3 * self.parameters.d) // 2 + 1