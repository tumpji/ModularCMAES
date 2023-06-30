from typing import Optional, Tuple

import dgpsi
import numpy as np

from modcma import Parameters
from modcma.surrogate.regression_models.model import SurrogateModelBase
from modcma.typing_utils import XType, YType


class DeepGaussianProcessStochImp(SurrogateModelBase):
    ModelName = "DGP"

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        # TODO load from parameters

        self.training_iter = 50
        self.num_imputations = 100
        self.prediction_method = "mean_var"

        lay1 = [
            dgpsi.kernel(length=np.array([1]), name='sexp'),
            dgpsi.kernel(length=np.array([1]), name='sexp')
        ]
        lay2 = [
            dgpsi.kernel(length=np.array([1]), name='sexp', connect=np.arange(2)),
            dgpsi.kernel(length=np.array([1]), name='sexp', connect=np.arange(2))
        ]
        lay3 = [
            dgpsi.kernel(length=np.array([1]), name='sexp', scale_est=True, connect=None)
        ]
        self.deep_gp_layers = dgpsi.combine(lay1, lay3)

    def _fit(self, X: Optional[XType], F: Optional[YType], W: Optional[YType] = None):
        if F.ndim == 1:
            F = np.expand_dims(F, axis=-1)
        assert F.ndim == 2

        self.model = dgpsi.dgp(X, F, self.deep_gp_layers)
        self.model.train(N=self.training_iter)

        dgp_layers_est = self.model.estimate()
        self.emulator = dgpsi.emulator(dgp_layers_est, N=self.num_imputations)

        return self

    def _predict(self, X: XType) -> YType:
        mean, _ = self._predict_with_confidence(X)
        return mean

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        mean, variance = self.emulator.predict(x=X, method=self.prediction_method)
        return mean, variance

    @property
    def df(self) -> int:
        # TODO count degree of freedom
        return 0
