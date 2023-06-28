from typing import Type

from modcma.surrogate.regression_models.deep_gaussian_processes import DeepGaussianProcessStochImp
from modcma.surrogate.regression_models.model import SklearnSurrogateModelBase
from modcma.surrogate.regression_models.model import SurrogateModelBase
from modcma.surrogate.regression_models.model_gp import GaussianProcess
from modcma.surrogate.regression_models.model_gp_basic_selection import GaussianProcessBasicAdditiveSelection
from modcma.surrogate.regression_models.model_gp_basic_selection import GaussianProcessBasicBinarySelection
from modcma.surrogate.regression_models.model_gp_basic_selection import GaussianProcessBasicMultiplicativeSelection
from modcma.surrogate.regression_models.model_gp_basic_selection import GaussianProcessBasicSelection
from modcma.surrogate.regression_models.model_gp_basic_selection import GaussianProcessGreedySearch
from modcma.surrogate.regression_models.model_gp_basic_selection import GaussianProcessHeuristic
from modcma.surrogate.regression_models.polynomial import Linear_SurrogateModel, QuadraticPure_SurrogateModel, \
    QuadraticInteraction_SurrogateModel, Quadratic_SurrogateModel, LQ_SurrogateModel
from modcma.utils import normalize_string, all_subclasses, normalize_str_eq


def get_model(model_name: str) -> Type[SurrogateModelBase]:
    sur_model_to_find = normalize_string(model_name)

    sur_model_classes = all_subclasses(SurrogateModelBase)
    for sur_model_cls in sur_model_classes:
        if normalize_str_eq(sur_model_cls.ModelName, sur_model_to_find):
            sur_model_cls: Type[SurrogateModelBase]
            return sur_model_cls

    raise NotImplementedError(
        f'Cannot find model with name "{model_name}"')


__all__ = (
    'get_model',
    'SurrogateModelBase',
    'SklearnSurrogateModelBase',

    'Linear_SurrogateModel',
    'QuadraticPure_SurrogateModel',
    'QuadraticInteraction_SurrogateModel',
    'Quadratic_SurrogateModel',
    'LQ_SurrogateModel',

    'GaussianProcess',
    'DeepGaussianProcessStochImp',

    'GaussianProcessBasicSelection',
    'GaussianProcessBasicAdditiveSelection',
    'GaussianProcessBasicMultiplicativeSelection',
    'GaussianProcessBasicBinarySelection',
    'GaussianProcessGreedySearch',
    'GaussianProcessHeuristic'
)
