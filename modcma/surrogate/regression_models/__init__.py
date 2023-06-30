import typing

from modcma.surrogate.regression_models.model import \
    SurrogateModelBase, \
    SklearnSurrogateModelBase

from modcma.surrogate.regression_models.polynomial import \
    Linear_SurrogateModel, \
    QuadraticPure_SurrogateModel, \
    QuadraticInteraction_SurrogateModel, \
    Quadratic_SurrogateModel, \
    LQ_SurrogateModel

from modcma.surrogate.regression_models.gaussian_processes import *

def get_model(model_name: str) -> typing.Type[SurrogateModelBase]:
    from modcma.utils import normalize_string, all_subclasses, normalize_str_eq
    sur_model_to_find = normalize_string(model_name)

    sur_model_classes = all_subclasses(SurrogateModelBase)
    for sur_model_cls in sur_model_classes:
        if normalize_str_eq(sur_model_cls.ModelName, sur_model_to_find):
            sur_model_cls: Type[SurrogateModelBase]
            return sur_model_cls

    raise NotImplementedError(
        f'Cannot find model with name "{model_name}"')
