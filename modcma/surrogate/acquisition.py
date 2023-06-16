from abc import ABC, abstractmethod
from typing import Optional, Type

from scipy.special import ndtr
from typing_extensions import override

from modcma.parameters import Parameters
from modcma.typing_utils import XType, YType, yType
from modcma.utils import normalize_string, normalize_str_eq


class AcquisitionFunctionBase(ABC):
    AcquisitionName = "Base"

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()

    @abstractmethod
    def calculate(self, x: XType, y: YType, std: Optional[YType] = None, est_min: yType = None):
        pass

    @staticmethod
    def check_shapes(x: XType, y: YType, std: Optional[YType] = None):
        assert x.shape[0] == y.shape[0]

        if std is not None:
            assert std.shape[0] == y.shape[0]


class ProbabilityOfImprovement(AcquisitionFunctionBase):
    AcquisitionName = "POI"

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        self.exploration_xi = parameters.acquisition_poi_xi

    @staticmethod
    def poi(mean, std, est_min, exploration_xi=1e-3):
        """
        Probability of improvement function for minimization tasks.
        """

        # scale tradeoff for minimum
        tradeoff = exploration_xi * est_min
        return ndtr((est_min - mean - tradeoff) / std)

    @override
    def calculate(self, x: XType, y: YType, std: Optional[YType] = None, est_min: yType = None):
        self.check_shapes(x, y, std)
        return self.poi(y, std, est_min)


def get_acquisition(parameters: Parameters) -> AcquisitionFunctionBase:
    acq_name_to_find = normalize_string(parameters.acquisition_function)

    acq_func_classes = AcquisitionFunctionBase.__subclasses__()
    for acq_cls in acq_func_classes:
        if normalize_str_eq(acq_cls.AcquisitionName, acq_name_to_find):
            acq_cls: Type[AcquisitionFunctionBase]
            return acq_cls(parameters)

    raise NotImplementedError(
        f'Cannot find acquisition function with name "{parameters.acquisition_function}"')
