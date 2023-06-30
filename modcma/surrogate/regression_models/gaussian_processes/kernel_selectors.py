import time
from abc import ABCMeta, abstractmethod, ABC
from typing import Optional, List, Type

from modcma import Parameters
from surrogate.gp_kernels import GP_kernel_concrete_base

from surrogate.regression_models.model_gp_basic_selection import RationalQuadratic, Linear, ExponentiatedQuadratic, \
    MaternFiveHalves, MaternOneHalf, MaternThreeHalves, ExpSinSquared, Quadratic


class KernelSpaceBase(metaclass=ABCMeta):
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    @abstractmethod
    def __iter__(self):
        # length
        self._max_length_count: int = 0
        self._max_length: Optional[int] = self.parameters.surrogate_model_selection_max_models

        # time
        self._max_time = None
        if self.parameters.surrogate_model_selection_max_seconds is not None:
            self._max_time = self.parameters.surrogate_model_selection_max_seconds + time.time()

        return self

    @abstractmethod
    def __next__(self):
        # length
        self._max_length_count += 1
        if self._max_length is not None and self._max_length <= self._max_length_count:
            raise StopIteration()

        # time
        if self._max_time is not None and time.time() >= self._max_time:
            raise StopIteration()

    @abstractmethod
    def provide_result(self, value):
        pass


class NeatKernelSpace(KernelSpaceBase, ABC):
    """ provides minimal kernel selection """

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)
        self._building_blocks: List[Type[GP_kernel_concrete_base]] = [
            RationalQuadratic,
            Linear,
            ExponentiatedQuadratic,
        ]

    def __iter__(self):
        return super().__iter__(self)

    def __next__(self):
        if self._max_length_count > len(self._building_blocks):
            raise StopIteration()
        return self._building_blocks[self._max_length_count - 1]


class ExtendedKernelSpace(NeatKernelSpace, ABC):
    def __init__(self, parameters: Parameters):
        super().__init__(parameters)
        self._building_blocks: List[Type[GP_kernel_concrete_base]] = [
            MaternFiveHalves,
            MaternOneHalf,
            MaternThreeHalves,
            RationalQuadratic,
            ExponentiatedQuadratic,
            ExpSinSquared,
            Linear,
            Quadratic,
        ]
