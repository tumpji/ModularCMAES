from abc import abstractmethod, ABCMeta
from typing import Tuple, Optional, Type, List, Generator, Any
from dataclasses import field, dataclass
import itertools
import time

import numpy as np

from modcma.parameters import Parameters
from modcma.typing_utils import XType, YType
from modcma.surrogate.model import SurrogateModelBase
from modcma.surrogate.model_gp import _GaussianProcessModel

# import kernels
from modcma.surrogate.gp_kernels import basic_kernels, functor_kernels, GP_kernel_concrete_base

for k in basic_kernels + functor_kernels:
    locals()[k.__name__] = k

# Stuff for statitc typing
MaternFiveHalves: Type[GP_kernel_concrete_base]
MaternOneHalf: Type[GP_kernel_concrete_base]
MaternThreeHalves: Type[GP_kernel_concrete_base]
RationalQuadratic: Type[GP_kernel_concrete_base]
ExponentiatedQuadratic: Type[GP_kernel_concrete_base]
ExpSinSquared: Type[GP_kernel_concrete_base]
Linear: Type[GP_kernel_concrete_base]
Quadratic: Type[GP_kernel_concrete_base]
Cubic: Type[GP_kernel_concrete_base]
Parabolic: Type[GP_kernel_concrete_base]
ExponentialCurve: Type[GP_kernel_concrete_base]
Constant: Type[GP_kernel_concrete_base]



# class _GaussianProcessModelMixtureBase:

class _GaussianProcessModelSelectionBase(SurrogateModelBase):
    # TODO: RENAME

    def __init__(self, parameters: Parameters) -> None:
        self.parameters = parameters
        self.MAX_MODELS = self.parameters.surrogate_model_selection_max_models
        self.MAX_TIME = self.parameters.surrogate_model_selection_max_seconds
        self.random_state = np.random.RandomState(
            self.parameters.surrogate_model_selection_random_state)

        # the selection ...
        self._building_blocks: List[Type[GP_kernel_concrete_base]] = [
            MaternFiveHalves,
            MaternOneHalf,
            MaternThreeHalves,
            RationalQuadratic,
            ExponentiatedQuadratic,
            ExpSinSquared,
            Linear,
            Quadratic,
            #### #Cubic,
            #Parabolic,
            #ExponentialCurve,
            #Constant,
        ]

    @abstractmethod
    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from self._building_blocks

    def _shuffle_kernel_space(self):
        space = np.array(list(self._generate_kernel_space()))
        if self.parameters.surrogate_model_selection_randomization:
            self.random_state.shuffle(space)
        yield from space

    def _fit(self, X: XType, F: YType, W: YType):
        time_start = time.time()
        models, losses = [], []

        for i, kernel in enumerate(self._shuffle_kernel_space()):
            if self.MAX_MODELS and i >= self.MAX_MODELS:
                break

            model = _GaussianProcessModel(self.parameters, kernel)
            loss = model.compute_loss(X, F, W)
            models.append(model)
            losses.append(loss)

            if self.MAX_TIME and (time.time() - time_start) >= self.MAX_TIME:
                break

        if np.all(np.isnan(losses)):
            self.best_model = None
        else:
            best_index = np.nanargmin(losses)
            self.best_model = models[best_index]
            self.best_model._fit(X, F, W)
        return self

    def _predict(self, X: XType) -> Optional[YType]:
        if self.best_model is None:
            return None
        return self.best_model._predict(X)

    def _predict_with_confidence(self, X: XType) -> Optional[Tuple[YType, YType]]:
        if self.best_model is None:
            return None
        return self.best_model._predict_with_confidence(X)

    def df(self):
        return 0


class GaussianProcessBasicSelection(_GaussianProcessModelSelectionBase):
    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()


class GaussianProcessBasicAdditiveSelection(_GaussianProcessModelSelectionBase):
    ''' <model> Gaussian Process model that chooses the best addition of two kernels'''

    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) in [(Linear, Linear), (Quadratic, Quadratic), ]:
                continue
            yield a + b


class GaussianProcessBasicMultiplicativeSelection(_GaussianProcessModelSelectionBase):
    ''' <model> Gaussian Process model that chooses the best multiplication of two kernels'''

    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) in [(Linear, Linear), ]:
                continue
            yield a * b


class GaussianProcessBasicBinarySelection(_GaussianProcessModelSelectionBase):
    ''' <model> Gaussian Process model that chooses the best
    addition or multiplication of two kernels'''

    def _generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super()._generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) not in [(Linear, Linear), ]:
                yield a * b
            if (a, b) not in [(Linear, Linear), (Quadratic, Quadratic), ]:
                yield a + b


# ##############################################################################
# #### << Search Based Methods >>
# ##############################################################################


class GaussianProcessGreedySearch(_GaussianProcessModelSelectionBase):
    ''' Expands the best node (based on the loss)
        The limit is:
            a) the best node is already expanded
            b) time
            c) number of expansion of kernels
    '''

    def _expand_node(self, base_kernel):
        if base_kernel is None:
            yield from self._basis
        else:
            for kernel in self._basis:
                yield base_kernel + kernel
                yield base_kernel * kernel

    def _shuffle_expand_node(self, base_kernel):
        space = np.array(list(self._expand_node(base_kernel)))
        if self.parameters.surrogate_model_selection_randomization:
            self.random_state.shuffle(space)
        yield from space

    def _init_time(self):
        self.time_start = time.time()
        self.models_checked = 0

    def _check_timeup(self, add_model_count=0):
        # time
        if self.MAX_TIME:
            if time.time() - self.time_start > self.MAX_TIME:
                return True

        # number of models
        self.models_checked += add_model_count
        if self.MAX_MODELS:
            if self.models_checked >= self.MAX_MODELS:
                return True

        return False

    def _search_method(self, X: XType, F: YType, W: YType):
        # implements greedy search
        self._init_time()
        best_state = None
        best_value = self._evaluate_kernel(Constant, X, F, W)

        while not self._check_timeup():
            new_states, new_values = [], []

            for new_state in self._shuffle_expand_node(best_state):
                new_values.append(self._evaluate_kernel(new_state, X, F, W))
                new_states.append(new_state)

                if self._check_timeup(add_model_count=1):
                    break

            if len(new_values) == 0:
                break

            best_new_index = np.argmin(new_values)

            if new_values[best_new_index] < best_value:
                best_value = new_values[best_new_index]
                best_state = new_states[best_new_index]
            else:
                break

        return best_state if best_state is not None else Constant

    def _evaluate_kernel(self, kernel, X: XType, F: YType, W: YType) -> float:
        model = _GaussianProcessModel(self.parameters, kernel)
        return model.compute_loss(X, F, W)

    def _fit(self, X: XType, F: YType, W: YType):
        best_kernel = self._search_method(X, F, W)

        self.best_model = _GaussianProcessModel(self.parameters, best_kernel)
        self.best_model._fit(X, F, W)
        return self


class GaussianProcessHeuristic(GaussianProcessGreedySearch):
    ''' Expands the best promissing node (based on the loss) up to the limit
        The limit is a) time b) number of expansion of kernels
    '''

    def _search_method(self, X: XType, F: YType, W: YType):
        # implements greedy search
        self._init_time()

        @dataclass(order=True)
        class Node:
            kernel: Any = field(compare=False)
            value: float

        expanded_nodes = []
        priority_queue = [
            Node(None, self._evaluate_kernel(Constant, X, F, W))
        ]

        while not self._check_timeup():
            priority_queue.sort()
            act_node = priority_queue.pop(0)
            expanded_nodes.append(act_node)

            for new_kernel in self._shuffle_expand_node(act_node.kernel):
                new_node = Node(new_kernel, self._evaluate_kernel(new_kernel, X, F, W))
                priority_queue.append(new_node)

                if self._check_timeup(add_model_count=1):
                    break

        all_nodes = expanded_nodes + priority_queue
        all_nodes.sort()
        best_state = all_nodes[0].kernel
        return best_state if best_state is not None else Constant

