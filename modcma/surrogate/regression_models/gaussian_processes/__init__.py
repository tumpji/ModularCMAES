# gaussian processes
from modcma.surrogate.regression_models.gaussian_processes.model_gp import GaussianProcess
from modcma.surrogate.regression_models.gaussian_processes.model_gp_basic_selection import \
    GaussianProcessBasicSelection, \
    GaussianProcessBasicAdditiveSelection, \
    GaussianProcessBasicMultiplicativeSelection, \
    GaussianProcessBasicBinarySelection, \
    GaussianProcessGreedySearch, \
    GaussianProcessHeuristic

# deep gaussian processes
from modcma.surrogate.regression_models.gaussian_processes.deep_gaussian_processes import DeepGaussianProcessStochImp

