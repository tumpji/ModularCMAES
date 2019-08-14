from typing import Optional
import numpy as np


class BaseParameters(dict):
    alpha_mu = 2

    ## Threshold Convergence ##
    init_threshold = .2
    decay_factor = .995

    ## TPA ##
    alpha = 0.5
    tpa_factor = 0.5
    beta_tpa = 0
    c_alpha = 0.3

    ## Break conditions ##
    max_generations = int(1e10)
    rtol = 1e-8

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        self[attr] = value


class Parameters(BaseParameters):
    # TODO: think of more semantic variable names
    bool_default_opts = dict.fromkeys(
        ['active', 'elitist', 'mirrored',
         'orthogonal', 'sequential', 'threshold_convergence', 'tpa'], False)

    string_default_opts = dict.fromkeys(
        ['base-sampler', 'selection', 'weights_option'])

    def __init__(
        self,
        d: int,
        mu: Optional[int] = None,
        lambda_: Optional[int] = None,
        sigma: Optional[float] = None,
        budget: Optional[int] = None,
        target: Optional[float] = None,
        seq_cutoff_factor: Optional[int] = 1
    ) -> None:
        # TODO: check mu/lambda interdependencies
        self.d = d
        self.target = target

        self.lambda_ = lambda_ or int(4 + np.floor(3 * np.log(self.d)))
        self.mu = mu or int(1 + np.floor((self.lambda_ - 1) * .5))

        self.seq_cutoff_factor = seq_cutoff_factor
        self.seq_cutoff = self.mu * self.seq_cutoff_factor
        self.budget = budget or 1e4 * self.d
        self.used_budget = 0

        self.set_default_parameters()
        self.set_static_variables()
        self.init_dynamic_vars(sigma)

    def set_default_parameters(self):
        # TODO: make this configurable
        self.update(
            {**self.bool_default_opts,
             **self.string_default_opts,
             **self}
        )
        self.init_bounds()

    def init_bounds(self):
        self.lb = -5
        self.ub = 5
        self.diameter = np.sqrt(
            np.sum(
                np.square((
                    (np.ones((self.d, 1)) * self.ub) -
                    (np.ones((self.d, 1)) * self.lb)
                ))
            )
        )
        self.wcm = (np.random.randn(self.d, 1) *
                    (self.ub - self.lb)) + self.lb

    def set_static_variables(self):
        # TODO: Check which varaibles are static
        self.mu_eff = 1 / np.sum(np.square(self.recombination_weights))

        self.c_sigma = (self.mu_eff + 2) / (self.mu_eff + self.d + 5)
        self.c_c = (4 + self.mu_eff / self.d) / \
            (self.d + 4 + 2 * self.mu_eff / self.d)
        self.c_1 = 2 / ((self.d + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, self.alpha_mu * (
            (self.mu_eff - 2 + 1 / self.mu_eff) / (
                (self.d + 2)**2 + self.alpha_mu * self.mu_eff / 2)))
        self.damps = 1 + 2 * \
            np.max([0, np.sqrt((self.mu_eff - 1) / (self.d + 1)) - 1]
                   ) + self.c_sigma
        self.chi_N = self.d**.5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d**2))

    def init_dynamic_vars(self, sigma=None):
        self.sigma = self.sigma_old = sigma or 1.
        self.C = np.eye(self.d)
        self.sqrt_C = np.eye(self.d)
        self.B = np.eye(self.d)
        self.D = np.ones((self.d, 1))
        self.p_sigma = np.zeros((self.d, 1))
        self.p_c = np.zeros((self.d, 1))

    def adapt(self, pop):
        self.p_sigma = (
            (1 - self.c_sigma) * self.p_sigma +
            np.sqrt(
                self.c_sigma * (2 - self.c_sigma) * self.mu_eff
            ) *
            np.dot(
                self.sqrt_C, (self.wcm - self.wcm_old) / self.sigma
            )
        )

        hsig = (
            (self.p_sigma ** 2).sum() /
            (
                (1 - (1 - self.c_sigma) **
                    (2 * self.used_budget / self.lambda_)
                 )
            ) / self.d) < 2 + 4 / (self.d + 1)

        # hsig_wiki = np.linalg.norm(self.p_sigma) / np.sqrt(
        #     1 - (1 - self.c_sigma) ** (
        #         2 * self.used_budget / self.lambda_
        #     )
        # ) / self.chi_N < (1.4 + 2 / (self.d + 1))

        self.p_c = (
            (1 - self.c_c) * self.p_c + hsig *
            np.sqrt(
                self.c_c * (2 - self.c_c) * self.mu_eff
            ) * (self.wcm - self.wcm_old) / self.sigma
        )

        offset = pop.mutation_vectors[:, :self.mu]

        self.C = (
            (1 - self.c_1 - self.c_mu) * self.C + self.c_1 *
            (np.outer(self.p_c, self.p_c) + (1 - hsig) * self.c_c *
             (2 - self.c_c) * self.C) + self.c_mu *
            np.dot(offset, self.recombination_weights * offset.T)
        )

        if self.active:
            self.active_update(pop)

        if self.tpa:
            self.tpa_update()
        else:
            self.sigma_update()

        self.sigma_old = self.sigma

        try:
            self.diagonalize()
        except np.linalg.LinAlgError as err:
            self.init_dynamic_vars()

    def diagonalize(self):
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        # are these values correct ?
        # The smaller you make right hand side -> the earlier you restart,
        # the quicker you converge

        if np.isinf(self.C).any() or (not 1e-16 < self.sigma_old < 1e6):
            raise np.linalg.LinAlgError(
                'The Covariance matrix has degenerated')
        # casting to complex to allow negative sqrt
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.sqrt(eigenvalues.astype(complex).reshape(-1, 1))

        if np.isinf(eigenvalues).any() or (~np.isreal(eigenvalues)).any():
            raise np.linalg.LinAlgError(
                'The eigenvalues of the Covariance matrix are infinite or not real')

        self.D = np.real(eigenvalues)
        self.B = eigenvectors
        self.sqrt_C = np.dot(
            eigenvectors, eigenvalues ** -1 * eigenvectors.T)

    def tpa_update(self):
        # tpa_result, alpha_s and beta_tpa are still undefined
        alpha = self.tpa_result * self.alpha + (
            self.beta_tpa * (self.tpa_result > 1)
        )
        self.alpha_s += self.c_alpha * (alpha - self.alpha_s)
        self.sigma *= np.exp(self.alpha_s)

    def sigma_update(self):
        self.sigma *= np.exp((
            (self.c_sigma / self.damps) *
            (np.linalg.norm(self.p_sigma) / self.chi_N - 1)
        ))
        if np.isinf(self.sigma):
            self.sigma = self.sigma_old

    def active_update(self, pop):
        if pop.n >= (2 * self.mu):
            offset = pop.mutation_vectors[:, -self.mu:]
            self.C -= self.c_mu * np.dot(offset, self.rw * offset.T)

    @property
    def recombination_weights(self):
        if self.weights_option == '1/n':
            return np.ones((self.mu, 1)) * (1 / self.mu)
        else:
            _mu_prime = (self.lambda_ - 1) / 2.0
            weights = np.log(_mu_prime + 1.0) - \
                np.log(np.arange(1, self.mu + 1)[:, np.newaxis])
            return weights / np.sum(weights)

    @property
    def threshold(self):
        return (
            self.init_threshold * self.diameter *
            ((self.budget - self.used_budget) / self.budget)
            ** self.decay_factor)
