from __future__ import absolute_import, division, print_function

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from numpy import isfinite
import torch
import truncnorm.torch_truncnorm as torch_truncnorm


def calculate_prior(params, a, b):
    """
    calculates the prior from beta and normal distribution
    :param params: current parameters (methylation efficiencies)
    :param a: parameter for beta distribution
    :param b: parameter for beta distribution
    :return: multiplied distribution pdf
    """
    beta_pdfs = pyro.distributions.Beta(a, b).log_prob(params[0:2:7])
    prior_pdfs = pyro.distributions.Normal(loc=0.0, scale=1 / 3).log_prob(params[1:2:6])
    prior = torch.sum(beta_pdfs) + torch.sum(prior_pdfs)
    return prior


def truncated_log_prob(x, y, sigma, t_max):
    """
    calculates the log pdf of proposal distribution
    :param x: candidate
    :param y: second candidate
    :param sigma: variance
    :param t_max: max time point
    :return: log pdf of proposal
    """
    a = torch.zeros(7)
    b = torch.ones(7)

    for i in range(0, 7):
        if i % 2 == 1:
            # bounds for gradient are based on intercept
            a[i] = -x[i - 1] / t_max
            b[i] = (1 - x[i - 1]) / t_max

    # draw sample from truncated distribution
    new_dist = torch_truncnorm.TruncatedNormal(loc=y, scale=sigma, a=a, b=b).log_prob(x)
    g_x_y = torch.sum(new_dist)
    return g_x_y


class MH(MCMCKernel):
    def __init__(self, model, proposal_dist, sigma, t_max, count_bs, count_ox, conversions_bs, conversions_ox,
                 pi_0, time_points, init):
        self.proposal_dist = proposal_dist

        # store all of our traces, and keep count on acceptance ratio
        self._accept_cnt = 0
        self._tune_cnt = 0
        self._call_cnt = 0

        # added parameters
        self._sigma = sigma
        self._t_max = t_max
        self._count_bs = count_bs
        self._count_ox = count_ox
        self._conversions_bs = conversions_bs
        self._conversions_ox = conversions_ox
        self._pi_0 = pi_0
        self._time_points = time_points
        self._initial_params = init
        self._model = model
        self._transforms = {}
        super(MH, self).__init__()

    @property
    def initial_params(self):
        return self._initial_params

    @property
    def model(self):
        return self._model

    @property
    def transforms(self):
        return self._transforms

    @property
    def num_accepts(self):
        return self._accept_cnt

    def sample(self, params):
        """
        One step in the Metropolis Hastings algorithm: sample new candidate, calculate proposal, accept candidate with
        prob. alpha
        :param params: current parameters
        :return: new parameters (either old ones or the newly sampled one)
        """
        # sample called
        self._call_cnt += 1

        current_sample = params["params"]
        # get new proposal candidate
        proposal_value = self._model(current_sample, sigma=self._sigma, t_max=self._t_max)
        # get logp for both parameters (prior + likelihood)
        logp_original = calculate_prior(current_sample, 2, 2) + self.proposal_dist(current_sample, self._count_bs,
                                                                                   self._count_ox, self._conversions_bs,
                                                                                   self._conversions_ox, self._pi_0,
                                                                                   self._t_max, self._time_points)
        logp_proposal = calculate_prior(proposal_value, 2, 2) + self.proposal_dist(proposal_value, self._count_bs,
                                                                                   self._count_ox, self._conversions_bs,
                                                                                   self._conversions_ox, self._pi_0,
                                                                                   self._t_max, self._time_points)

        # acceptance correction: truncated log-prob values calculated here
        R = truncated_log_prob(current_sample, proposal_value, self._sigma, self._t_max)
        F = truncated_log_prob(proposal_value, current_sample, self._sigma, self._t_max)
        # alpha is the acceptance probability for the new candidate (proposal_value)
        alpha = (logp_proposal - logp_original) + (R - F)
        rand = dist.Uniform(low=0.0, high=1.0).sample()
        if isfinite(alpha) and rand.log() < alpha:
            # accept!
            self._accept_cnt += 1

            # accept the proposal, clean up the old params
            params = {"params": proposal_value}
        # else:
            # keep the same params, reject the proposed
        return params

    @property
    def acceptance_ratio(self):
        return self._accept_cnt / self._call_cnt
