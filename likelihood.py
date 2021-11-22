import torch 
import numpy
import transitions
from math import log

# calculate the negative log-likelihood with given mu constants, pi_0 distribution, input file distribution and a
# fixed time step variable
def likelihood(parameters, bs_counts, ox_counts, bs_conversions, ox_conversions, pi_0, max_t, time_points):
    """
    calculates the log-likelihood for current parameters
    :param parameters: current efficiencies
    :param bs_counts: bisulfite counts
    :param ox_counts: oxidative bisulfite counts
    :param bs_conversions: conversion rates for bisulfite sequencing
    :param ox_conversions: conversion rates for ox-bs seq
    :param pi_0: initial distribution of methylation states
    :param max_t: max. time point
    :param time_points: array of time points, excluding initial day 0
    :return: log-likelihood
    """
    log_likelihood = 0.0
    pi_t = pi_0
    mu_1 = parameters[0]  # maintenance methylation parameter (intercept)
    mu_2 = parameters[2]  # de-novo methylation parameter (intercept)
    mu_3 = parameters[4]  # hydroxyl-methylation parameter (intercept)
    mu_4 = parameters[6]  # parameter p (maintenance semi-hydroxylation)

    mu_12 = parameters[1]  # slope for maintenance
    mu_22 = parameters[3]  # slope for de-novo
    mu_32 = parameters[5]  # slope for hydroxylation

    for i in range(1, max_t + 1):
        # current transition parameters (dependent on time point)
        mu_1 = mu_1 + mu_12  # maintenance methylation parameter
        mu_2 = mu_2 + mu_22  # de-novo methylation parameter
        mu_3 = mu_3 + mu_32  # hydroxyl-methylation parameter
        p = transitions.transition_matrix(mu_1, mu_2, mu_3, mu_4)
        pi_t = pi_t @ p

        if i in time_points:
            index = time_points.index(i)
            emissions_bs = transitions.emission_matrix(bs_conversions[index], True)
            emissions_ox = transitions.emission_matrix(ox_conversions[index], False)

            pi_bs = torch.log(pi_t @ emissions_bs)[0]
            pi_ox = torch.log(pi_t @ emissions_ox)[0]
            current_bs_counts = torch.tensor(bs_counts[index])
            current_ox_counts = torch.tensor(ox_counts[index])

            log_array = torch.mul(current_bs_counts, pi_bs) + torch.mul(current_ox_counts, pi_ox)
            log_likelihood += torch.sum(log_array)
    return log_likelihood


def init_likelihood(parameters, bs_conversion_rate, ox_conversion_rate, counts_bs, counts_ox):
    """
    calculates log-likelihood for parameters of initial estimation of methylation states
    :param parameters: current parameters (methylation distribution)
    :param bs_conversion_rate: conversion rates for bs-seq for initial time point
    :param ox_conversion_rate: conversion rates for ox-bs seq for initial time point
    :param counts_bs: initial bs counts
    :param counts_ox: initial ox counts
    :return: negative log-likelihood
    """
    conversions_bisulfite = transitions.emission_matrix(bs_conversion_rate, True)
    conversions_oxbisulfite = transitions.emission_matrix(ox_conversion_rate, False)

    pi_bs = numpy.matmul(parameters, conversions_bisulfite)
    pi_ox = numpy.matmul(parameters, conversions_oxbisulfite)

    log_likelihood = 0.0
    for i, x in enumerate(counts_bs):
        log_likelihood += x * log(pi_bs[i]) + counts_ox[i] * log(pi_ox[i])
    return -log_likelihood

