import numpy
import getopt
import sys
from scipy.optimize import minimize
from math import log
import random
import gzip
import torch
import metropolis_kernel
from pyro.infer.mcmc import MCMC
import truncnorm.torch_truncnorm as torch_truncnorm
import likelihood
import transitions
import read_from_files


# variable explanations:
# m: methylated
# h: hydroxy-methylated
# u: not methylated

# mu_1: maintenance methylation parameter
# mu_2: de-novo methylation parameter
# mu_3: hydroxylation parameter
# mu_4/p: probability that hydroxyl group prevents maintenance

# fully methylated: cc, hemi-methylated (+ strand): tc, hemi-methylated (- strand): ct, unmethylated: tt

# constraint function for initial distribution optimization
def con(x):
    return numpy.sum(x) - 1


def print_results(result, time_points, chrom, pos, pi_0, acc_rate, st_dev):
    """
    recalculates part of the results and creates string for writing to file afterwards
    :param result: final efficiency parameters
    :param time_points: array of time points, except 0
    :param chrom: current chromosome
    :param pos: current position
    :param pi_0: initial distribution of methylation (estimated with MLE)
    :param acc_rate: acceptance ratio (from bayesian inference)
    :param st_dev: standard deviation of efficiency parameters
    :return: returns line to be printed to file
    """
    output_line = chrom + '\t' + str(pos) + '\t' + str(acc_rate) + '\t'
    for entry in pi_0[0]:
        output_line += str(entry.item()) + '\t'

    pi_t = pi_0
    p = result[6]
    for i in range(1, time_points[-1] + 1):
        mu_1 = result[0] + result[1] * i
        mu_2 = result[2] + result[3] * i
        mu_3 = result[4] + result[5] * i
        transition = transitions.transition_matrix(mu_1, mu_2, mu_3, p)
        pi_t = pi_t @ transition
        if i in time_points:
            for entry in pi_t[0]:
                output_line += str(entry.item()) + '\t'
    for i in result:
        output_line += str(i.item()) + '\t'

    for var in st_dev:
        output_line += str(var.item()) + '\t'

    output_line += '\n'
    return output_line


def model(x, sigma, t_max):
    """
    generates new sample from proposal distribution
    :param x: previous candidate sample
    :param sigma: variance
    :param t_max: max. time points
    :return: new candidate sample
    """
    new_sample = torch.zeros(7)
    # sample intercepts and parameter p
    y = torch_truncnorm.TruncatedNormal(loc=x[0:7:2], scale=sigma[0:7:2], a=0.0, b=1.0).sample()
    for j, val in enumerate(y):
        new_sample[j * 2] = val

    a = torch.zeros(3)
    b = torch.ones(3)
    for i in range(0, 3):
        # bounds for gradient are based on intercept
        a[i] = -new_sample[i * 2] / t_max
        b[i] = (1 - new_sample[i * 2]) / t_max

    # draw gradient sample from truncated distribution
    y = torch_truncnorm.TruncatedNormal(loc=x[1:7:2], scale=sigma[1:7:2], a=a, b=b).sample()
    for j, val in enumerate(y):
        new_sample[j * 2 + 1] = val
    return new_sample


def get_sigmas():
    cov_defs = numpy.array([[0.3947, -0.0679, -0.1393, 0.0273, 0.1416, -0.0339, 0.0601],
                            [-0.0679, 0.0162, 0.0225, -0.0041, -0.0227, 0.0070, 0.0001],
                            [-0.1393, 0.0225, 0.0732, -0.0141, -0.0491, 0.0106, -0.0007],
                            [0.0273, -0.0041, -0.0141, 0.0029, 0.0091, -0.0019, 0.0035],
                            [0.1416, -0.0227, -0.0491, 0.0091, 0.0815, -0.0193, -0.0123],
                            [-0.0339, 0.0070, 0.0106, -0.0019, -0.0193, 0.0058, 0.0019],
                            [0.0601, 0.0001, -0.0007, 0.0035, -0.0123, 0.0019, 0.1657]])
    temp = numpy.diag(cov_defs)
    sigmas = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i, val in enumerate(temp):
        sigmas[i] = numpy.sqrt(val)
    sigmas = torch.tensor(sigmas)
    return sigmas


def pyro_metropolis(time_points, count_bs, count_ox, conversions_bs, conversions_ox, pi_0):
    """
    Method which performs bayesian inference
    :param time_points: array of time points excluding day 0
    :param count_bs: bs counts of all days except day 0
    :param count_ox: ox counts of all days except day 0
    :param conversions_bs: conversion rates for bs-seq of all days except day 0
    :param conversions_ox: conversion rates for ox-seq of all days except day 0
    :param pi_0: initial distribution of methylation states
    :return: efficiency parameters, acceptance rate and standard deviation of efficiency parameters
    """
    steps = 8500
    t_max = time_points[-1]
    exclude = 500
    current_sample = {"params": torch.Tensor([0.9, -0.1, 0.9, -0.1, 0.9, -0.1, 0.5])}

    current_var = get_sigmas()
    c = 15
    current_var = torch.div(current_var, c)
    metrop_kernel = metropolis_kernel.MH(model, likelihood.likelihood, current_var, t_max, count_bs, count_ox, conversions_bs,
                                         conversions_ox, pi_0, time_points, current_sample)
    mcmc = MCMC(metrop_kernel, num_samples=steps, warmup_steps=exclude)
    mcmc.run(current_sample)

    results = mcmc.get_samples()['params']
    mean_results = torch.mean(results, 0)
    st_dev = numpy.diag(numpy.cov(results, rowvar=False))
    acc_rate = metrop_kernel.acceptance_ratio
    return mean_results, acc_rate, st_dev


def estimate_cpgs(input_dir, conversion_dir, output_file, chr_list, pos_list):
    """
    main method, opens files, reads in correct lines and calls initial optimization, afterwards bayesian inference
    :param input_dir: directory for files containing counts
    :param conversion_dir: path + file which contains conversion errors
    :param output_file: opened file to write results line by line
    :param chr_list: list of chromosomes to perform modeling on
    :param pos_list: list of positions, in same order than chr_list, this will be the starting point
    """
    time_points = [3, 6]
    # open files from all time points
    bs_init_file = gzip.open(input_dir + "/GSM5176043_WT-Serum-BS.pileup.CG.dsi.txt.gz", 'r')
    ox_init_file = gzip.open(input_dir + "/GSM5176044_WT-Serum-oxBS.pileup.CG.dsi.txt.gz", 'r')

    bs_g3 = gzip.open(input_dir + "/GSM5176045_WT-72h-2i-BS.pileup.CG.dsi.txt.gz", 'r')
    ox_g3 = gzip.open(input_dir + "/GSM5176046_WT-72h-2i-oxBS.pileup.CG.dsi.txt.gz", 'r')

    bs_g6 = gzip.open(input_dir + "/GSM5176047_WT-144h-2i-BS.pileup.CG.dsi.txt.gz", 'r')
    ox_g6 = gzip.open(input_dir + "/GSM5176048_WT-144h-2i-oxBS.pileup.CG.dsi.txt.gz", 'r')

    # save files (except init) state for later
    input_files = [bs_g3, ox_g3, bs_g6, ox_g6]

    # save line and file index from previous loop
    skip_index = []
    skip_counts = []
    skip_pos = -1
    chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                   '19', 'X', 'Y']

    sorted_chr_list = sorted(chr_list)
    chr_max = sorted_chr_list[-1]
    chrom = '1'
    while True:
        prev_chrom = chrom
        while True:
            chrom, pos, bs_counts_init = read_from_files.get_next_counts(bs_init_file)
            # coverage cutoff at 5
            if sum(bs_counts_init) > 5:
                break

        if (chrom == -1) or (chrom > chr_max):
            bs_init_file.close()
            ox_init_file.close()
            read_from_files.close_files(input_files)
            return

        # only read lines from given chromosomes
        if chrom not in chr_list:
            continue

        # only lines with chromosome after listed position
        if chrom in chr_list:
            i = chr_list.index(chrom)
            start_pos = int(pos_list[i])
            if pos < start_pos:
                continue

        if chrom != prev_chrom:
            print("new chromosome: " + str(chrom))

        while True:
            # get next line from ox-init file
            chrom1, pos1, ox_counts_init = read_from_files.get_next_counts(ox_init_file)
            if sum(ox_counts_init) > 5:
                break

        if (chrom1 == -1) or (chrom1 > chr_max):
            bs_init_file.close()
            ox_init_file.close()
            read_from_files.close_files(input_files)
            return

        # read lines until positions from both files (bs-init and ox-init) match
        while (chrom != chrom1) or (pos != pos1):
            # current bs position is greater than ox -> ox positions not present in bs file, read lines until bs-pos
            # is found (potentially)
            if (chrom > chrom1) or ((pos > pos1) and (chrom == chrom1)):
                while True:
                    chrom1, pos1, ox_counts_init = read_from_files.get_next_counts(ox_init_file)
                    if sum(ox_counts_init) > 5:
                        break
                if (chrom1 == -1) or (chrom1 > chromosomes[-1]):
                    bs_init_file.close()
                    ox_init_file.close()
                    read_from_files.close_files(input_files)
                    return
            # current ox position is greater than bs -> read more bs lines to match position
            elif (chrom < chrom1) or ((pos < pos1) and (chrom == chrom1)):
                while True:
                    chrom, pos, bs_counts_init = read_from_files.get_next_counts(bs_init_file)
                    if sum(bs_counts_init) > 5:
                        break
                if (chrom == -1) or (chrom > chromosomes[-1]):
                    bs_init_file.close()
                    ox_init_file.close()
                    read_from_files.close_files(input_files)
                    return

        # positions are equal now, we can continue with initial estimation
        bs_conversion_rates, ox_conversion_rates = read_from_files.get_conversion_errors(conversion_dir)
        cons = ({'type': 'eq', 'fun': con})
        # initial guess for methylation states
        init = numpy.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # run up to 10 times if minimization does not return successful results
        for i in range(0, 10):
            # MLE: minimizing log-likelihood
            results_init = minimize(likelihood.init_likelihood, init,
                                    args=(
                                        bs_conversion_rates[0], ox_conversion_rates[0], bs_counts_init, ox_counts_init),
                                    constraints=cons,
                                    bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                                            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                    method='SLSQP', tol=1e-06)
            if results_init.success:
                break
            else:
                # generate random init (within the constraints) for new estimation, might lead to a better outcome
                init = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                temp_sum = 1.0
                for j in range(0, len(init)):
                    if temp_sum < 0.1:
                        init[j] = temp_sum
                        break
                    new_x = random.uniform(0.0, min(temp_sum, 0.5))
                    init[j] = new_x
                    temp_sum -= new_x
        if results_init.success:
            pi_0 = results_init.x
            # read in counts for later time points
            if pos == skip_pos:
                # pos has been read in previous loop, remember this
                bs_counts, ox_counts, skip_index_temp, skip_pos_temp = read_from_files.read_counts(input_files, chrom, pos, skip_index,
                                                                                   skip_counts)
            # min. one file already ahead of current pos -> current pos is not present in all files, skip to next
            # position
            elif pos < skip_pos:
                continue
            else:
                # read next counts in normally
                bs_counts, ox_counts, skip_index_temp, skip_pos_temp = read_from_files.read_counts(input_files, chrom, pos, [], [])

            # method read_counts returns new skip_index (new positions to remember for next round)
            if (skip_pos_temp == skip_pos) and (skip_index_temp != -1):
                skip_index.append(skip_index_temp)

            # new skip position, older skip_pos can be deleted
            if skip_pos_temp > skip_pos:
                skip_pos = skip_pos_temp
                skip_index = [skip_index_temp]
                skip_counts = []

            if bs_counts is None:
                # end of file
                bs_init_file.close()
                ox_init_file.close()
                return

            if (len(bs_counts) == 0) and (len(ox_counts) == 0):
                continue
            elif len(bs_counts) == 0:
                skip_counts.append(ox_counts)
                continue
            elif len(ox_counts) == 0:
                skip_counts.append(bs_counts)
                continue

            pi_0 = torch.Tensor([pi_0])
            # run bayesian inference here
            result_bi, acc_rate, st_dev = pyro_metropolis(time_points, bs_counts, ox_counts, bs_conversion_rates,
                                                          ox_conversion_rates, pi_0)

            # method to create output string to write to file
            output_line = print_results(result_bi, time_points, chrom, pos, pi_0, acc_rate, st_dev)
            output_file.write(output_line)

        else:
            print("no minimum found for init results")

    read_from_files.close_files(input_files)
    bs_init_file.close()
    ox_init_file.close()


def main(argv):
    # code for get-opt for program parameters
    conversion_dir = ''
    input_dir = ''
    output_dir = ''
    chromosomes = ''
    positions = ''
    try:
        opts, args = getopt.getopt(argv, "hc:i:o:p:e:", ["chrs=", "idir=", "odir=", "plist=", "edir="])
    except getopt.GetoptError:
        print('bayesian_inference.py -e <conversiondir> -i <inputdir> -o <outputdir> -c <chromosomes> -p <position>')
        print("Program arguments:\n-e: directory containing files with conversion errors for each time point\n"
              "-i: directory containing input data for each time point\n"
              "-o: output directory\n"
              "-c: list of chromosomes, comma separated\n"
              "-p: list of positions of chromosomes to start with\n"
              "remember to change the file names listed in the code to match your files")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('bayesian_inference.py -e <conversiondir> -i <inputdir> -o <outputdir> -c <chromosomes> -p <position>')
            print("Program arguments:\n-e: directory containing files with conversion rates for each time point\n"
                  "-i: directory containing input data for each time point\n"
                  "-o: output directory\n"
                  "-c: list of chromosomes, comma separated\n"
                  "-p: list of positions of chromosomes to start with\n"
                  "remember to change the file names listed in the code to match your files")
            sys.exit()
        elif opt in ("-c", "--chrs"):
            chromosomes = arg
        elif opt in ("-i", "--idir"):
            input_dir = arg
        elif opt in ("-o", "--odir"):
            output_dir = arg
        elif opt in ("-p", "--plist"):
            positions = arg
        elif opt in ("-e", "--edir"):
            conversion_dir = arg

    chr_list = chromosomes.strip('\n ').split(',')
    pos_list = positions.strip('\n ').split(',')

    all_chr = False 
    if (len(chr_list) == 1) and (chr_list[0] == ''): 
        # all chromosomes 
        chr_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                   '19', 'X', 'Y']
        file_name = output_dir + "/BI_results_all_chrs.txt"
        all_chr = True

    if (len(pos_list) == 1) and (pos_list[0] == ''): 
        # all starting positions
        pos_list = []
        for i in range(0, len(chr_list)): 
            pos_list.append('1')

    if not all_chr: 
        file_name = output_dir + "/BI_results_chr" + str(chr_list[0]) + "-" + str(pos_list[0]) + ".txt"

    output_file = open(file_name, 'w')
    # header
    output_file.write("Chromosome\tPosition\tAcceptance_Rate\tmm_d0\tmu_d0\tum_d0\tuu_d0\thh_d0\t"
                      "hu_d0\tuh_d0\thm_d0\tmh_d0\tmm_d3\tmu_d3\tum_d3\tuu_d3\thh_d3\thu_d3\tuh_d3\thm_d3\tmh_d3\t"
                      "mm_d6\tmu_d6\tum_d6\tuu_d6\thh_d6\thu_d6\tuh_d6\thm_d6\tmh_d6\tb_0_maint\tb_1_maint\tb_0_denovo"
                      "\tb_1_denovo\tb_0_hydroxy\t_1_hydroxy\tpRecogn\tb_0_maint_sd\tb_1_maint_sd\tb_0_denovo_sd\t"
                      "b_1_denovo_sd\tb_0_hydroxy_sd\tb_1_hydroxy_sd\tpRecogn_sd\n")

    estimate_cpgs(input_dir, conversion_dir, output_file, chr_list, pos_list)
    output_file.close()


main(sys.argv[1:])

