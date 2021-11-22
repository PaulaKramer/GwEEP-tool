def get_conversion_errors(path):
    """
    get all the conversion errors from given file for all time points
    :param path: path contains complete path + file name
    :return: conversion errors
    """
    ox_rates = []
    bs_rates = []

    with open(path, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            # skip header
            if i == 0:
                continue
            conversion_error = [0.0, 0.0, 0.0]
            line.strip()
            line = line.split('\t')
            conversion_error[0] = float(line[4])
            conversion_error[1] = 1 - float(line[5])
            if line[3] == "bisulfite":
                conversion_error[2] = 1 - float(line[6])
                bs_rates.append(conversion_error)
            else:
                conversion_error[2] = float(line[6])
                ox_rates.append(conversion_error)
    return bs_rates, ox_rates

    
# closes all files in array input_files
def close_files(input_files):
    for file in input_files:
        file.close()


def get_next_counts(file):
    """
    read next line in file, extract counts from line
    :param file: given file, already opened
    :return: chromosome, position and counts
    """
    new_line = file.readline()
    line = new_line.decode()
    line = line.strip('\n').split('\t')
    if len(line) == 11:
        chrom = line[0]

        pos = int(line[1])
        counts = [int(line[10]), int(line[9]), int(line[8]), int(line[7])]
        return chrom, pos, counts
    else:
        # end of file
        return -1, -1, []


def read_counts(input_files, chrom, pos, skip_index, skip_counts):
    """
    Reads in next lines from file to (potentially) find counts for given position
    :param input_files: array containing open files for all time points except initial states
    :param chrom: current chromosome
    :param pos: current position
    :param skip_index: open file already (potentially) contains current position (from previous loop), index of file
    :param skip_counts: counts from previous loop from file [skip_index]
    :return: returns counts for bs and oxbs, new skip_index and counts
    """
    data_bs = []
    data_ox = []
    for i in range(0, len(input_files)):
        # count saved from from previous loop -> add to correct count array
        if i in skip_index:
            index = skip_index.index(i)
            if i % 2 == 0:
                data_bs.append(skip_counts[index])
                continue
            else:
                data_ox.append(skip_counts[index])
                continue
        while True:
            chrom1, pos1, counts = get_next_counts(input_files[i])
            if sum(counts) > 5:
                break
        if chrom1 == -1:
            print("end of file")
            close_files(input_files)
            return None, None, -1, -1
        # position in file is still smaller than needed, read next lines to find matching pos
        while (chrom > chrom1) or ((pos > pos1) and (chrom == chrom1)):
            while True:
                chrom1, pos1, counts = get_next_counts(input_files[i])
                if sum(counts) > 5:
                    break
            if chrom1 == -1:
                print("end of file")
                close_files(input_files)
                return None, None, -1, -1
        # current position is greater than needed -> needed position is not in file, it would have been found by now
        if (chrom < chrom1) or ((pos < pos1) and (chrom == chrom1)):
            # because position is greater, the line could still be needed for upcoming positions, we should not discard
            # it -> save this position, file index (skip-index) and counts for next loop
            if i % 2:
                return counts, [], i, pos1
            else:
                return [], counts, i, pos1
        if pos != pos1:
            print("still not the same cpg")
            print(chrom + '\t' + str(pos) + '\t' + chrom1 + '\t' + str(pos1))
        if i % 2 == 0:
            data_bs.append(counts)
        else:
            data_ox.append(counts)
    return data_bs, data_ox, -1, -1

