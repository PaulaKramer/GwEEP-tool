# GwEEP-tool

This is a *G*enome-*w*ide *E*pigenetic *E*fficiency *P*rofiling tool which estimates methylation efficiencies over time given bisulfite and oxidative bisulfite sequencing data. The tool can predict the efficiencies of maintenance, de-novo and hydroxyl-methylation of single CpGs and infer the methylation levels. The efficiencies are modeled using a Hidden Markov model and the estimation of these parameters is done using Bayesian Inference. 


## Usage
The tool is implemented in Python and can be called in the following way:
```bash
python bayesian_inference.py -i /path/to/input/directory -o /path/to/output/directory -e conversion_file -c [list of chromosomes] -p [list of positions]

```

#### Program Arguments 
- `i`: full path to input directory, where the files containing the bisulfite and oxidative bisulfite counts are for all the time points. The input files should not contain a header. 
- `o`: full path to output directory
- `e`: full path including the exact file name. The file contains the conversion rates during bisulfite and oxidative bisulfite sequencing for the different time points.
- `c`: list of chromosomes (comma separated). If this argument is missing, all chromosomes will be considered. 
- `p`: list of starting positions (comma separated). The number of positions should be the same than the number of chromosomes. Then the first position corresponds to the first listed chromosome. If this argument is missing, the whole chromosome will be considered.


The program can run in parallel, as long as the chromosomes with their corresponding position are not identical. Exemplary files containing the layout of CpG counts and the conversion errors are in folder `example_files`. At the moment, the file names containing the counts still need to be added by hand in the script. The resulting file will contain the methylation levels for each time point, the methylation efficiencies and the standard deviations of the efficiencies. 
