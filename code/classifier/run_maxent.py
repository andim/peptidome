import glob, json, random
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

#proteome = 'Humanviruses'
#model = 'nskewfcov'
#k = 9
proteome = snakemake.wildcards.proteome
model = snakemake.wildcards.model
k = snakemake.wildcards.k

energy = make_energy(np.load('../maxent/data/{proteome}_{model}_k{k}_params.npz'.format(proteome=proteome, model=model, k=k)))
energy_human = make_energy(np.load('../maxent/data/{proteome}_{model}_k{k}_params.npz'.format(proteome='Human', model=model, k=k)))
F = np.float(pd.read_csv('../maxent/data/{proteome}_{model}_k{k}_entropy.csv'.format(proteome=proteome, model=model, k=k),
                        header=None, index_col=0).loc['F'])
F_human = np.float(pd.read_csv('../maxent/data/{proteome}_{model}_k{k}_entropy.csv'.format(proteome='Human', model=model, k=k),
                        header=None, index_col=0).loc['F'])
loglikelihood_pathogen  = lambda seq: -energy(seq) + F
loglikelihood_human  = lambda seq: -energy_human(seq) + F_human


human_kmers = load_matrix('../maxent/data/Human_test_k{k}_matrix.csv.gz'.format(k=k))
pathogen_kmers = load_matrix('../maxent/data/{proteome}_test_k{k}_matrix.csv.gz'.format(proteome=proteome, k=k))

print('data loaded')

logp_hh = np.array([loglikelihood_human(kmer) for kmer in human_kmers])
print('.')
logp_hp = np.array([loglikelihood_pathogen(kmer) for kmer in human_kmers])
print('.')
logp_pp = np.array([loglikelihood_pathogen(kmer) for kmer in pathogen_kmers])
print('.')
logp_ph = np.array([loglikelihood_human(kmer) for kmer in pathogen_kmers])

np.savez(snakemake.output[0],
        logp_hh=logp_hh, logp_hp=logp_hp, logp_pp=logp_pp, logp_ph=logp_ph)
