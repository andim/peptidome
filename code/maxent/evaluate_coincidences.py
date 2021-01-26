import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from numba import jit

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

proteome = snakemake.wildcards.proteome
models = ['independent', 'ncov', 'nskew', 'nskewfcov', 'train']
k = snakemake.wildcards.k

coincidence_probs = {}
for model in models:
    if not model == 'train':
        energy = make_energy(np.load('../maxent/data/{proteome}_{model}_k{k}_params.npz'.format(
            proteome=proteome, model=model, k=k)))
        F = np.float(pd.read_csv('../maxent/data/{proteome}_{model}_k{k}_entropy.csv'.format(
            proteome=proteome, model=model, k=k),
                                header=None, index_col=0).loc['F'])
        loglikelihood  = lambda seq: -energy(seq) + F
    matrix = load_matrix('data/{proteome}_{model}_k{k}_matrix.csv.gz'.format(
                         proteome=proteome, model=model, k=k))
    if not model == 'train':
        logp = np.array([loglikelihood(row) for row in matrix])
        coincidence_prob = np.mean(np.exp(logp))
    else:
        coincidence_prob = calc_coincidence_prob(matrix)
    coincidence_probs[model] = coincidence_prob

series = pd.Series(coincidence_probs, name='coincidence_prob')
series.to_csv(snakemake.output[0])
