import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *


k = int(snakemake.wildcards.k)
proteome = snakemake.wildcards.proteome
model = snakemake.wildcards.model
reference = snakemake.params.reference
q = naminoacids

entropy = pd.read_csv('data/{proteome}_{model}_k{k}_entropy.csv'.format(proteome=proteome, model=model, k=k),
                      header=None, index_col=0)
entropy_reference = pd.read_csv('data/{proteome}_{model}_k{k}_entropy.csv'.format(proteome=reference, model=model, k=k),
                      header=None, index_col=0)

params = np.load('data/{proteome}_{model}_k{k}_params.npz'.format(proteome=proteome, model=model, k=k))
params_reference = np.load('data/{proteome}_{model}_k{k}_params.npz'.format(proteome=reference, model=model, k=k))


matrix = load_matrix('data/{proteome}_{model}_k{k}_matrix.csv.gz'.format(proteome=proteome, model=model, k=k))

if model == 'independent':
    DKL = k*scipy.stats.entropy(params['f'], qk=params_reference['f'])
else:
    energy = make_energy(params)
    energy_reference = make_energy(params_reference)
    energies = np.array([energy(x) for x in matrix])
    energies_reference = np.array([energy_reference(x) for x in matrix])
    DKL = float(entropy.loc['F']) - np.mean(energies) + np.mean(energies_reference) - float(entropy_reference.loc['F'])

series = pd.Series(data=[reference, DKL], index=['reference', 'DKL'])
series.to_csv(snakemake.output[0])
