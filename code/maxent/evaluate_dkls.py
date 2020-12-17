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

def make_energy(params):
    if params.files == ['f']:
        raise NotImplementedError('independent model dkl not implemented')
    elif params.files == ['h', 'J']:
        model = 'ncov'
        h = params['h']
        J = params['J']

        @njit
        def energy(x):
            return energy_ncov(x, h, J)
    elif params.files == ['h', 'J', 'J2']:
        model = 'nskew'
        h = params['h']
        J = params['J']
        J2 = params['J2']

        @njit
        def energy(x):
            return energy_nskew(x, h, J, J2)
    elif params.files == ['h', 'J', 'J2', 'hi', 'Jij']:
        model = 'nskewfcov'
        h = params['h']
        hi = params['hi']
        J = params['J']
        J2 = params['J2']
        Jij = params['Jij']

        @njit
        def energy(x):
            return energy_nskewfcov(x, h, J, J2, hi, Jij)
    return energy

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
