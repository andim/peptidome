import glob, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

proteome = 'Humanviruses'
model = 'nskewfcov'
k = 9

energy = make_energy(np.load('../maxent/data/{proteome}_{model}_k{k}_params.npz'.format(proteome=proteome, model=model, k=k)))
energy_human = make_energy(np.load('../maxent/data/{proteome}_{model}_k{k}_params.npz'.format(proteome='Human', model=model, k=k)))
F = np.float(pd.read_csv('../maxent/data/{proteome}_{model}_k{k}_entropy.csv'.format(proteome=proteome, model=model, k=k),
                        header=None, index_col=0).loc['F'])
F_human = np.float(pd.read_csv('../maxent/data/{proteome}_{model}_k{k}_entropy.csv'.format(proteome='Human', model=model, k=k),
                        header=None, index_col=0).loc['F'])
loglikelihood_pathogen  = lambda seq: -energy(seq) + F
loglikelihood_human  = lambda seq: -energy_human(seq) + F_human

logp_hh = np.array([loglikelihood_human(map_aatonumber(seq[i:i+k])) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1)
                if isvalidaa(seq[i:i+k])])

logp_hp = np.array([loglikelihood_pathogen(map_aatonumber(seq[i:i+k])) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1)
               if isvalidaa(seq[i:i+k])])

logp_pp = np.array([loglikelihood_pathogen(map_aatonumber(seq[i:i+k])) for h, seq in fasta_iter(proteome_path(proteome)) for i in range(len(seq)-k+1)
               if isvalidaa(seq[i:i+k])])

logp_ph = np.array([loglikelihood_human(map_aatonumber(seq[i:i+k])) for h, seq in fasta_iter(proteome_path(proteome)) for i in range(len(seq)-k+1)
               if isvalidaa(seq[i:i+k])])

np.savez('data/{proteome}_{model}_k{k}_likelihoods.npz'.format(proteome=proteome, model=model, k=k),
        logp_hh=logp_hh, logp_hp=logp_hp, logp_pp=logp_pp, logp_ph=logp_ph)
