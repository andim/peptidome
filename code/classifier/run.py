import glob, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

pathogen = 'Malaria'
k = 9

with open('../../data/triplet-human.json', 'r') as f:
    tripletparams_human = json.load(f)
    loglikelihood_human = lambda seq: loglikelihood_independent(seq, **tripletparams_human)
with open('../../data/triplet-%s.json'%pathogen.lower(), 'r') as f:
    tripletparams_malaria = json.load(f)
    loglikelihood_pathogen = lambda seq: loglikelihood_independent(seq, **tripletparams_malaria)

logp_hh = np.array([loglikelihood_human(seq[i:i+k]) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1)
                if isvalidaa(seq[i:i+k])])

logp_hp = np.array([loglikelihood_pathogen(seq[i:i+k]) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1)
               if isvalidaa(seq[i:i+k])])

logp_pp = np.array([loglikelihood_pathogen(seq[i:i+k]) for h, seq in fasta_iter(proteome_path(pathogen)) for i in range(len(seq)-k+1)
               if isvalidaa(seq[i:i+k])])

logp_ph = np.array([loglikelihood_human(seq[i:i+k]) for h, seq in fasta_iter(proteome_path(pathogen)) for i in range(len(seq)-k+1)
               if isvalidaa(seq[i:i+k])])

np.savez('data/%s.npz'%pathogen, logp_hh=logp_hh, logp_hp=logp_hp, logp_pp=logp_pp, logp_ph=logp_ph)
