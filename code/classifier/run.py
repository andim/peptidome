import glob, json, random
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

proteome = 'Humanviruses'
k = 9
#nkmers = int(1e5)

with open('../../data/triplet-human.json', 'r') as f:
    tripletparams_human = json.load(f)
    loglikelihood_human = lambda seq: loglikelihood_independent(seq, **tripletparams_human)
with open('../../data/triplet-%s.json'%proteome.lower(), 'r') as f:
    tripletparams_malaria = json.load(f)
    loglikelihood_pathogen = lambda seq: loglikelihood_independent(seq, **tripletparams_malaria)


human_kmers = [seq[i:i+k] for h, seq in fasta_iter(human) for i in range(len(seq)-k+1)
              if isvalidaa(seq[i:i+k])]
pathogen_kmers = [seq[i:i+k] for h, seq in fasta_iter(proteome_path(proteome)) for i in range(len(seq)-k+1)
                 if isvalidaa(seq[i:i+k])]
#human_kmers = random.sample(human_kmers, nkmers)
#pathogen_kmers = random.sample(pathogen_kmers, nkmers)

logp_hh = np.array([loglikelihood_human(kmer) for kmer in human_kmers])
logp_hp = np.array([loglikelihood_pathogen(kmer) for kmer in human_kmers])
logp_pp = np.array([loglikelihood_pathogen(kmer) for kmer in pathogen_kmers])
logp_ph = np.array([loglikelihood_human(kmer) for kmer in pathogen_kmers])

#logp_hh = np.array([loglikelihood_human(seq[i:i+k]) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1)
#                if isvalidaa(seq[i:i+k])])
#
#logp_hp = np.array([loglikelihood_pathogen(seq[i:i+k]) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1)
#               if isvalidaa(seq[i:i+k])])
#
#logp_pp = np.array([loglikelihood_pathogen(seq[i:i+k]) for h, seq in fasta_iter(proteome_path(pathogen)) for i in range(len(seq)-k+1)
#               if isvalidaa(seq[i:i+k])])
#
#logp_ph = np.array([loglikelihood_human(seq[i:i+k]) for h, seq in fasta_iter(proteome_path(pathogen)) for i in range(len(seq)-k+1)
#               if isvalidaa(seq[i:i+k])])

np.savez('data/%s.npz'%proteome, logp_hh=logp_hh, logp_hp=logp_hp, logp_pp=logp_pp, logp_ph=logp_ph)
