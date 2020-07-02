import itertools, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

output = True
N = 9
q = naminoacids
niter = 20
stepsize = 0.1

proteome = proteome_path('Human')
seed = 1234
prng = np.random.RandomState(seed)

seqs = [s for s in fasta_iter(proteome, returnheader=False)]
train, test = train_test_split(seqs, test_size=0.5, random_state=prng)

train_arr =  np.array([list(kmer) for kmer in to_kmers(train, k=N)])
train_matrix = map_matrix(train_arr, map_)

fi = frequencies(train_matrix, num_symbols=q, pseudocount=1e-3)
fij = pair_frequencies(train_matrix, num_symbols=q, fi=fi, pseudocount=1e-3)
cij = compute_covariance_matrix(fi, fij).flatten()

def sampler(*args, **kwargs):
    return mcmcsampler(*args, nsteps=1e7, nsample=10, **kwargs)
hi, Jij = fit_full_potts(fi, fij, sampler=sampler, niter=niter,
                         epsilon=stepsize, prng=prng, output=output)

np.savez('data/Human_full_k%g.npz'%N, hi=hi, Jij=Jij)
