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
niter = 30
stepsize = 0.1
nsteps = 1e7
nsample = 10

proteome = proteome_path('Human')
seed = 1234
prng = np.random.RandomState(seed)

seqs = [s for s in fasta_iter(proteome, returnheader=False)]
train, test = train_test_split(seqs, test_size=0.5, random_state=prng)

for label, data in [('train', train), ('test', test)]:
    arr =  np.array([list(kmer) for kmer in to_kmers(data, k=N)])
    matrix = map_matrix(arr, map_)
    np.savetxt('data/%s_matrix.csv.gz'%label, matrix, fmt='%i')
    if label == 'train':
        fi = frequencies(matrix, num_symbols=q, pseudocount=1e-3)
        fij = pair_frequencies(matrix, num_symbols=q, fi=fi, pseudocount=1e-3)

def sampler(*args, **kwargs):
    return mcmcsampler(*args, nsteps=nsteps, nsample=nsample, **kwargs)
hi, Jij = fit_full_potts(fi, fij, sampler=sampler, niter=niter,
                         epsilon=stepsize, prng=prng, output=output)

jump = lambda x: local_jump(x, q)
x0 = prng.randint(q, size=N)
nsteps_generate = int(matrix.shape[0]/nsample)
model_matrix = mcmcsampler(x0, lambda x: energy_potts(x, hi, Jij), jump,
                           nsteps=nsteps_generate, nsample=nsample, prng=prng)
np.savetxt('data/model_matrix.csv.gz', model_matrix, fmt='%i')

np.savez('data/Human_full_k%g.npz'%N, hi=hi, Jij=Jij)
