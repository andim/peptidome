import itertools, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

from numba import njit

output = True
L = 9
q = naminoacids
pseudocount = 1.0
niter = 30
stepsize = 0.1
nsteps = 1e7
nsample = L
nburnin = 1e3

prng = np.random

matrix = load_matrix('data/train_matrix.csv.gz')
fi = frequencies(matrix, num_symbols=q, pseudocount=pseudocount)
fij = pair_frequencies(matrix, num_symbols=q, fi=fi, pseudocount=pseudocount)

def sampler(*args, **kwargs):
    return mcmcsampler(*args, nsteps=nsteps, nsample=nsample, nburnin=nburnin)
hi, Jij = fit_full_potts(fi, fij, sampler=sampler, niter=niter,
                         epsilon=stepsize, prng=prng, output=output)

@njit
def jump(x):
    return local_jump_jit(x, q)
@njit
def energy(x):
    return energy_potts(x, hi, Jij)
x0 = prng.randint(q, size=L)
nsteps_generate = int(matrix.shape[0]*nsample)
model_matrix = mcmcsampler(x0, energy, jump, nsteps=nsteps_generate,
                           nsample=nsample, nburnin=nburnin)
np.savetxt('data/model_matrix.csv.gz', model_matrix, fmt='%i')

np.savez('data/Human_reference_%g.npz'%(L), hi=hi, Jij=Jij)
