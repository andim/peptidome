import itertools, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

from numba import njit

k = int(snakemake.wildcards.k)
nsample = k
output = True
q = naminoacids
pseudocount = 1.0
niter = 200
stepsize = 0.01 
nsteps = 1e6
nburnin = 1e3

prng = np.random

matrix = load_matrix('data/train_matrix_k%i.csv.gz'%k)

arr = np.load('data/Human_ncov_k%i.npz'%k)
h = arr['h']
J = arr['J']

def sampler(*args, **kwargs):
    return mcmcsampler(*args, nsteps=nsteps, nsample=nsample, nburnin=nburnin)
h, J, J2 = fit_nskew(matrix, sampler=sampler, h=h, J=J,
                niter=niter, pseudocount=pseudocount,
                epsilon=stepsize, prng=prng, output=output)

@njit
def jump(x):
    return local_jump_jit(x, q)
@njit
def energy(x):
    return energy_nskew(x, h, J, J2)
x0 = prng.randint(q, size=k)
nsteps_generate = int(matrix.shape[0]*nsample)
model_matrix = mcmcsampler(x0, energy, jump, nsteps=nsteps_generate,
                           nsample=nsample, nburnin=nburnin)
np.savetxt('data/model_nskew_matrix_k%i.csv.gz'%k, model_matrix, fmt='%i')
np.savez('data/Human_nskew_k%i.npz'%k, h=h, J=J, J2=J2)
