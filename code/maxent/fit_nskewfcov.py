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
nsteps = 1e7
nburnin = 1e3

prng = np.random

matrix = load_matrix('data/train_matrix_k%i.csv.gz'%k)

arr = np.load('data/Human_nskew_k%i.npz'%k)
h = arr['h']
J = arr['J']
J2 = arr['J2']

def sampler(*args, **kwargs):
    return mcmcsampler(*args, nsteps=nsteps, nsample=nsample, nburnin=nburnin)
h, J, J2, hi, Jij = fit_nskewfcov(matrix, sampler=sampler, h=h, J=J,
                niter=niter, pseudocount=pseudocount,
                epsilon=stepsize, prng=prng, output=output)

@njit
def jump(x):
    return local_jump_jit(x, q)
@njit
def energy(x):
    return energy_nskewfcov(x, h, J, J2, hi, Jij)
x0 = prng.randint(q, size=k)
nsteps_generate = int(matrix.shape[0]*nsample)
model_matrix = mcmcsampler(x0, energy, jump, nsteps=nsteps_generate,
                           nsample=nsample, nburnin=nburnin)
np.savetxt('data/model_nskewfcov_matrix_k%i.csv.gz'%k, model_matrix, fmt='%i')
np.savez('data/Human_nskewfcov_k%i.npz'%k, h=h, J=J, J2=J2, hi=hi, Jij=Jij)
