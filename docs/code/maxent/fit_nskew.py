import itertools, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

from numba import njit

L = 4
nsample = L
output = True
q = naminoacids
pseudocount = 1.0
niter = 200
stepsize = 0.01 
nsteps = 1e6
nburnin = 1e3

prng = np.random

matrix = load_matrix('data/train_matrix_L%i.csv.gz'%L)

arr = np.load('data/Human_ncov_%i.npz'%L)
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
x0 = prng.randint(q, size=L)
nsteps_generate = int(matrix.shape[0]*nsample)
model_matrix = mcmcsampler(x0, energy, jump, nsteps=nsteps_generate,
                           nsample=nsample, nburnin=nburnin)
np.savetxt('data/model_nskew_matrix_L%i.csv.gz'%L, model_matrix, fmt='%i')
np.savez('data/Human_nskew_%i.npz'%L, h=h, J=J, J2=J2)
