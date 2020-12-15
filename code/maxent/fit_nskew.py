import itertools, json
import numpy as np
np.seterr(all='raise')
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

from numba import njit

k = int(snakemake.wildcards.k)
output = True
pseudocount = 1.0
params = snakemake.params.fit
niter = params['niter']
stepsize = params['stepsize']
nsteps = params['nsteps']
nburnin = params['nburnin']
nsample = params['nsample']

q = naminoacids
prng = np.random

matrix = load_matrix(snakemake.input[0])
arr = np.load(snakemake.input[1])
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
np.savetxt(snakemake.output[0], model_matrix, fmt='%i')
np.savez(snakemake.output[1], h=h, J=J, J2=J2)
