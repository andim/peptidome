import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

integration_intervals = 5
L = 9
mcmc_kwargs = dict(nsteps=1e6, nsample=20, nburnin=1e4)

params = np.load('data/Human_9.npz')
hi_human = params['hi']
Jij_human = params['Jij']

def entropy_thermodynamic_integration(hi, Jij,
        integration_intervals=1,
        mcmc_kwargs=dict(), prng=np.random):
    L, q = hi.shape
    
    @njit
    def jump(x):
        return local_jump_jit(x, q)
    @njit
    def energy(x):
        return energy_potts(x, hi, Jij)
    x0 = prng.randint(q, size=L)
    matrix = mcmcsampler(x0, energy, jump, **mcmc_kwargs)
    energy = np.array([energy(x) for x in matrix])
    energy_human = np.array([energy_potts(x, hi_human, Jij_human) for x in matrix])
    energy_mean = np.mean(energy)
    deltaE = np.mean(energy_human-energy)
    
    F = Fpotts_thermodynamic_integration(hi, Jij,
            integration_intervals=integration_intervals, mcmc_kwargs=mcmc_kwargs)
    S = energy_mean - F
    return S, energy_mean, F, deltaE

proteomes = load_proteomes()
if len(sys.argv) < 2:
    print(proteomes.shape[0])
else:
    idx = int(sys.argv[1])
    row = proteomes.iloc[idx]
    name = row.name
    path = 'data/%s_%g.npz'%(name, L)
    params = np.load(path)
    hi = params['hi']
    Jij = params['Jij']
    S, E, F, Ehuman = entropy_thermodynamic_integration(hi, Jij,
            integration_intervals=integration_intervals, mcmc_kwargs=mcmc_kwargs)
    print(name, S)
    with open('data/entropies.csv', 'a') as f:
        f.write(','.join([str(s) for s in [name, S, E, F, Ehuman]]))
        f.write('\n')
