import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

mcmc_kwargs = dict(nsteps=1e6, nsample=20, nburnin=1e4)
integration_intervals = 5
L = 9

params = np.load('data/Human_9.npz')
hi_human = params['hi']
Jij_human = params['Jij']

def entropy_thermodynamic_integration(hi, Jij, integration_intervals=1,
        mcmc_kwargs=dict(), prng=np.random):
    F0 = -np.sum(np.log(np.sum(np.exp(hi), axis=1)))
    N, q = hi.shape
    
    jump = lambda x: local_jump(x, q)
    x0 = prng.randint(q, size=N)
    matrix = mcmcsampler(x0, lambda x: energy_potts(x, hi, Jij), jump, **mcmc_kwargs)
    energy_mean = np.mean([energy_potts(x, hi, Jij) for x in matrix])
    energy_human = np.mean([energy_potts(x, hi_human, Jij_human) for x in matrix])
    
    def Fprime(alpha):
        jump = lambda x: local_jump(x, q)
        x0 = prng.randint(q, size=N)
        matrix = mcmcsampler(x0, lambda x: energy_potts(x, hi, alpha*Jij), jump, **mcmc_kwargs)
        return np.mean([energy_potts(x, np.zeros_like(hi), Jij) for x in matrix])
    
    xs = np.linspace(0, 1, integration_intervals+1)
    Fprimes = [Fprime(x) for x in xs]
    Fint = scipy.integrate.simps(Fprimes, xs)
   
    F = F0 + Fint
    S = energy_mean - F
    return S, energy_mean, F, energy_human

proteomes = load_proteomes()
if len(sys.argv) < 2:
    print(proteomes.shape[0])
else:
    row = proteomes.iloc[int(sys.argv[1])-1]
    name = row.name
    path = 'data/%s_%g.npz'%(name, L)
    params = np.load(path)
    hi = params['hi']
    Jij = params['Jij']
    S, E, F, Ehuman = entropy_thermodynamic_integration(hi, Jij,
            integration_intervals=integration_intervals, mcmc_kwargs=mcmc_kwargs)
    with open('data/entropies.csv', 'a') as f:
        f.write(','.join([str(s) for s in [name, S, E, F, Ehuman]]))
