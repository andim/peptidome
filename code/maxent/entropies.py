import glob
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

mcmc_kwargs = dict(nsteps=1e6, nsample=10, nburnin=1e3)
integration_intervals = 5


def entropy_thermodynamic_integration(hi, Jij, integration_intervals=1,
        mcmc_kwargs=dict(), prng=np.random):
    F0 = -np.sum(np.log(np.sum(np.exp(hi), axis=1)))
    N, q = hi.shape
    
    jump = lambda x: local_jump(x, q)
    x0 = prng.randint(q, size=N)
    matrix = mcmcsampler(x0, lambda x: energy_potts(x, hi, Jij), jump, **mcmc_kwargs)
    energy_mean = np.mean([energy_potts(x, hi, Jij) for x in matrix])
    
    def Fprime(alpha):
        jump = lambda x: local_jump(x, q)
        x0 = prng.randint(q, size=N)
        matrix = mcmcsampler(x0, lambda x: energy_potts(x, hi, alpha*Jij), jump, **mcmc_kwargs)
        return np.mean([energy_potts(x, np.zeros_like(hi), Jij) for x in matrix])
    
    xs = np.linspace(0, 1, integration_intervals+1)
    Fprimes = [Fprime(x) for x in xs]
    Fint = scipy.integrate.simps(Fprimes, xs)
    
    S = energy_mean - (F0 + Fint)
    return S

for path in glob.glob('data/*_9.npz'):
    params = np.load(path)
    hi = params['hi']
    Jij = params['Jij']
    entropy = entropy_thermodynamic_integration(hi, Jij,
            integration_intervals=integration_intervals, mcmc_kwargs=mcmc_kwargs)
    print(path, entropy)
