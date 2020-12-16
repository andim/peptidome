import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *


k = int(snakemake.wildcards.k)
q = naminoacids

integration_intervals = 20
mcmc_kwargs = dict(nsteps=1e7, nsample=k, nburnin=1e4)

def entropy_thermodynamic_integration(model_params,
        integration_intervals=1,
        mcmc_kwargs=dict(), prng=np.random):

    @njit
    def jump(x):
        return local_jump_jit(x, q)

    if params.files == ['f']:
        model = 'independent'
        f = params['f']
        h = np.log(f)
        h -= np.mean(h)
    elif params.files == ['h', 'J']:
        model = 'ncov'
        h = params['h']
        J = params['J']

        @njit
        def energy(x):
            return energy_ncov(x, h, J)
        @njit
        def energy_alpha_gen(x, alpha):
            return energy_ncov(x, h, alpha*J)
        @njit
        def deltaenergy(x):
            return energy_ncov(x, np.zeros_like(h), J)
    elif params.files == ['h', 'J', 'J2']:
        model = 'nskew'
        h = params['h']
        J = params['J']
        J2 = params['J2']

        @njit
        def energy(x):
            return energy_nskew(x, h, J, J2)
        @njit
        def energy_alpha_gen(x, alpha):
            return energy_nskew(x, h, alpha*J, alpha*J2)
        @njit
        def deltaenergy(x):
            return energy_nskew(x, np.zeros_like(h), J, J2)
    elif params.files == ['h', 'J', 'J2', 'hi', 'Jij']:
        model = 'nskewfcov'
        h = params['h']
        hi = params['hi']
        J = params['J']
        J2 = params['J2']
        Jij = params['Jij']

        @njit
        def energy(x):
            return energy_nskewfcov(x, h, J, J2, hi, Jij)
        @njit
        def energy_alpha_gen(x, alpha):
            return energy_nskewfcov(x, h, alpha*J, alpha*J2, hi, alpha*Jij)
        @njit
        def deltaenergy(x):
            return energy_nskewfcov(x, np.zeros_like(h), J, J2, hi, Jij)

    F0 = -k*np.log(np.sum(np.exp(h)))
    if model == 'independent':
        S = -k*np.sum(f*np.log(f))
        return S, S+F0, F0
    
    x0 = prng.randint(q, size=k)
    matrix = mcmcsampler(x0, energy, jump, **mcmc_kwargs)
    energy = np.array([energy(x) for x in matrix])
    energy_mean = np.mean(energy)

    def Fprime(alpha):
        @njit
        def energy_alpha(x):
            return energy_alpha_gen(x, alpha)
        x0 = prng.randint(q, size=k)
        matrix = mcmcsampler(x0, energy_alpha, jump, **mcmc_kwargs)
        return np.mean([deltaenergy(x) for x in matrix])

    xs = np.linspace(0, 1, integration_intervals+1)
    Fprimes = [Fprime(x) for x in xs]
    Fint = scipy.integrate.simps(Fprimes, xs)
    
    F =  F0 + Fint
    S = energy_mean - F
    return S, energy_mean, F

params = np.load(snakemake.input[0])
S, E, F = entropy_thermodynamic_integration(params,
        integration_intervals=integration_intervals,
        mcmc_kwargs=mcmc_kwargs)
series = pd.Series(data=[S, E, F], index=['S', 'E', 'F'])
series.to_csv(snakemake.output[0])

