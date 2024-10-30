import sys
import os

import numpy as np

repopath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from numba import jit, njit

sys.path.append("..")
sys.path.append(f"{repopath}/code/lib")
from lib import *
from lib.maxent import *

plt.style.use("../peptidome.mplstyle")


def entropy_thermodynamic_integration(
    model_params, integration_intervals=1, mcmc_kwargs=dict(), prng=np.random
):

    @njit
    def jump(x):
        return local_jump_jit(x, q)

    if params.files == ["f"]:
        model = "independent"
        f = params["f"]
        h = np.log(f)
        h -= np.mean(h)
    elif params.files == ["h", "J"]:
        model = "ncov"
        h = params["h"]
        J = params["J"]

        @njit
        def energy(x):
            return energy_ncov(x, h, J)

        @njit
        def energy_alpha_gen(x, alpha):
            return energy_ncov(x, h, alpha * J)

        @njit
        def deltaenergy(x):
            return energy_ncov(x, np.zeros_like(h), J)

    elif params.files == ["h", "J", "J2"]:
        model = "nskew"
        h = params["h"]
        J = params["J"]
        J2 = params["J2"]

        @njit
        def energy(x):
            return energy_nskew(x, h, J, J2)

        @njit
        def energy_alpha_gen(x, alpha):
            return energy_nskew(x, h, alpha * J, alpha * J2)

        @njit
        def deltaenergy(x):
            return energy_nskew(x, np.zeros_like(h), J, J2)

    elif params.files == ["h", "J", "J2", "hi", "Jij"]:
        model = "nskewfcov"
        h = params["h"]
        hi = params["hi"]
        J = params["J"]
        J2 = params["J2"]
        Jij = params["Jij"]

        @njit
        def energy(x):
            return energy_nskewfcov(x, h, J, J2, hi, Jij)

        @njit
        def energy_alpha_gen(x, alpha):
            return energy_nskewfcov(x, h, alpha * J, alpha * J2, hi, alpha * Jij)

        @njit
        def deltaenergy(x):
            return energy_nskewfcov(x, np.zeros_like(h), J, J2, hi, Jij)

    F0 = -k * np.log(np.sum(np.exp(h)))
    if model == "independent":
        S = -k * np.sum(f * np.log(f))
        sample_size = (mcmc_kwargs["nsteps"] - mcmc_kwargs["nburnin"]) // mcmc_kwargs[
            "nsample"
        ]
        return (
            S,
            S + F0,
            F0,
            np.random.choice(np.arange(0, 20, 1), size=((sample_size), 9), p=f),
        )

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

    xs = np.linspace(0, 1, integration_intervals + 1)
    Fprimes = [Fprime(x) for x in xs]
    try:
        Fint = scipy.integrate.simps(Fprimes, xs)
    except AttributeError:
        Fint = scipy.integrate.simpson(Fprimes, x=xs)

    F = F0 + Fint
    S = energy_mean - F
    return S, energy_mean, F, matrix


if __name__ == "__main__":

    k = 9
    q = naminoacids

    integration_intervals = 10

    [proteome, model, size, thinning] = sys.argv[1].split("_")

    burnin = 1e4
    N = float(size)
    thinning = int(thinning)
    nsteps = (N * thinning) + burnin

    mcmc_kwargs = dict(nsteps=int(nsteps), nsample=thinning, nburnin=int(burnin))

    params = np.load(f"data/{proteome}_{model}_k9_params.npz")

    S, E, F, matrix = entropy_thermodynamic_integration(
        params, integration_intervals=integration_intervals, mcmc_kwargs=mcmc_kwargs
    )
    series = pd.Series(data=[S, E, F], index=["S", "E", "F"])
    series.to_csv(f"data/mcmc_thermo/{proteome}_{model}_{size}_{thinning}_entropy.csv")
    data_path = f"data/mcmc_thermo/{proteome}_{model}_{size}_{thinning}_matrix.csv.gz"

    np.savetxt(data_path, matrix, fmt="%i")
