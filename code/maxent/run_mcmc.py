import argparse
import sys
import os

import numpy as np

repopath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from numba import jit

sys.path.append("..")
sys.path.append(f"{repopath}/code/lib")
from lib import *
from lib.maxent import *

plt.style.use("../peptidome.mplstyle")


# sys.path.append(f"{repopath}/code/lib")


def mcmcsampler(energy, jump, nsteps, nburn, nsample, prng=None):
    """Markov chain Monte carlo sampler.

    energy(x): function for calculating energy
    jump(x): function for calculating a proposed new position
    nburnin: burnin period in which states are not saved
    nsample: sample interval for saving states

    returns array of states
    """
    if not prng:
        prng = np.random
    x0 = prng.randint(20, size=9)
    x = x0
    Ex = energy(x)
    samples = np.zeros((nsteps, x0.shape[0]), dtype=np.int64)
    energies = np.zeros((nsteps, 1), dtype=np.int64)

    # samples[0] = x
    # energies[0] = Ex
    counter = 0
    i = 1
    while counter < nsteps:
        xp = jump(x)
        Exp = energy(xp)
        if (Exp < Ex) or (prng.rand() < np.exp(-Exp + Ex)):
            i += 1
            x = xp
            Ex = Exp
            if (i > nburn) and ((i - nburn) % nsample == 0):
                samples[counter] = x
                energies[counter] = Exp
                counter += 1
    return (samples, energies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        type=str,
        required=True,
        help='one of ["independent", "ncov", "nskew", "nskewfcov", "uniform"]',
    )
    parser.add_argument(
        "-proteome",
        type=str,
        required=True,
        help="one of supported proteomes",
        default="Human",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        required=True,
        help="Directory to store resulting matrices",
    )
    parser.add_argument(
        "-N",
        type=int,
        required=False,
        default=10000,
        help="length of MCMC chain",
    )
    parser.add_argument(
        "-burn",
        type=int,
        required=False,
        help="number of samples to throw out at beginning of chain",
        default="1000",
    )
    parser.add_argument(
        "-nsample",
        type=int,
        required=False,
        help="thinning parameter. only keep every nsample-th index",
        default="10",
    )
    args = parser.parse_args()

    try:
        energy = make_energy(
            np.load(f"./data/{args.proteome}_{args.model}_k9_params.npz")
        )
    except:
        energy = make_energy(
            np.load(f"./data/model_params/{args.proteome}_{args.model}_k9_params.npz")
        )

    @njit
    def jump(x):
        return local_jump_jit(x, q)

    samples, energies = mcmcsampler(
        energy, jump, nsteps=args.N, nburn=args.burn, nsample=args.nsample
    )
    data_path = f"{args.proteome}_{args.model}_{args.N}_steps_{args.burn}_burn_{args.nsample}_thin.csv.gz"
    full_data_path = os.path.join(args.outdir, data_path)

    np.savetxt(full_data_path, samples, fmt="%i")
    plt.figure()
    energy_exp = [np.mean(energies[:i]) for i in range(len(energies))]
    plt.plot(range(len(energy_exp)), energy_exp)
    plt.xlabel("Index")
    plt.ylabel("<E[p]>")
    fig_path = f"{args.proteome}_{args.model}_{args.N}_steps_{args.burn}_burn_{args.nsample}_thin.png"
    full_fig_path = os.path.join(args.outdir, fig_path)

    plt.savefig(full_fig_path)
