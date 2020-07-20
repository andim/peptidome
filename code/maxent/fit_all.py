import itertools, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

output = True
aas_arr = np.array(list(aminoacids))
N = 9
q = naminoacids
pseudocount = 1e-3
niter = 30
stepsize = 0.1
nsample = 2*N
nsteps = 1e6 
nburnin = 1e3
seed = 1234

proteomes = load_proteomes()
if len(sys.argv) < 2:
    print(proteomes.shape[0])
else:
    row = proteomes.iloc[int(sys.argv[1])-1]
    name = row.name
    print(name)

    proteome = proteome_path(name)
    seqs = [s for s in fasta_iter(proteome, returnheader=False)]
    arr =  np.array([list(kmer) for kmer in to_kmers(seqs, k=N)])
    matrix = map_matrix(arr, map_)
    fi = frequencies(matrix, num_symbols=q, pseudocount=pseudocount)
    fij = pair_frequencies(matrix, num_symbols=q, fi=fi, pseudocount=pseudocount)

    prng = np.random.RandomState(seed)
    def sampler(*args, **kwargs):
        return mcmcsampler(*args, nsteps=nsteps, nsample=nsample, nburnin=nburnin, **kwargs)
    hi, Jij = fit_full_potts(fi, fij, sampler=sampler, niter=niter,
                             epsilon=stepsize, prng=prng, output=output)

    np.savez('data/%s_%g.npz'%(name, N), hi=hi, Jij=Jij)
