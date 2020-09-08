import itertools, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

output = True
aas_arr = np.array(list(aminoacids))
L = 9
q = naminoacids
pseudocount = 1e-3
niter = 30
stepsize = 0.1
mcmc_kwargs = dict(nsteps=1e6, nsample=20, nburnin=1e4)
extra = [(datadir+'human-viruses-swissprot.fasta', 'viruses')]

if __name__ == "__main__":
    proteomes = load_proteomes()
    if len(sys.argv) < 2:
        print(proteomes.shape[0])
    else:
        idx = int(sys.argv[1])-1
        if idx < proteomes.shape[0]:
            row = proteomes.iloc[idx]
            name = row.name
            proteome = proteome_path(name)
        else:
            proteome, name = extra[idx-proteomes.shape[0]] 
        print(name)

        matrix = kmers_to_matrix(to_kmers(fasta_iter(proteome, returnheader=False), k=L))
        fi = frequencies(matrix, num_symbols=q, pseudocount=pseudocount)
        fij = pair_frequencies(matrix, num_symbols=q, fi=fi, pseudocount=pseudocount)

        prng = np.random
        def sampler(*args, **kwargs):
            mcmc_kwargs.update(kwargs)
            return mcmcsampler(*args, **mcmc_kwargs)
        hi, Jij = fit_full_potts(fi, fij, sampler=sampler, niter=niter,
                                 epsilon=stepsize, prng=prng, output=output)

        np.savez('data/%s_%g.npz'%(name, L), hi=hi, Jij=Jij)
