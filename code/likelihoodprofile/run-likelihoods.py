import json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')

from lib import *

ref = 'human'
with open(datadir+ 'triplet-%s.json'%ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

k = 9
proteomes = load_proteomes()

for name, row in proteomes.iterrows():
    print(name)
    iedbname = row['iedbname']
    path = datadir + row['path']
    likelihoods = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    protein = np.array([h.split('|')[1] for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, protein=protein))
    df.to_csv('data/proteome-ref%s-k%i-%s.zip'%(ref, k, name), compression='zip', index=False, float_format='%.4f')
