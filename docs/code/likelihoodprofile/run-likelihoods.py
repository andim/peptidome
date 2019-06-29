import os.path
import json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'
with open(datadir+ 'triplet-%s.json'%ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

def run(name, path, proteinname=True):
    print(name)
    likelihoods = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    if proteinname:
        protein = np.array([h.split('|')[1] for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    else:
        protein = np.array([ind for ind, (h, seq) in enumerate(fasta_iter(path)) for i in range(len(seq)-k+1) ])
    df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, protein=protein))
    df.to_csv('data/proteome-ref%s-k%i-%s.zip'%(ref, k, name), compression='zip', index=False, float_format='%.4f')

path = datadir+'human-viruses-uniref90_nohiv.fasta'
pathout = 'data/proteome-ref%s-k%i-%s.zip'%(ref, k, 'Viruses')
if not os.path.exists(pathout):
    run('Viruses', pathin, proteinname=False)

proteomes = load_proteomes()
for name, row in proteomes.iterrows():
    path = datadir + row['path']
    pathout = 'data/proteome-ref%s-k%i-%s.zip'%(ref, k, name)
    if not os.path.exists(pathout):
        run(name, path)
