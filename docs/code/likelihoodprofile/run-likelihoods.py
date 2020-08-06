import os.path
import json
import glob
import numpy as np
import pandas as pd

import sys
sys.path.append('..')

from lib import *
from lib.maxent import *

k = 9
ref = 'human'
#with open(datadir+ 'triplet-%s.json'%ref, 'r') as f:
#    tripletparams = json.load(f)
#loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
#likelihoodname = 'triplet'

params = np.load('../maxent/data/Human_reference_9.npz')
hi = params['hi']
Jij = params['Jij']
loglikelihood = lambda seq, k: -energy_potts(map_aatonumber(seq.upper()), hi, Jij) if isvalidaa(seq) else np.nan
likelihoodname = 'maxent'

def run(name, path, pathout, proteinname=True, sequence=False):
    print(name)
    likelihoods = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    if sequence:
        sequence = np.array([seq[i:i+k] for h, seq in fasta_iter(path) for i in range(len(seq)-k+1)])
    if proteinname:
        protein = np.array([h.split('|')[1] for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    else:
        protein = np.array([ind for ind, (h, seq) in enumerate(fasta_iter(path)) for i in range(len(seq)-k+1) ])
    if sequence:
        df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, protein=protein, sequence=sequence))
    else:
        df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, protein=protein))
    df.dropna(inplace=True)
    df.to_csv(pathout, compression='zip', index=False, float_format='%.4f')

# All viruses
path = datadir+'human-viruses-uniref90_nohiv.fasta'
pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, 'Viruses')
if not os.path.exists(pathout):
    run('Viruses', path, pathout, proteinname=False)

# Cancer datasets
filenames = ['frameshifts.fasta.gz']
for filename in filenames:
    name = filename.split('.')[0]
    path = datadir+'cancer/' + filename
    pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, name)
    if not os.path.exists(pathout):
        run(name, path, pathout, proteinname=False)

# Ufo datasets
filenames = glob.glob(datadir + 'ufos/*.csv')
for filename in filenames:
    name = filename.split('/')[-1].split('.')[0]
    print(name)
    df_in = pd.read_csv(filename, sep='\t')
    sequences = np.array([seq[i:i+k] for seq in df_in['AA_seq'] for i in range(len(seq)-k+1)])
    likelihoods = np.array([loglikelihood(seq, k) for seq in sequences])
    df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, sequence=sequences))
    df.dropna(inplace=True)
    pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, name)
    df.to_csv(pathout, compression='zip', index=False, float_format='%.4f')

    # only middle part
    sequences = np.array([seq[i:i+k] for seq in df_in['AA_seq'] for i in range(10, min(len(seq)-k+1, 51))])
    likelihoods = np.array([loglikelihood(seq, k) for seq in sequences])
    df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, sequence=sequences))
    df.dropna(inplace=True)
    df.to_csv('data/proteome-ref%s-k%i-%s-middle.zip'%(ref, k, name), compression='zip', index=False, float_format='%.4f')



# SARS CoV 2 dataset
filenames = ['SARSCoV2.fasta']
for filename in filenames:
    name = filename.split('.')[0]
    path = datadir + filename
    pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, name)
    if not os.path.exists(pathout):
        run(name, path, pathout, proteinname=False)

# Proteomes
proteomes = load_proteomes()
for name, row in proteomes.iterrows():
    path = datadir + row['path']
    pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, name)
    if not os.path.exists(pathout):
        run(name, path, pathout, proteinname=False)
