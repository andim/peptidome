import os.path
import json
import glob
import numpy as np
import pandas as pd

import sys
sys.path.append('..')

from lib import *

k = 5
ref = 'human'
with open(datadir+ 'triplet-%s.json'%ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

def run(name, path, proteinname=True, sequence=False):
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
    df.to_csv('data/proteome-ref%s-k%i-%s.zip'%(ref, k, name), compression='zip', index=False, float_format='%.4f')

# All viruses
path = datadir+'human-viruses-uniref90_nohiv.fasta'
pathout = 'data/proteome-ref%s-k%i-%s.zip'%(ref, k, 'Viruses')
if not os.path.exists(pathout):
    run('Viruses', path, proteinname=False)

# Cancer datasets
filenames = ['frameshifts.fasta.gz']
for filename in filenames:
    name = filename.split('.')[0]
    path = datadir+'cancer/' + filename
    pathout = 'data/proteome-ref%s-k%i-%s.zip'%(ref, k, name)
    if not os.path.exists(pathout):
        run(name, path, proteinname=False)

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
    df.to_csv('data/proteome-ref%s-k%i-%s.zip'%(ref, k, name), compression='zip', index=False, float_format='%.4f')


# SARS CoV 2 dataset
filenames = ['SARSCoV2.fasta']
for filename in filenames:
    name = filename.split('.')[0]
    path = datadir + filename
    pathout = 'data/proteome-ref%s-k%i-%s.zip'%(ref, k, name)
    if not os.path.exists(pathout):
        run(name, path, proteinname=False)

# Proteomes
proteomes = load_proteomes()
for name, row in proteomes.iterrows():
    path = datadir + row['path']
    pathout = 'data/proteome-ref%s-k%i-%s.zip'%(ref, k, name)
    if not os.path.exists(pathout):
        run(name, path)
