import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *

with open(datadir+ 'triplet-human.json', 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

k = 9
dfproteomes = pd.read_csv(datadir+ 'proteomes.csv', sep=',')

for idx, row in dfproteomes.iterrows():
    name = row['shortname']
    iedbname = row['iedbname']
    path = datadir + row['path']
    likelihoods = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    protein = np.array([index for index, (h, seq) in enumerate(fasta_iter(path)) for i in range(len(seq)-k+1) ])
    df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, protein=protein))
    df.to_csv('data/proteome-%s.zip'%name, compression='zip', index=False)
    
