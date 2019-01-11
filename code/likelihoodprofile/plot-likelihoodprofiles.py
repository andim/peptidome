import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'

likelihoods_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']

with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

pathogenproteomes = load_proteomes(only_pathogens=True)

df_ts = load_iedb_tcellepitopes(human_only=True)
#df_ts['length'] = [len(d) for d in df_ts['Epitope', 'Description']]
df_bs = load_iedb_bcellepitopes(human_only=True)
#df_bs['length'] = [len(d) for d in df_bs['Epitope', 'Description']]


for name, row in pathogenproteomes.iterrows():
    iedbname = row['iedbname']
    path = datadir + row['path']
    print(name)

    likelihoods_pathogen = pd.read_csv('data/proteome-ref%s-k%i-%s.zip'%(ref, k, name))['likelihoods']

    df_t = df_ts[df_ts['Epitope', 'Parent Species'].str.contains(iedbname, na=False)]
    likelihoods_t, weights_t = likelihoods_epitopes(df_t['Epitope', 'Description'].unique(), loglikelihood, k)
    df_b = df_bs[df_bs['Epitope', 'Parent Species'].str.contains(iedbname, na=False)]
    likelihoods_b, weights_b = likelihoods_epitopes(df_b['Epitope', 'Description'].unique(), loglikelihood, k)

    if (len(likelihoods_t) > 100) or (len(likelihoods_b) > 100):
        fig, ax = plt.subplots()
        ps = [likelihoods_human, likelihoods_pathogen]
        labels = ['human', 'pathogen']
        weights = [np.ones(len(likelihoods_human)), np.ones(len(likelihoods_pathogen))]
        if len(likelihoods_t) > 100:
            ps.append(likelihoods_t)
            labels.append('T epitopes')
            weights.append(weights_t)
        if len(likelihoods_b) > 100:
            ps.append(likelihoods_b)
            labels.append('B epitopes')
            weights.append(weights_b)
        plot_histograms(ps, labels, weights=weights, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
        ax.set_xlim(-14, -9)
        ax.set_ylabel('probability density')
        ax.set_xlabel('$log_2$ likelihood')
        plt.title(name)
        fig.tight_layout()
        fig.savefig('plots/likelihoodprofile-%s-%s-k%i.png' % (name, likelihoodname, k), dpi=300)
        plt.close()
