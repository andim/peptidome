import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *

with open(datadir+ 'triplet-human.json', 'r') as f:
    tripletparams = json.load(f)

dfproteomes = pd.read_csv(datadir+ 'proteomes.csv', sep=',')
pathogenproteomes = dfproteomes[dfproteomes['type'].isin(['bacterium', 'virus', 'parasite'])]

df_ts = load_iedb_tcellepitopes(human_only=True)
df_ts['length'] = [len(d) for d in df_ts['Epitope', 'Description']]
df_bs = load_iedb_bcellepitopes(human_only=True)
df_bs['length'] = [len(d) for d in df_bs['Epitope', 'Description']]

loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'
#loglikelihood = lambda seq, k: loglikelihood_independent(seq, humanaaprobdict, k=k)
#likelihoodname = 'independent'

#for k in [9, 15]:
for k in [9]:
    phuman = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1) ])

    df_t = df_ts[df_ts['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)]
    df_b = df_bs[df_bs['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)]

    #epi = list(df_t[df_t['length']==k]['Description'])
    #pepitope = np.array([loglikelihood(seq, k) for seq in epi])
    epi_t = list(df_t['Epitope', 'Description'])
    pepitope_t = np.array([loglikelihood(seq[i:i+k], k) for seq in epi_t for i in range(len(seq)-k+1)])
    pepitope_t = pepitope_t[~np.isnan(pepitope_t)]

    epi_b = list(df_b['Epitope', 'Description'])
    pepitope_b = np.array([loglikelihood(seq[i:i+k], k) for seq in epi_b for i in range(len(seq)-k+1)])
    pepitope_b = pepitope_b[~np.isnan(pepitope_b)]


    print(len(phuman), len(pepitope_t), len(pepitope_b))

    if (len(pepitope_t) > 100) or (len(pepitope_b) > 100):
        fig, ax = plt.subplots()
        ps = [phuman]
        labels = ['human']
        if len(pepitope_t) > 100:
            ps.append(pepitope_t)
            labels.append('T epitopes')
        if len(pepitope_b) > 100:
            ps.append(pepitope_b)
            labels.append('B epitopes')
        plot_histograms(ps, labels, xmin=-14, xmax=-9, ax=ax)
        ax.set_ylabel('frequency')
        ax.set_xlabel('probability given human proteome statistics')
        plt.title('all human')
        fig.tight_layout()
        fig.savefig('plots/likelihoodprofile-allhuman-%s-k%i.png' % (likelihoodname, k), dpi=300)
