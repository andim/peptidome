import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
import sklearn.decomposition
import sklearn.manifold
import seaborn as sns
import matplotlib.pyplot as plt

from lib import *

with open('../data/triplet-human.json', 'r') as f:
    tripletparams = json.load(f)

dfproteomes = pd.read_csv('../data/proteomes.csv', sep=',')
pathogenproteomes = dfproteomes[dfproteomes['type'].isin(['bacterium', 'virus', 'parasite'])]

dfepitopes = pd.read_csv(datadir+'allhuman-iedb-epitopes.csv', header=1, usecols=(2, 9, 14))
dfepitopes = dfepitopes.fillna('')
dfepitopes['length'] = [len(d) for d in dfepitopes['Description']]

loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'
#loglikelihood = lambda seq, k: loglikelihood_independent(seq, humanaaprobdict, k=k)
#likelihoodname = 'independent'

#for k in [9, 15]:
for k in [9]:
    phuman = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1) ])

    for idx, row in pathogenproteomes.iterrows():
        name = row['shortname']
        iedbname = row['iedbname']
        path = datadir + row['path']
        print(name)

        dfepitope = dfepitopes[dfepitopes['Parent Organism'].str.contains(iedbname)]
        print(len(dfepitope))

        #epi = list(dfepitope[dfepitope['length']==k]['Description'])
        #pepitope = np.array([loglikelihood(seq, k) for seq in epi])
        epi = list(dfepitope['Description'])
        pepitope = np.array([loglikelihood(seq[i:i+k], k) for seq in epi for i in range(len(seq)-k+1)])
        pepitope = pepitope[~np.isnan(pepitope)]

        ppathogen = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
        ppathogen = ppathogen[~np.isnan(ppathogen)]

        print(len(phuman), len(ppathogen), len(pepitope))

        if len(pepitope) > 2:
            fig, ax = plt.subplots()
            plot_histograms([phuman, ppathogen, pepitope], ['human', 'pathogen', 'epitope'], xmin=-14, xmax=-9, ax=ax)
            ax.set_ylabel('frequency')
            ax.set_xlabel('probability given human proteome statistics')
            plt.title(name)
            fig.tight_layout()
            fig.savefig('plots/likelihoodprofile-%s-%s-k%i.png' % (name, likelihoodname, k), dpi=300)
