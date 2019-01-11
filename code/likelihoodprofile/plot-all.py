import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'

phuman = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']

with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'


df_ts = load_iedb_tcellepitopes(human_only=True)
df_ts['length'] = [len(d) for d in df_ts['Epitope', 'Description']]
df_bs = load_iedb_bcellepitopes(human_only=True)
df_bs['length'] = [len(d) for d in df_bs['Epitope', 'Description']]
df_t = df_ts[~df_ts['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)]
df_b = df_bs[~df_bs['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)]

epi_t = list(df_t['Epitope', 'Description'])
pepitope_t = np.array([loglikelihood(seq[i:i+k], k) for seq in epi_t for i in range(len(seq)-k+1)])
weights_t = np.array([1.0/(len(seq)-k+1) for seq in epi_t for i in range(len(seq)-k+1)])

epi_b = list(df_b['Epitope', 'Description'])
pepitope_b = np.array([loglikelihood(seq[i:i+k], k) for seq in epi_b for i in range(len(seq)-k+1)])
weights_b = np.array([1.0/(len(seq)-k+1) for seq in epi_b for i in range(len(seq)-k+1)])

print(weights_t)
print(len(phuman), len(pepitope_t), len(pepitope_b))

fig, ax = plt.subplots()
ps = [phuman, pepitope_t, pepitope_b]
labels = ['human', 'T epitopes', 'B epitopes']
weights = [np.ones(len(phuman)), weights_t, weights_b]
plot_histograms(ps, labels, weights=weights, xmin=-14, xmax=-9, ax=ax)
ax.set_ylabel('frequency')
ax.set_xlabel('probability given human proteome statistics')
plt.title('all')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-all-%s-k%i.png' % (likelihoodname, k), dpi=300)
