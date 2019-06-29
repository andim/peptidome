import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'

likelihood_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']
likelihood_virus = pd.read_csv('data/proteome-ref%s-k%i-Viruses.zip'%(ref, k))['likelihoods']

with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

df_ts = load_iedb_tcellepitopes(human_only=True)
mask = ~df_ts['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)
mask &= df_ts['Epitope', 'Parent Species'].str.contains('virus', case=False, na=False)
mask &= ~df_ts['Epitope', 'Parent Species'].str.contains('Human immunodeficiency virus 1', case=False, na=False)
df_t = df_ts[mask]
likelihoods_t, weights_t = likelihoods_epitopes(df_t['Epitope', 'Description'].unique(), loglikelihood, k)
df_bs = load_iedb_bcellepitopes(human_only=True)
mask = ~df_bs['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)
mask &= df_bs['Epitope', 'Parent Species'].str.contains('virus', case=False, na=False)
mask &= ~df_bs['Epitope', 'Parent Species'].str.contains('Human immunodeficiency virus 1', case=False, na=False)
df_b = df_bs[mask]
likelihoods_b, weights_b = likelihoods_epitopes(df_b['Epitope', 'Description'].unique(), loglikelihood, k)

print(len(likelihood_human), len(likelihood_virus), len(likelihoods_t), len(likelihoods_b))

fig, ax = plt.subplots(figsize=(3.4, 2.0))
ps = [likelihood_human, likelihood_virus, likelihoods_t, likelihoods_b]
labels = ['human', 'viruses', 'T cell epitopes', 'B cell epitopes']
weights = [np.ones(len(likelihood_human)), np.ones(len(likelihood_virus)), weights_t, weights_b]
plot_histograms(ps, labels, weights=weights, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
ax.set_xlim(-14, -9)
ax.set_ylabel('probability density')
ax.set_xlabel('$log_2$ likelihood')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-Viruses-%s-k%i.png' % (likelihoodname, k), dpi=300)
fig.savefig('../../paper/images/viruses.pdf', dpi=300)
