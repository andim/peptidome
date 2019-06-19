import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *
plt.style.use('../peptidome.mplstyle')


k = 9
ref = 'human'

likelihood_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']

with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'


df_ts = load_iedb_tcellepitopes(human_only=True)
mask = ~(df_ts['Assay', 'Qualitative Measure'] == 'Negative')
likelihoods_t, weights_t = likelihoods_epitopes(df_ts[mask]['Epitope', 'Description'], loglikelihood, k)
likelihoods_t_neg, weights_t_neg = likelihoods_epitopes(df_ts[~mask]['Epitope', 'Description'], loglikelihood, k)
#df_bs = load_iedb_bcellepitopes(human_only=True)
#df_b = df_bs[~df_bs['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)]
#likelihoods_b, weights_b = likelihoods_epitopes(df_b['Epitope', 'Description'], loglikelihood, k)

print(len(likelihood_human), len(likelihoods_t))#, len(likelihoods_b))

fig, ax = plt.subplots()
ps = [likelihood_human, likelihoods_t, likelihoods_t_neg]#, likelihoods_b]
labels = ['Human proteins', 'IEDB positive', 'IEDB negative']#, 'B epitopes']
weights = [np.ones(len(likelihood_human)), weights_t, weights_t_neg]#, weights_b]
plot_histograms(ps, labels, weights=weights, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
ax.set_xlim(-14, -9)
ax.set_ylabel('Frequency')
ax.set_xlabel('$log_{10}$ Likelihood under human proteome statistics')
ax.legend(title='Peptide')
#plt.title('all')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-all-%s-k%i.png' % (likelihoodname, k), dpi=300)
fig.savefig('../../paper/images/likelihoodprofile-iedb-tcell.pdf')
