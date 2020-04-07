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
likelihood_cov2 = pd.read_csv('data/proteome-ref%s-k%i-SARSCoV2.zip'%(ref, k))['likelihoods']
likelihood_flua = pd.read_csv('data/proteome-ref%s-k%i-InfluenzaA.zip'%(ref, k))['likelihoods']

fig, ax = plt.subplots(figsize=(3.4, 2.0))
ps = [likelihood_human, likelihood_virus, likelihood_cov2, likelihood_flua]
labels = ['human', 'viruses (averaged)', 'SARS-CoV-2', 'InfluenzaA']
plot_histograms(ps, labels, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
ax.set_xlim(-14, -9)
ax.set_ylim(0.0)
ax.set_ylabel('probability density')
ax.set_xlabel('$log_2$ likelihood')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-SARSCoV2-%s-k%i.png' % (ref, k), dpi=300)
