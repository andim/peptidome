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

likelihoodname = 'maxent'

likelihood_human = pd.read_csv('data/proteome-ref%s-%s-k%i-Human.zip'%(ref, likelihoodname, k))['likelihoods']
likelihood_virus = pd.read_csv('data/proteome-ref%s-%s-k%i-Viruses.zip'%(ref, likelihoodname, k))['likelihoods']
likelihood_frameshifts = pd.read_csv('data/proteome-ref%s-%s-k%i-frameshifts.zip'%(ref, likelihoodname, k))['likelihoods']
likelihood_pb1ufo = pd.read_csv('data/proteome-ref%s-%s-k%i-pb1ufo.zip'%(ref, likelihoodname, k))['likelihoods']

fig, ax = plt.subplots(figsize=(3.4, 2.0))
ps = [likelihood_human, likelihood_virus, likelihood_frameshifts, likelihood_pb1ufo]
labels = ['human', 'viruses', 'frameshifts', 'pb1 ufo']
plot_histograms(ps, labels, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
ax.set_xlim(-14, -9)
ax.set_ylim(0.0)
ax.set_ylabel('probability density')
ax.set_xlabel('$log_10$ likelihood')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-Cancer-%s-k%i.png' % (ref, k), dpi=300)
fig.savefig('../../paper/images/cancer.pdf', dpi=300)
