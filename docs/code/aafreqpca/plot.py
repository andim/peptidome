import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition

import sys
sys.path.append('..')
from lib import *
plt.style.use('../peptidome.mplstyle')

data = np.load('data/data.npz')

aa_cmv = data['cmv']
aa_malaria = data['malaria']
aa_human = data['human']

fig, ax = plt.subplots(figsize=(5, 5))
pca = sklearn.decomposition.PCA(n_components=2)
pca = pca.fit(np.vstack([aa_human, aa_malaria, aa_cmv]))
for label, aa in [('Human', aa_human), ('Malaria', aa_malaria), ('CMV', aa_cmv)]:
    pcad_pathogen = pca.transform(aa)
    ax.scatter(pcad_pathogen[:, 0], pcad_pathogen[:, 1], label=label, s=.5)
ax.legend()
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
lim = 0.25
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
fig.tight_layout()
fig.savefig('main.png')
plt.show()
