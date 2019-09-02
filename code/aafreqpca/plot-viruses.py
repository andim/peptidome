import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition

import sys
sys.path.append('..')
from lib import *
plt.style.use('../peptidome.mplstyle')

data = np.load('data/data.npz')

aa_human = data['human']
aa_viruses = data['viruses']
aa_viruses = aa_viruses[np.random.randint(0, len(aa_viruses), len(aa_human))]
print(aa_viruses.shape, aa_human.shape)

fig, ax = plt.subplots(figsize=(5, 5))#, ncols=2, sharex=True, sharey=True)
pca = sklearn.decomposition.PCA(n_components=2)
pca = pca.fit(np.vstack([aa_human, aa_viruses]))
for i, (label, aa) in enumerate([('Human', aa_human), ('Viral', aa_viruses)]):
    pcad_pathogen = pca.transform(aa)
    ax.scatter(pcad_pathogen[:, 0], pcad_pathogen[:, 1], label=label+' proteins', s=.25)

ax.legend()
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
lim = 0.2
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
fig.tight_layout()
fig.savefig('viruses.png')
plt.show()
