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
aa_malaria = data['malaria']

malaria_antigen_indices = np.array(pd.read_csv('data/malaria_antigens.csv', header=None)).flatten()
print(malaria_antigen_indices)

fig, ax = plt.subplots(figsize=(5, 5))
pca = sklearn.decomposition.PCA(n_components=2)
pca = pca.fit(np.vstack([aa_human, aa_malaria]))
indices_malaria = data['indices_malaria']
aa_malaria_antigen = np.array(pd.DataFrame(aa_malaria, index=indices_malaria).loc[malaria_antigen_indices].dropna())
for label, aa in [('Human proteins', aa_human), ('Malaria proteins', aa_malaria), ('Malaria antigens', aa_malaria_antigen)]:
    pcad_pathogen = pca.transform(aa)
    ax.scatter(pcad_pathogen[:, 0], pcad_pathogen[:, 1], label=label, s=(5 if label == 'Malaria antigens' else .5))

ax.legend()
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
lim = 0.25
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
fig.tight_layout()
fig.savefig('malaria_antigens.png')
plt.show()
