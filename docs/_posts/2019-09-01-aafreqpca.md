---
layout: post
title: PCA of amino acid frequencies
---

A comparison of proteins based on their amino acid frequencies. A minimal protein length of 100 amino acids is used and a pseudocount of 1 is added for regularization.

{% include post-image-gallery.html filter="aafreqpca/" %}

### Code 
#### run.py

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from lib import *

min_length = 100

aas =  aminoacids

def aa_frequencies(proteome, min_length=1):
    proteome = proteome_path(proteome)
    n = sum([1 for h, seq in fasta_iter(proteome) if len(seq)>=min_length])
    array = np.zeros((n, len(aas)))
    i = 0
    for h, seq in fasta_iter(proteome):
        seq = seq.replace('X', '')
        seq = seq.replace('U', '')
        if len(seq) < min_length:
            continue
        counter = {}
        for aa in aas:
            counter[aa] = 1
        count_kmers(seq, 1, counter=counter)
        sum_ = np.sum(list(counter.values()))
        for j, aa in enumerate(aas):
            array[i, j] = counter[aa]/sum_
        i += 1
    return array


aa_human = aa_frequencies('Human', min_length=min_length)
aa_malaria = aa_frequencies('Malaria', min_length=min_length)
aa_cmv = aa_frequencies('CMV', min_length=min_length)

np.savez('data/data.npz', human=aa_human, malaria=aa_malaria, cmv=aa_cmv)

```
#### plot.py

```python
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

```
