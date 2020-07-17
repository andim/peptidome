---
layout: post
title: PCA of amino acid frequencies
---

A comparison of proteins based on their amino acid frequencies. A minimal protein length of 200 amino acids is used and a pseudocount of 1 is added for regularization.

{% include post-image-gallery.html filter="aafreqpca/" %}

### Code 
#### pca.ipynb

```python
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition
plt.style.use('../peptidome.mplstyle')

from lib import *
```


```python
data = np.load('data/data.npz')

aa_cmv = data['cmv']
aa_malaria = data['malaria']
aa_human = data['human']

fig, ax = plt.subplots(figsize=(5, 5))
pca = sklearn.decomposition.PCA(n_components=2)
pca = pca.fit(np.vstack([aa_human, aa_malaria, aa_cmv]))
for label, aa in [('Human', aa_human), ('Malaria', aa_malaria), ('CMV', aa_cmv)]:
    pcad_pathogen = pca.transform(aa)
    ax.scatter(pcad_pathogen[:, 0], pcad_pathogen[:, 1], label=label+' proteins', s=.5)

freqss = []
pcomps = []
for filename in ['composition_human.tsv', 'composition_pfalciparum.tsv']:
    seqtk_columns = 'chr', 'length', '#A', '#C', '#G', "#T"
    composition = pd.read_csv('../aafreqs/data/%s'%filename, sep='\t',
                usecols=list(range(len(seqtk_columns))),
                names=seqtk_columns, index_col=0)
    pcomp = np.array([composition[seqtk_columns[i]].sum() for i in range(2, len(seqtk_columns))], dtype=np.float)
    pcomp /= np.sum(pcomp)
    pcomps.append(pcomp)
    frequencies = ntfreq_to_aafreq(pcomp)
    frequencies = [frequencies[aa] for aa in aminoacids]
    freqss.append(frequencies)
#frequencies = ntfreq_to_aafreq(np.ones(4)/4.0)
#frequencies = [frequencies[aa] for aa in aminoacids]
#freqss.append(frequencies)
freqss.append(np.ones(20)/20.0)
pcad = pca.transform(freqss)
ax.plot(pcad[0, 0], pcad[0, 1], 'd', markeredgecolor='w', ms=8, label=r'human genomic (%GC = '+'%g)'%round(np.sum(pcomps[0][1:3])*100))
ax.plot(pcad[1, 0], pcad[1, 1], 'd', markeredgecolor='w', ms=8, label=r'Malaria genomic (%GC = '+'%g)'%round(np.sum(pcomps[1][1:3])*100))
#ax.plot(pcad[2, 0], pcad[2, 1], 'd', markeredgecolor='w', ms=8, label='uniform genomic')
#ax.plot(pcad[2, 0], pcad[2, 1], 'kd', markeredgecolor='w', ms=8, label='uniform')

pgc = 0.42
pcomp = np.array([(1-pgc)/2, pgc/2, pgc/2, (1-pgc)/2])
freqs = lambda x: dict_to_array(ntfreq_to_aafreq(x))
delta = np.array([1,-1, -1, 1])
epsilon = 1e-3
deltaf = (freqs(pcomp+epsilon*delta)-freqs(pcomp))/epsilon
deltaf /= np.sum(deltaf**2)**.5
deltaf *= 0.1
deltaf_pcad = pca.transform(deltaf.reshape(1, -1))
plt.arrow(0, 0, deltaf_pcad[0, 0], deltaf_pcad[0, 1])

fpgc = lambda pgc: dict_to_array(ntfreq_to_aafreq(np.array([(1-pgc)/2, pgc/2, pgc/2, (1-pgc)/2])))
pgcs = np.linspace(0.2, 0.8, 10)
freqss = np.array([fpgc(pgc) for pgc in pgcs])
pcad = pca.transform(freqss)
ax.scatter(pcad[:, 0], pcad[:, 1], c=1-pgcs)


ax.legend()
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
lim = 0.25
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
fig.tight_layout()
```


![png](notebook_files/pca_1_0.png)



```python

```
#### run_malaria_antigen.py

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from lib import *

df_t = load_iedb_tcellepitopes(human_only=True)

iedbname = 'Plasmodium falciparum'
epitope_proteins = [s.split('/')[-1] for s in df_t[df_t['Epitope', 'Organism Name'] == iedbname]['Epitope', 'Parent Protein IRI'].unique() if type(s) == type('')]

epitope_proteins_indices = [i
                            for i, (h, seq) in enumerate(fasta_iter(proteome_path('Malaria')))
                            if iscontained(h, epitope_proteins)]
pd.Series(epitope_proteins_indices).to_csv('data/malaria_antigens.csv', index=False, header=False)

```
#### plot_malaria_antigen.py

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

```
#### run.py

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from lib import *

# Also stores indices of proteins within the proteome to allow their reidentification

min_length = 100

aas =  aminoacids
def aa_frequencies(proteome, min_length=1):
    seqs = []
    indices = []
    for i, (h, seq) in enumerate(fasta_iter(proteome)):
        for sym in 'XUBZ':
            seq = seq.replace(sym, '')
        if len(seq) < min_length:
            continue
        seqs.append(seq)
        indices.append(i)
    array = np.zeros((len(seqs), len(aas)))
    i = 0
    for seq in seqs:
        counter = {}
        for aa in aas:
            counter[aa] = 1
        count_kmers(seq, 1, counter=counter)
        sum_ = np.sum(list(counter.values()))
        for j, aa in enumerate(aas):
            array[i, j] = counter[aa]/sum_
        i += 1
    return array, indices

aa_human, indices_human = aa_frequencies(proteome_path('Human'), min_length=min_length)
aa_malaria, indices_malaria = aa_frequencies(proteome_path('Malaria'), min_length=min_length)
aa_cmv, indices_cmv = aa_frequencies(proteome_path('CMV'), min_length=min_length)
aa_viruses, indices_viruses = aa_frequencies(datadir+'human-viruses-uniref90.fasta', min_length=min_length)

np.savez('data/data.npz',
        human=aa_human, malaria=aa_malaria, cmv=aa_cmv, viruses=aa_viruses,
        indices_human=indices_human,
        indices_malaria=indices_malaria,
        indices_cmv=indices_cmv,
        indices_viruses=indices_viruses
        )

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
    ax.scatter(pcad_pathogen[:, 0], pcad_pathogen[:, 1], label=label+' proteins', s=.5)

freqss = []
pcomps = []
for filename in ['composition_human.tsv', 'composition_pfalciparum.tsv']:
    seqtk_columns = 'chr', 'length', '#A', '#C', '#G', "#T"
    composition = pd.read_csv('../aafreqs/data/%s'%filename, sep='\t',
                usecols=list(range(len(seqtk_columns))),
                names=seqtk_columns, index_col=0)
    pcomp = np.array([composition[seqtk_columns[i]].sum() for i in range(2, len(seqtk_columns))], dtype=np.float)
    pcomp /= np.sum(pcomp)
    pcomps.append(pcomp)
    frequencies = ntfreq_to_aafreq(pcomp)
    frequencies = [frequencies[aa] for aa in aminoacids]
    freqss.append(frequencies)
#frequencies = ntfreq_to_aafreq(np.ones(4)/4.0)
#frequencies = [frequencies[aa] for aa in aminoacids]
#freqss.append(frequencies)
freqss.append(np.ones(20)/20.0)
pcad = pca.transform(freqss)
ax.plot(pcad[0, 0], pcad[0, 1], 'd', markeredgecolor='w', ms=8, label=r'human genomic (%GC = '+'%g)'%round(np.sum(pcomps[0][1:3])*100))
ax.plot(pcad[1, 0], pcad[1, 1], 'd', markeredgecolor='w', ms=8, label=r'Malaria genomic (%GC = '+'%g)'%round(np.sum(pcomps[1][1:3])*100))
#ax.plot(pcad[2, 0], pcad[2, 1], 'd', markeredgecolor='w', ms=8, label='uniform genomic')
ax.plot(pcad[2, 0], pcad[2, 1], 'kd', markeredgecolor='w', ms=8, label='uniform')

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
#### plot-viruses.py

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

```
