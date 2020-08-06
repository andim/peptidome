---
layout: post
title: Tissue distributions
---

Do the average protein features vary accross tissues?


{% include post-image-gallery.html filter="tissuedistributions/" %}

### Code 
#### tissuedistribution.ipynb

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from matplotlib import colors

import sys
sys.path.append('../')
from lib import *
from lib.maxent import *
```


```python
proteinatlas = pd.read_csv('../../data/proteinatlas.tsv.zip', sep='\t')
tissues = [c for c in proteinatlas.columns if 'Tissue' in c]
proteinatlas[tissues] = proteinatlas[tissues].div(proteinatlas[tissues].sum(axis=0), axis=1)
proteinatlas.fillna(0, inplace=True)
```


```python
bins = np.linspace(-7, -2, 100)
for tissue in tissues:
    plt.hist(np.log10(1e-7+proteinatlas[tissue]), histtype='step', bins=bins)
plt.yscale('log')
```


![png](notebook_files/tissuedistribution_2_0.png)



```python
L = 9
```


```python
humanproteome = load_proteome_as_df('Human')
humanproteome = humanproteome[~(humanproteome['Gene'] == '')]
humanproteome = humanproteome.set_index('Gene')
humanproteome = humanproteome.squeeze()
humanproteome = humanproteome[humanproteome['Sequence'].str.len()>L]
```


```python
missing = set(proteinatlas['Gene'].unique()) - set(humanproteome.index)
```


```python
proteinatlas = proteinatlas[~proteinatlas['Gene'].isin(missing)]
proteinatlas[tissues] = proteinatlas[tissues].div(proteinatlas[tissues].sum(axis=0), axis=1)
```


```python
proteinatlas_geneindex = proteinatlas.groupby('Gene').agg('sum')
sequences = np.array(humanproteome['Sequence'])
```


```python
def generator(N, k, sequences, p):
    N = int(N)
    counter = 0
    seqs = np.random.choice(sequences, p=expression, size=N)
    lengths = np.array([len(seq) for seq in seqs])
    startindices = np.random.randint(0, lengths-k)
    for sequence, startindex in zip(seqs, startindices):
        kmer = sequence[startindex:startindex+k]
        if isvalidaa(kmer):
            yield kmer
```


```python
observables_dict = {'fi':dict(), 'cij':dict(), 'cijk':dict()}
selected_tissues = np.random.choice(tissues, 5)
for tissue in selected_tissues:
    expression = np.array([proteinatlas_geneindex.loc[gene][tissue] if gene in proteinatlas_geneindex.index else 0.0
              for gene in humanproteome.index])
    expression /= np.sum(expression)
    matrix = kmers_to_matrix(generator(1e6, 9, sequences, expression))
    fi = frequencies(matrix, num_symbols=naminoacids)
    fij = pair_frequencies(matrix, num_symbols=naminoacids, fi=fi)
    cij = compute_covariance_matrix(fi, fij)
    fijk = triplet_frequencies(matrix, num_symbols=naminoacids)
    cijk = compute_cijk(fijk, fij, fi)
    observables_dict['fi'][tissue] = fi
    observables_dict['cij'][tissue] = cij
    observables_dict['cijk'][tissue] = cijk
```


```python
for dataset in ['train']:
    params = np.load('../maxent/data/%s_observables.npz'%dataset)
    for observable in observables_dict.keys():
        observables_dict[observable][dataset] = params[observable]
```


```python
fig, ax = plt.subplots(figsize=(4, 4))
fi = observables_dict['fi']['train']
xmin, xmax = 0.5*np.amin(fi), 2*np.amax(fi)
x = np.logspace(np.log10(xmin), np.log10(xmax))
ax.plot(x, x, 'k', lw=3)
ax.plot(x, x*2, '--k', lw=3)
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
ax.plot(x, x/2, '--k', lw=3)
for tissue in selected_tissues:
    ax.plot(np.mean(fi, axis=0), np.mean(observables_dict['fi'][tissue], axis=0), 'o', ms=2,
            label=tissue.split('-')[-1][:-5])
#ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('frequency unweighted')
ax.set_ylabel('frequency weighted')
fig.tight_layout()
fig.savefig('main.png')
```


![png](notebook_files/tissuedistribution_11_0.png)



```python
fig, axes = plt.subplots(figsize=(8, 5), ncols=3, nrows=2)

for j, (observable, label, lims, flattener) in enumerate([('fi', '$f_i$', (0, 0.12), np.ravel),
                                               ('cij', '$C_{ij}$', (-0.0025, 0.004), flatten_ij),
                                               ('cijk', '$C_{ijk}$', (-8e-4, 8e-4), flatten_ijk)]):
    for i, dataset in enumerate([selected_tissues[0], selected_tissues[1]]):
        ax = axes[i, j]
        if observable in ['cij', 'cijk']:
            plotting.density_scatter(flattener(observables_dict[observable]['train']),
                             flattener(observables_dict[observable][dataset]),
                             norm=colors.LogNorm(vmin=1),
                             s=0.5,
                             bins=40, ax=ax)
        else:
            ax.plot(flattener(observables_dict[observable]['train']),
                    flattener(observables_dict[observable][dataset]),
                    'o', ms=1)
        
        ax.set_xlabel('train %s'%label)
        ax.set_ylabel('%s %s'%(dataset[12:-5], label))
        ax.plot(lims, lims, 'k')
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)

for ax in axes[:, 1:].flatten():
    ax.ticklabel_format(style='sci', scilimits=(0,0))

fig.tight_layout()
fig.savefig('higher_order_stats.png')
```


![png](notebook_files/tissuedistribution_12_0.png)


# Weighted means


```python
prot_tidy = pd.read_csv('~/repos/mskpeptidome/data/proteinatlas_tidy.csv.gz', index_col=0)
prot_tidy = prot_tidy.loc[~prot_tidy.index.duplicated(keep='first')]
prot_tidy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean expression</th>
      <th>GMean expression</th>
      <th>Proteinlength</th>
      <th>no localization</th>
      <th>Nucleoplasm</th>
      <th>Cytosol</th>
      <th>Vesicles</th>
      <th>Plasma membrane</th>
      <th>Nucleoli</th>
      <th>Mitochondria</th>
      <th>...</th>
      <th>Nuclear membrane</th>
      <th>Microtubules</th>
      <th>Actin filaments</th>
      <th>Intermediate filaments</th>
      <th>Centriolar satellite</th>
      <th>Focal adhesion sites</th>
      <th>Cytokinetic bridge</th>
      <th>Cytoplasmic bodies</th>
      <th>Midbody</th>
      <th>Mitotic spindle</th>
    </tr>
    <tr>
      <th>Gene</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TSPAN6</th>
      <td>8.660000</td>
      <td>1.975018</td>
      <td>245</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>TNMD</th>
      <td>0.402187</td>
      <td>0.013358</td>
      <td>317</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>DPM1</th>
      <td>26.325625</td>
      <td>24.470388</td>
      <td>260</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>SCYL3</th>
      <td>7.055312</td>
      <td>6.590234</td>
      <td>742</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>C1orf112</th>
      <td>9.986563</td>
      <td>8.107200</td>
      <td>853</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
expression = np.array([prot_tidy.loc[gene]['Mean expression'] if gene in prot_tidy.index else 0.0
                       for gene in humanproteome.index], dtype=float)
```


```python
kmers, indices = zip(*to_kmers(sequences, 9, return_index=True))
```


```python
weights = expression[list(indices)]
```


```python
matrix = kmers_to_matrix(kmers)
```


```python
for name, w in [('unweighted', None), ('weighted', weights)]:
    fi = frequencies(matrix, num_symbols=naminoacids, weights=w)
    fij = pair_frequencies(matrix, num_symbols=naminoacids, fi=fi, weights=w)
    cij = compute_covariance_matrix(fi, fij)
    fijk = triplet_frequencies(matrix, num_symbols=naminoacids, weights=w)
    cijk = compute_cijk(fijk, fij, fi)
    observables_dict['fi'][name] = fi
    observables_dict['cij'][name] = cij
    observables_dict['cijk'][name] = cijk
```


```python
fig, ax = plt.subplots(figsize=(4, 4))
fi = observables_dict['fi']['unweighted'] 
xmin, xmax = 0.5*np.amin(fi), 2*np.amax(fi)
x = np.logspace(np.log10(xmin), np.log10(xmax))
ax.plot(x, x, 'k', lw=3)
ax.plot(x, x*2, '--k', lw=3)
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
ax.plot(x, x/2, '--k', lw=3)
ax.plot(np.mean(observables_dict['fi']['unweighted'] , axis=0),
        np.mean(observables_dict['fi']['weighted'] , axis=0),
        'o', ms=3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('frequency unweighted')
ax.set_ylabel('frequency weighted')
fig.tight_layout()
```


![png](notebook_files/tissuedistribution_20_0.png)



```python
fig, axes = plt.subplots(figsize=(8, 2.5), ncols=3, nrows=1)

for j, (observable, label, lims, flattener) in enumerate([('fi', '$f_i$', (0, 0.12), np.ravel),
                                               ('cij', '$C_{ij}$', (-0.0025, 0.004), flatten_ij),
                                               ('cijk', '$C_{ijk}$', (-8e-4, 8e-4), flatten_ijk)]):
        ax = axes[j]
        if observable in ['cij', 'cijk']:
            plotting.density_scatter(flattener(observables_dict[observable]['unweighted']),
                             flattener(observables_dict[observable]['weighted']),
                             norm=colors.LogNorm(vmin=1),
                             s=0.5,
                             bins=40, ax=ax)
        else:
            ax.plot(flattener(observables_dict[observable]['unweighted']),
                    flattener(observables_dict[observable]['weighted']),
                    'o', ms=1)
        
        ax.set_xlabel('unweighted %s'%label)
        ax.set_ylabel('weighted %s'%label)
        ax.plot(lims, lims, 'k')
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)

for ax in axes[1:].flatten():
    ax.ticklabel_format(style='sci', scilimits=(0,0))

fig.tight_layout()
```

    /home/amayer/.conda/envs/py3/lib/python3.6/site-packages/matplotlib/colors.py:1110: RuntimeWarning: invalid value encountered in less_equal
      mask |= resdat <= 0



![png](notebook_files/tissuedistribution_21_1.png)



```python

```
#### run.py

```python
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from lib import *

proteinatlas = pd.read_csv('../../data/proteinatlas.tsv.zip', sep='\t')
tissues = [c for c in proteinatlas.columns if 'Tissue' in c]
proteinatlas.fillna(0, inplace=True)

humanproteome = load_proteome_as_df('Human')
humanproteome = humanproteome[~(humanproteome['Gene'] == '')]
humanproteome = humanproteome.set_index('Gene')
humanproteome = humanproteome.squeeze()

missing = set(proteinatlas['Gene'].unique()) - set(humanproteome.index)

proteinatlas = proteinatlas[~proteinatlas['Gene'].isin(missing)]
proteinatlas[tissues] = proteinatlas[tissues].div(proteinatlas[tissues].sum(axis=0), axis=1)

def generator(N, k, p):
    counter = 0
    while counter < N:
        sequence = humanproteome.loc[np.random.choice(proteinatlas['Gene'], p=p)]
        if len(sequence)>k:
            startindex = np.random.randint(0, len(sequence)-k)
            counter += 1
            yield sequence[startindex:startindex+k]

freqs = {}
df = Counter(proteome_path('Human'), 1).to_df()
df.set_index('seq', inplace=True)
df = df.squeeze()
freqs['All'] = df
for tissue in tissues:
    df = Counter(generator(5000, 9, np.array(proteinatlas[tissue])), 1).to_df()
    df.set_index('seq', inplace=True)
    df = df.squeeze()
    freqs[tissue] = df
pd.DataFrame(freqs).to_csv('freqs.csv', index=True)

```
#### plot.py

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import sys
sys.path.append('../')
from lib import *
plt.style.use('../peptidome.mplstyle')

freqs = pd.read_csv('freqs.csv', index_col=0)
tissues = [c for c in freqs.columns if 'Tissue' in c]

fig, ax = plt.subplots(figsize=(4, 4))
xmin, xmax = 0.5*np.amin(freqs['All']), 2*np.amax(freqs['All'])
x = np.logspace(np.log10(xmin), np.log10(xmax))
ax.plot(x, x, 'k', lw=3)
ax.plot(x, x*2, '--k', lw=2)
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
ax.plot(x, x/2, '--k', lw=2)
for tissue in np.random.choice(tissues, 5, replace=False):
    ax.plot(freqs['All'], freqs[tissue], 'o', label=tissue.split('-')[-1][:-5])
ax.legend(title='Tissue')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency unweighted')
ax.set_ylabel('Frequency weighted by expression')
fig.tight_layout()
fig.savefig('main.png')
plt.show()


```
