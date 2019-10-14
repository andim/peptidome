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

import sys
sys.path.append('../')
from lib import *
plt.style.use('../peptidome.mplstyle')
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
humanproteome = load_proteome_as_df('Human')
humanproteome = humanproteome[~(humanproteome['Gene'] == '')]
humanproteome = humanproteome.set_index('Gene')
humanproteome = humanproteome.squeeze()
```


```python
missing = set(proteinatlas['Gene'].unique()) - set(humanproteome.index)
```


```python
proteinatlas = proteinatlas[~proteinatlas['Gene'].isin(missing)]
proteinatlas[tissues] = proteinatlas[tissues].div(proteinatlas[tissues].sum(axis=0), axis=1)
```


```python
%timeit -t humanproteome.loc[np.random.choice(proteinatlas['Gene'], p=proteinatlas[tissues[3]])]
```

    505 µs ± 1.55 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)



```python
def generator(N, k, p):
    counter = 0
    while counter < N:
        sequence = humanproteome.loc[np.random.choice(proteinatlas['Gene'], p=p)]
        if len(sequence)>k:
            startindex = np.random.randint(0, len(sequence)-k)
            counter += 1
            yield sequence[startindex:startindex+k]
```


```python
freqs = {}
for tissue in tissues:
    df = Counter(generator(1000, 9, np.array(proteinatlas[tissue])), 1).to_df()
    df.set_index('seq', inplace=True)
    df = df.squeeze()
    freqs[tissue] = df
```


```python
df = Counter(proteome_path('Human'), 1).to_df()
df.set_index('seq', inplace=True)
df = df.squeeze()
freqs['All'] = df
```


```python
fig, ax = plt.subplots(figsize=(4, 4))
xmin, xmax = 0.5*np.amin(freqall), 2*np.amax(freqall)
x = np.logspace(np.log10(xmin), np.log10(xmax))
ax.plot(x, x, 'k', lw=3)
ax.plot(x, x*2, '--k', lw=3)
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
ax.plot(x, x/2, '--k', lw=3)
for tissue in np.random.choice(tissues, 5):
    ax.plot(freqs['All'], freqs[tissue], 'o', ms=1, label=tissue.split('-')[-1][:-5])
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('frequency human')
ax.set_ylabel('frequency ')
fig.tight_layout()
fig.savefig('main.png')

```


![png](notebook_files/tissuedistribution_10_0.png)



```python
pd.DataFrame(freqs).to_csv('freqs.csv', index=True)
```


```python
np.random.choice?
```


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
