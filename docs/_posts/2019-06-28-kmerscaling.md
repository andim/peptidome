---
layout: post
title: Scaling of the number of kmers
---

How many distinct kmers are their in the human proteome?

{% include post-image-gallery.html filter="kmerscaling/" %}

### Code 
#### Untitled.ipynb


What is the mutual information between an amino acid and the protein it comes from?


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')
from lib import *
```


```python
data = np.load('../aafreqpca/data/data.npz')
aa_human = data['human']
aa_human = aa_human#[np.random.randint(0, 10000, 5000), :]
```


```python
np.sum(aa_human * np.log2(aa_human/aa_human.mean(axis=0)))/aa_human.shape[0]
```




    0.09741991137385587




```python
meanfreqs.shape
```




    (20, 20)




```python
meanfreq = aa_human.mean(axis=0)
meanfreqs = np.einsum('i,j->ij', meanfreq, meanfreq)
assert meanfreqs[0, 0] == meanfreq[0]**2
```


```python
devs = np.zeros(aa_human.shape[0])
freqss = np.zeros_like(meanfreqs)
for row in range(aa_human.shape[0]):
    freq = aa_human[row]
    freqs = np.einsum('i,j->ij', freq, freq)
    devs[row] = np.sum(freqs * np.log2(freqs/meanfreqs))
    freqss += freqs
freqss /= aa_human.shape[0]
#np.sum(devs)/aa_human.shape[0], np.median(devs)
```

$\sum_i \sum_j < f_i^p f_j^p> \log_2 \frac{< f_i^p f_j^p>}{ <f_i^p> <f_j^p>}$


```python
freqss = np.einsum('ki,kj->ij', aa_human, aa_human)/aa_human.shape[0]
np.sum(freqss * np.log2(freqss/meanfreqs))
```




    0.0014361921190761484




```python
plt.hist(devs, bins=np.linspace(0.0, 1.0, 100));
```


![png](notebook_files/Untitled_9_0.png)



```python
sns.heatmap(np.log2(aa_human/aa_human.mean(axis=0)), vmin=-3, vmax=+3, cmap='PRGn')
#plt.colorbar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f400adf04e0>




![png](notebook_files/Untitled_10_1.png)


$I(X, Z) = H(X) - H(X|Z)$

$I(X_1, X_2) = H(X_2) - H(X_2 | X_1)$
#### plot.py

```python
import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.stats
import sklearn.decomposition
import sklearn.manifold
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

ks = np.arange(1, 10)
counters = [count_kmers_proteome(human, k, clean=True) for k in ks]

totalpeptides = np.sum(list(counters[0].values()))
print('%e, %e'%(totalpeptides, 20**6))

def fraction_overlapping(counter):
    counts = np.array(counter_to_df(counter, norm=False, clean=False)['count'])
    return np.sum(counts[counts>1])/np.sum(counts)
fractions = [fraction_overlapping(c) for c in counters]

fig, axes = plt.subplots(figsize=(3.42, 6.0), nrows=3, sharex=True)
ax = axes[0]
ax.plot(range(1, len(counters)+1), [len(c) for c in counters], 'o', label='human proteome')
ax.axhline(totalpeptides, color='k', label='number of amino acids')
ax.plot(ks, 20**ks, label='$20^x$')
ax.set_yscale('log')
ax.set_ylim(1e1, 10*totalpeptides)
ax.set_ylabel('# of distinct kmers')
ax.legend(loc='lower right')

n = totalpeptides
p0 = 1/20**ks

ax = axes[1]
ax.plot(ks, np.array([len(c) for c in counters])/20**ks, 'o', label='empirical')
ax.plot(ks, 1-np.exp(-n*p0), label='$1-e^{-N p_0}$')
ax.set_yscale('log')
ax.set_ylabel('fraction of all kmers seen')
ax.legend(loc='upper right')



ax = axes[2]
ax.plot(ks, fractions, 'o')
ax.plot(ks, 1 - p0*n/(np.exp(p0*n)-1), label=r'$1-\frac{N p_0}{e^{N p_0}-1}$')
ax.legend()
ax.set_ylabel('fraction of kmers seen\nmore than once')
ax.set_ylim(0.0)
ax.set_xlabel('k')

fig.tight_layout()


fig.savefig('main.png')
plt.show()

```
