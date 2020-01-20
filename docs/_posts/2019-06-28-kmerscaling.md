---
layout: post
title: Scaling of the number of kmers
---

How many distinct kmers are their in the human proteome?

{% include post-image-gallery.html filter="kmerscaling/" %}

### Code 
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
