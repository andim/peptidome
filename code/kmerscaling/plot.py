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

counters = [count_kmers_proteome(human, k, clean=True) for k in range(1, 10)]

totalpeptides = np.sum(list(counters[0].values()))
print('%e, %e'%(totalpeptides, 20**6))
fig, axes = plt.subplots(figsize=(3.42, 4.5), nrows=2)
ax = axes[0]
x = np.arange(1, 8)
ax.axhline(totalpeptides, color='k', label='number of amino acids')
ax.plot(x, 20**x, label='$20^x$')
ax.plot(range(1, len(counters)+1), [len(c) for c in counters], label='human proteome')
ax.set_yscale('log')
ax.set_xlabel('k')
ax.set_ylabel('# of distinct kmers')
ax.legend(loc='lower right')

ax = axes[1]
totalpeptides = np.sum(list(counters[0].values()))
print('%e, %e'%(totalpeptides, 20**6))
ks = np.arange(1, len(counters)+1)
ax.plot(ks, np.array([len(c) for c in counters])/20**ks, 'o', label='empirical')
ks = np.linspace(1, len(counters)+1)
N = totalpeptides
ax.plot(ks, 1-np.exp(-N/20**ks), label='$1-e^{-N/20^k}$')
ax.set_yscale('log')
ax.set_xlabel('k')
ax.set_ylabel('proportion of kmers in self')
ax.legend(loc='upper right')

fig.savefig('main.png')
plt.show()
