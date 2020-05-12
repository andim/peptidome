---
layout: post
title: Correlations between amino acid doublets
---

An analysis of fold enrichment of amino acid pairs relative to independence.

{% include post-image-gallery.html filter="doubletcorrelation/" %}

### Code 
#### plot_correlation_accross_species.py

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

names = ['Human', 'Mouse', 'Yeast', 'Malaria', 'Vaccinia']
gap = 1
dfs = {name: pd.read_csv('data/doubletfoldenrichment-%g-%s.csv'%(gap, name), index_col=0) for name in names}

#offdiagonal = {name: np.asarray(dfs[name])[~np.eye(len(aminoacids),dtype=bool)] for name in dfs}
flattened = {name: np.asarray(dfs[name]).flatten() for name in dfs}

fig, axes = plt.subplots(figsize=(8, 2), ncols=4, sharex=True, sharey=True)
lim = 1.3
for i, name in enumerate(names[1:]):
    x = flattened['Human']
    y = flattened[name]
    sns.regplot(x, y, ax=axes[i], scatter_kws=dict(s=3))
    mask = ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask], y[mask])
    print(r_value**2, p_value, slope)
    axes[i].set_ylabel(name)
    axes[i].set_xlabel('Human')
    axes[i].text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
            va='top', ha='left', transform=axes[i].transAxes)
    axes[i].set_xlim(-lim, lim)
    axes[i].set_ylim(-lim, lim)
fig.savefig('logfoldenrichment_correlation.png')

```
#### run.py

```python
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

name = sys.argv[1]
proteome = proteome_path(name)

df1 = Counter(proteome, k=1).to_df(norm=True, clean=True)
df1.set_index('seq', inplace=True)

gaps = np.arange(0, 4, 1)
for gap in gaps:
    df2 = counter_to_df(count_kmers_proteome(proteome, k=2, gap=gap), norm=True, clean=True)
    strcolumn_to_charcolumns(df2, 'seq')
    df2['theory'] = [float(df1.loc[s[0]] * df1.loc[s[1]]) for s in df2['seq']]
    df2['fold'] = np.log2(df2['freq']/df2['theory'])
    print(np.abs(df2['fold']).max(), np.abs(df2['fold']).mean())
    dfmat = df2.pivot(columns='aa1', index='aa2')['fold']
    print(dfmat)
    dfmat.to_csv('data/doubletfoldenrichment-%g-%s.csv'%(gap, name))

```
#### plot.py

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

name = sys.argv[1]
dfs = {gap: pd.read_csv('data/doubletfoldenrichment-%g-%s.csv'%(gap, name), index_col=0) for gap in range(0, 4)}
print(dfs[0])

fig, axes = plt.subplots(figsize=(10.0, 9), nrows=2, ncols=2,
        #gridspec_kw=dict(right=0.9),
        sharex=True, sharey=True)
for gap, ax in zip(list(dfs), axes.flatten()):
    ax.set_title('distance '+str(gap+1))
    im = sns.heatmap(dfs[gap], ax=ax, cbar=False,
            vmin=-1.0, vmax=1.0, cmap='RdBu_r')
            #cbar_kws=dict(label='log$_2$ doublet fold enrichment'))
print(im)
cax = fig.add_axes((0.91, 0.3, 0.01, 0.4))
fig.colorbar(im.collections[0], cax=cax, label='log$_2$ doublet fold enrichment')
for ax in axes[1, :]:
    ax.set_xlabel('amino acid 1')
for ax in axes[:, 0]:
    ax.set_ylabel('amino acid 2')
for ax in axes[:, 1]:
    ax.set_ylabel('')
fig.tight_layout(rect=[0, 0, 0.9, 1.0])
fig.savefig('main.png' if name == 'Human' else name+'.png')

plt.show()

```
