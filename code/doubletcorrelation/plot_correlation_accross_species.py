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
lim = 1.2
for i, name in enumerate(names[1:]):
    x = flattened['Human']
    y = flattened[name]
    sns.regplot(x, y, ax=axes[i])
    mask = ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask], y[mask])
    print(r_value**2, p_value, slope)
    axes[i].set_ylabel(name)
    axes[i].set_xlabel('Human')
    axes[i].text(1.0, 1.0, '$R^2={0:.2f}$'.format(r_value**2), va='top', ha='right', transform=axes[i].transAxes)
    axes[i].set_xlim(-lim, lim)
    axes[i].set_ylim(-lim, lim)
fig.savefig('logfoldenrichment_correlation.png')
