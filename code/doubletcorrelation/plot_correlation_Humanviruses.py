import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

names = ['Human', 'Humanviruses']
y_name = 'Humanviruses'

fig, axes = plt.subplots(figsize=(6, 2), ncols=3, sharex=True, sharey=True)
lim = 1.05
for gap in range(3):
    i = gap

    dfs = {name: pd.read_csv('data/doubletfoldenrichment-%g-%s.csv'%(gap, name), index_col=0) for name in names}
    #offdiagonal = {name: np.asarray(dfs[name])[~np.eye(len(aminoacids),dtype=bool)] for name in dfs}
    flattened = {name: np.asarray(dfs[name]).flatten() for name in dfs}

    x = flattened['Human']
    y = flattened[y_name]
    sns.regplot(x, y, ax=axes[i], scatter_kws=dict(s=3))
    mask = ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask], y[mask])
    print(r_value**2, p_value, slope)
    axes[i].set_title('gap = {gap}'.format(gap=gap))
    axes[i].set_ylabel(y_name)
    axes[i].set_xlabel('Human')
    axes[i].text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
            va='top', ha='left', transform=axes[i].transAxes)
    axes[i].set_xlim(-lim, lim)
    axes[i].set_ylim(-lim, lim)
fig.savefig('logfoldenrichment_correlation_Humanviruses.png')
fig.savefig(figuredir+'logfoldenrichment_correlation_Humanviruses.svg')
