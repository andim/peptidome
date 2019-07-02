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
