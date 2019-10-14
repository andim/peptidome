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

