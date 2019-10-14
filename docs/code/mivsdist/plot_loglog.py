import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

fig, ax = plt.subplots()
for species in ['Human', 'Yeast']:
    df = pd.read_csv('data/mutualinformation-%s.csv'%species)
    err = ax.errorbar(df['gaps']+1, df['mutualinformation'],
            2*df['mutualinformation_std'], fmt='_', label=species, ms=3)
    print(np.sum(df['mutualinformation']))
    ax.plot(df['gaps']+1, df['shuffledmutualinformation'], '_', ms=3,
            label=('shuffled') if species == 'Human' else '', c=err.lines[0].get_color())
d = np.linspace(1, 200)
#ax.plot(d, 0.005*np.exp(-d/50), zorder=10, label='exponential ($d_c=50$)')
l, = ax.plot(d, 0.02*d**-.5, zorder=10, c='k', label=r'power law ($-1/2$)')
ax.plot(d, 0.009*d**-.5, zorder=10, c=l.get_color())
ax.legend(loc='upper right', ncol=2)
ax.set_ylim(0.0004, 0.05)
ax.set_xlim(0.95, 201.0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Distance')
ax.set_ylabel('Mutual information in bits')
fig.tight_layout()
fig.savefig('loglog.png')
plt.show()
