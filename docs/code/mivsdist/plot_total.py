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
    print(df['gaps'].max())
    ks = np.arange(1, df['gaps'].max())
    totalmi = [np.sum(2*df['mutualinformation'].iloc[:k]) for k in ks]
    totalmistd = np.array([2*np.sum(df['mutualinformation_std'].iloc[:k]**2)**.5 for k in ks])
    err = ax.errorbar(ks, totalmi, 2*totalmistd,  fmt='o', label=species, ms=1)
d = np.linspace(1, 200)
#ax.plot(d, 0.005*np.exp(-d/50), zorder=10, label='exponential ($d_c=50$)')
#l, = ax.plot(d, 0.02*d**-.5, zorder=10, c='k', label=r'power law ($-1/2$)')
#ax.plot(d, 0.009*d**-.5, zorder=10, c=l.get_color())
ax.legend(loc='upper right', ncol=2)
#ax.set_ylim(0.0004, 0.05)
#ax.set_xlim(0.95, 201.0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Distance')
ax.set_ylabel('Mutual information in bits')
fig.tight_layout()
fig.savefig('total.png')
plt.show()
