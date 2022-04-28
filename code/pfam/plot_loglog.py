import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

fig, ax = plt.subplots(figsize=(1.75, 1.8))
df = pd.read_csv('data/mutualinformation_nozf.csv')
ax.plot(df['gaps']+1, df['mutualinformation'], 'o', ms=0.5, label='data')
#err = ax.errorbar(df['gaps']+1, df['mutualinformation'],
#        2*df['mutualinformation_std'], fmt='_', label=species, ms=3)
print(np.sum(df['mutualinformation']))
print(np.sum(2*df['mutualinformation'].iloc[:4]))
d = np.linspace(1, 200)
l, = ax.plot(d, 0.016*d**-.5, zorder=-1, c='k', label=r'$d^{-1/2}$')
ax.legend(loc='upper right', ncol=1)
ax.set_ylim(0.0004, 0.03)
ax.set_xlim(0.9, 201.0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Distance d')
ax.set_ylabel('Mutual information (bits)')
fig.tight_layout()
fig.savefig('loglog.png')
fig.savefig('mutualinformationdecay.pdf')
fig.savefig('../../figures/raw/mutualinformationdecay.svg')
plt.show()
