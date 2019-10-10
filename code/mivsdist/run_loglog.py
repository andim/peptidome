import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

species = 'Human'
fig, ax = plt.subplots()
df = pd.read_csv('data/mutualinformation-%s.csv'%species)
ax.errorbar(df['gaps']+1, df['mutualinformation'],
        2*df['mutualinformation_std'], label='data')
ax.plot(df['gaps']+1, df['shuffledmutualinformation'], label='shuffled')
d = np.linspace(1, 200)
ax.plot(d, 0.005*np.exp(-d/50), zorder=10, label='exponential ($d_c=50$)')
ax.plot(d, 0.025*d**-.5, zorder=10, label=r'power law ($\alpha -1/2$)')
ax.legend(loc='upper right')
ax.set_ylim(0.001, 0.05)
ax.set_xlim(0.95, 201.0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Distance')
ax.set_ylabel('Mutual information in bits')
fig.tight_layout()
fig.savefig('Human_loglog.png')
plt.show()
