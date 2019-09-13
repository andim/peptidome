import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

fig, ax = plt.subplots()
for species in ['Human', 'Mouse', 'Yeast']:
    df = pd.read_csv('data/mutualinformation-%s.csv'%species)
    ax.plot(df['gaps']+1, df['mutualinformation'], label=species)

#df = pd.read_csv('../pfam/data/mutualinformation.csv')
#ax.plot(df['gaps']+1, df['mutualinformation'], label='human filtered')
#mishuffled = 0.0012531812133396159
#plt.axhline(mishuffled, color='k', label='human shuffled')
ax.legend()
ax.set_ylim(0.0)
ax.set_xlim(1.0, 200.0)
ax.set_xscale('log')
ax.set_xlabel('Distance')
ax.set_ylabel('Mutual information in bits')
fig.tight_layout()
fig.savefig('mivsdist.pdf')
fig.savefig('main.png')
plt.show()
