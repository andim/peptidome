import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

df = pd.read_csv('data/entropy.csv')
df = df[df['k']<5]

fig, ax = plt.subplots(figsize=(1.75, 2.5))
column='Human'
ax.plot(df['k'], df[column]/df['k'], 'o-', label='empirical', color='C0')
ax.axhline(np.log2(20), label='uniform', color='C1', ls='--')
ax.axhline(df[column].loc[0], label='independent', color='C2', ls='--')
ax.legend(loc='upper right', ncol=1, bbox_to_anchor=(1.05, 1.05))
ax.set_ylim(4.1, 4.4)
ax.set_xlabel('k')
ax.set_ylabel('entropy of kmers in bit / k');
fig.tight_layout()
fig.savefig('../../paper/images/entropykmer.pdf')
fig.savefig('../../figures/raw//entropykmer.svg')
fig.savefig('main.png')
plt.show()
