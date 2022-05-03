import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

df = pd.read_csv('data/entropy.csv')
df = df[df['k']<5]

fig, ax = plt.subplots(figsize=(1.75, 2.25))
#column='Viruses'
#ax.plot(df['k'], df[column]/df['k'], 'o-', label='Human Viruses', color='C3', zorder=1)
column='Human'
ax.plot(df['k'], df[column]/df['k'], 'o-', label='Human Proteome', color='C0', zorder=1)
ax.axhline(np.log2(20), label='Uniform', color='C1', ls='-', zorder=0)
ax.axhline(df[column].loc[0], label='Independent', color='C2', ls='-', zorder=0)
ax.legend(loc='center right', ncol=1)
ax.set_ylim(4.12, 4.34)
ax.set_xticks(np.arange(1, 5))
ax.set_xlabel('k')
ax.set_ylabel('kmer Entropy in bit / k');
fig.tight_layout()
fig.savefig('../../paper/images/entropykmer.pdf')
fig.savefig(figuredir + 'entropykmer.svg')
fig.savefig('main.png')
plt.show()
