import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

df = pd.read_csv('data/entropy.csv')

fig, ax = plt.subplots()
for column in df.columns[1:4]:
    print(df[column])
    ax.plot(df['k'], df[column]/df['k'], 'o-', label=column)
ax.axhline(np.log2(20), c='k', label='random')
ax.legend(loc='lower left', ncol=2)
ax.set_xlabel('k')
ax.set_ylabel('entropy of kmers in bit / k');
fig.tight_layout()
fig.savefig('../../paper/images/entropykmer.pdf')
fig.savefig('main.png')
plt.show()
