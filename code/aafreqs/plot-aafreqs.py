import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

df = Counter(human, 1).to_df(norm=True, clean=True)
df = df.set_index('seq')
df = df.sort_values(by='freq', ascending=False)

seqtk_columns = 'chr', 'length', '#A', '#C', '#G', "#T"
composition = pd.read_csv('data/composition.tsv', sep='\t',
            usecols=list(range(len(seqtk_columns))),
            names=seqtk_columns, index_col=0)
pcomp = np.array([composition[seqtk_columns[i]].sum() for i in range(2, len(seqtk_columns))], dtype=np.float)
pcomp /= np.sum(pcomp)
print(pcomp)

dfm = df.copy(deep=True)
fig, ax = plt.subplots()
for name, p in [('uniform', np.ones(4)), ('background', pcomp)]:
    p /= np.sum(p)
    frequencies = ntfreq_to_aafreq(p)
    df_theory = pd.DataFrame.from_dict(frequencies, orient='index', columns=['nt '+name])
    dfm = dfm.merge(df_theory, left_index=True, right_index=True)
dfm.sort_values(by='freq', inplace=True, ascending=False)
dfm.plot(kind='bar', ax=ax)
ax.set_xlabel('Amino acid')
ax.set_ylabel('Frequency')
fig.tight_layout()
fig.savefig('main.png')
plt.show()
