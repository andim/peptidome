import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib import *

df = counter_to_df(count_kmers_proteome(human, 1), norm=True)

dfproteomes = pd.read_csv(datadir+'proteomes.csv', sep=',')
pathogenproteomes = dfproteomes[~(dfproteomes['shortname']=='Human')]

for idx, row in pathogenproteomes.iterrows():
    name = row['shortname']
    path = row['path']

    dfp = counter_to_df(count_kmers_proteome(datadir+path, 1), norm=True)
    dfmerged = pd.merge(df, dfp, on='seq', suffixes=['_human', '_pathogen'])

    print(name, dfmerged)
    fig, ax = plt.subplots(figsize=(4, 4))
    xmin, xmax = 0.25*np.amin(dfmerged['freq_human']), 2*np.amax(dfmerged['freq_human'])
    x = np.logspace(np.log10(xmin), np.log10(xmax))
    ax.plot(x, x, 'k', lw=3)
    ax.plot(x, x*2, '--k', lw=3)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.plot(x, x/2, '--k', lw=3)
    dfmerged.plot(x='freq_human', y='freq_pathogen', kind='scatter', logx=True, logy=True, ax=ax);
    ax.set_xlabel('frequency human')
    ax.set_ylabel('frequency ' + name)
    fig.tight_layout()
    fig.savefig('plots/aafreqs%s.png' % name, dpi=300)
