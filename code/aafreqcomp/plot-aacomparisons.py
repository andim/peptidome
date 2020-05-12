import sys
sys.path.append('..')
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from lib import *

df = counter_to_df(count_kmers_proteome(human, 1), norm=True)

proteomes = load_proteomes()
proteomes = proteomes[~(proteomes.index == 'Human')]

def compare(path, name):
    dfp = counter_to_df(count_kmers_proteome(path, 1), norm=True)
    dfmerged = pd.merge(df, dfp, on='seq', suffixes=['_human', '_pathogen'])

    print(name)
    fig, ax = plt.subplots(figsize=(4, 4))
    xmin, xmax = 0.5*np.amin(dfmerged['freq_human']), 2*np.amax(dfmerged['freq_human'])
    x = np.logspace(np.log10(xmin), np.log10(xmax))
    linecolor = 'gray'
    ax.plot(x, x, '-', lw=3, color=linecolor)
    ax.plot(x, x*2, '--', lw=3, color=linecolor)
    ax.plot(x, x/2, '--', lw=3, color=linecolor)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    dfmerged.plot(x='freq_human', y='freq_pathogen', kind='scatter', logx=True, logy=True, ax=ax);
    print(dfmerged)
    x, y = dfmerged['freq_human'], dfmerged['freq_pathogen'],
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    ax.set_title('$r^2={0:.2f}$'.format(r_value**2))
    for index, row in dfmerged.iterrows():
        x, y, label = row['freq_human'], row['freq_pathogen'], row['seq']
        ax.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,0), # distance from text to points (x,y)
                    ha='center', va='center')
    ax.set_xlabel('frequency human')
    ax.set_ylabel('frequency ' + name)
    fig.tight_layout()
    fig.savefig('aafreqs%s.png' % name, dpi=300)

compare(datadir+'human-viruses-uniref90.fasta', 'Viruses')
compare(datadir+'ufos/ufo.fasta', 'Ufo')
compare(datadir+'ufos/ext.fasta', 'Ufo-ext')

for name, row in proteomes.iterrows():
    path = datadir+row['path']

    compare(path, name)

shutil.move('aafreqsViruses.png', 'main.png')
