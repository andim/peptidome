import sys
sys.path.append('..')
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from lib import *
plt.style.use('../peptidome.mplstyle')

df = counter_to_df(count_kmers_proteome(human, 1), norm=True)

proteomes = load_proteomes()
proteomes = proteomes[~(proteomes.index == 'Human')]

def compare(path):
    dfp = counter_to_df(count_kmers_proteome(path, 1), norm=True)
    dfmerged = pd.merge(df, dfp, on='seq', suffixes=['_human', '_pathogen'])
    return dfmerged


dfmerged  = compare(datadir+'human-viruses-uniref50_nohiv.fasta')

fig, ax = plt.subplots(figsize=(2.25, 2.25))
xmin, xmax = 1/2**.5*np.amin(dfmerged['freq_human']), 2**.5*np.amax(dfmerged['freq_human'])
x = np.logspace(np.log10(xmin), np.log10(xmax))
linecolor = 'gray'
ax.plot(x, x, '-', lw=2, color=linecolor)
ax.plot(x, x*2, '--', lw=2, color=linecolor)
ax.plot(x, x/2, '--', lw=2, color=linecolor)
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
dfmerged.plot(x='freq_human', y='freq_pathogen', kind='scatter', logx=True, logy=True, ax=ax, s=25)
x, y = dfmerged['freq_human'], dfmerged['freq_pathogen'],
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
ax.text(0.05, 0.95, '$r^2={0:.2f}$'.format(r_value**2), transform=ax.transAxes, ha='left', va='top')
for index, row in dfmerged.iterrows():
    x, y, label = row['freq_human'], row['freq_pathogen'], row['seq']
    ax.annotate(label, # this is the text
                (x,y), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,-0.5), # distance from text to points (x,y)
                ha='center', va='center',
                fontsize='x-small')
ax.set_xlabel('Freq. Human Proteome')
ax.set_ylabel('Freq. Human Virome ')
fig.tight_layout()
fig.savefig(figuredir + 'aafreqsViruses.svg')


