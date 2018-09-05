import numpy as np
import pandas as pd
from scipy.stats import entropy
import sklearn.decomposition
import sklearn.manifold
import seaborn as sns
import matplotlib.pyplot as plt

from lib import *

name = 'mouse'
proteome = mouse

df1 = counter_to_df(count_kmers_proteome(proteome, k=1))
df1.set_index('seq', inplace=True)
meanabsfoldchanges = []
gaps = np.arange(0, 41, 1)
for gap in gaps:
    df2 = counter_to_df(count_kmers_proteome(proteome, k=2, gap=gap))
    df2 = df2[~df2['seq'].str.contains('U|B|X|Z')]
    strcolumn_to_charcolumns(df2, 'seq')
    df2['theory'] = [float(df1.loc[s[0]] * df1.loc[s[1]]) for s in df2['seq']]
    df2['fold'] = np.log2(df2['freq']/df2['theory'])
    print(np.abs(df2['fold']).max(), np.abs(df2['fold']).mean())
    meanabsfoldchanges.append(np.abs(df2['fold']).mean())
    dfmat = df2.pivot(columns='aa1', index='aa2')['fold']
    fig = plt.figure(figsize=(5.5, 5))
    sns.heatmap(dfmat, vmin=-1.0, vmax=1.0, cmap='RdBu_r', cbar_kws=dict(label='log$_2$ doublet fold enrichment'))
    fig.tight_layout()
    fig.savefig('plots/doublet-%s-gap%i.png'%(name, gap), dpi=300)


fig = plt.figure(figsize=(5, 3))
plt.plot(gaps, meanabsfoldchanges)
plt.ylim(0.0)
plt.xlabel('Gap')
plt.ylabel('Mean absolute fold change')
fig.tight_layout()
fig.savefig('plots/doublet-meangaps.png', dpi=300)
