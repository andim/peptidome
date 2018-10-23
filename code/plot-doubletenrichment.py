import numpy as np
import pandas as pd
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt

from lib import *

name = 'human'
proteome = human

df1 = counter_to_df(count_kmers_proteome(proteome, k=1))
df1 = df1[~df1['seq'].str.contains('U|B|X|Z')]
df1.set_index('seq', inplace=True)
entropy1 = entropy(df1['freq'], base=2)

meanabsfoldchanges = []
mutualinformations = []
gaps = np.arange(0, 5, 1)
for gap in gaps:
    df2 = counter_to_df(count_kmers_proteome(proteome, k=2, gap=gap))
    df2 = df2[~df2['seq'].str.contains('U|B|X|Z')]
    entropy2 = entropy(df2['freq'], base=2)
    mi = 2*entropy1 - entropy2
    print(gap, mi)
    mutualinformations.append(mi)
    strcolumn_to_charcolumns(df2, 'seq')
    df2['theory'] = [float(df1.loc[s[0]] * df1.loc[s[1]]) for s in df2['seq']]
    df2['fold'] = np.log2(df2['freq']/df2['theory'])
    print(np.abs(df2['fold']).max(), np.abs(df2['fold']).mean(), mi)
    meanabsfoldchanges.append(np.abs(df2['fold']).mean())
    dfmat = df2.pivot(columns='aa1', index='aa2')['fold']
    print(dfmat)
    dfmat.to_csv('doubletfoldenrichment-%g-%s.csv'%(gap, name))
#    fig = plt.figure(figsize=(5.5, 5))
#    sns.heatmap(dfmat, vmin=-1.0, vmax=1.0, cmap='RdBu_r', cbar_kws=dict(label='log$_2$ doublet fold enrichment'))
#    fig.tight_layout()
#    fig.savefig('plots/doublet-%s-gap%i.png'%(name, gap), dpi=300)
#    plt.close(fig)

#df2 = counter_to_df(count_kmers_iterable(scrambled(fasta_iter(human, returnheader=False)), k=2, gap=0))
#df2 = df2[~df2['seq'].str.contains('U|B|X|Z')]
#print(df2['freq'], df2['freq'].sum())
#mishuffled = 2*entropy1-entropy(df2['freq'], base=2)
#print(mishuffled)

#fig = plt.figure(figsize=(5, 3))
#plt.plot(gaps, meanabsfoldchanges)
#plt.ylim(0.0)
#plt.xlabel('Gap')
#plt.ylabel('Mean absolute fold change')
#fig.tight_layout()
#fig.savefig('plots/doublet-%s-meangap.png'%name, dpi=300)

df = pd.DataFrame.from_dict(dict(gaps=gaps, mutualinformation=mutualinformations))
df.to_csv('mutualinformation-%s.csv'%name)
fig = plt.figure(figsize=(5, 3))
plt.plot(gaps, mutualinformations)
#plt.axhline(mishuffled)
plt.ylim(0.0)
plt.xlabel('Gap')
plt.ylabel('Mutual information in bits')
fig.tight_layout()
fig.savefig('plots/doublet-%s-mutualinformation.png'%name, dpi=300)
