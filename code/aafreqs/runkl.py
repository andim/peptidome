import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from lib import *

proteomes = load_proteomes()
human = datadir + proteomes.ix['Human']['path']


df = counter_to_df(count_kmers_proteome(human, 1), norm=True)
df2 = counter_to_df(count_kmers_proteome(human, 2), norm=True)

pathogenproteomes = proteomes[~proteomes.index.isin(['Human'])]

round_to_n = lambda x, n: round(x, -int(np.floor(np.log10(x))) + (n - 1))

def dkl(path, name): 
    dfp = counter_to_df(count_kmers_proteome(path, 1), norm=True)
    dfmerged = pd.merge(df, dfp, on='seq', suffixes=['_human', '_pathogen'])

    h = np.asarray(dfmerged['freq_human'])
    p = np.asarray(dfmerged['freq_pathogen'])

    dkl = scipy.stats.entropy(p, qk=h, base=2)

    # calculate dkl between twomers
    dfp = counter_to_df(count_kmers_proteome(path, 2), norm=True)
    dfmerged = pd.merge(df2, dfp, on='seq', suffixes=['_human', '_pathogen'])

    h = np.asarray(dfmerged['freq_human'])
    p = np.asarray(dfmerged['freq_pathogen'])

    dkl2 = scipy.stats.entropy(p, qk=h, base=2)

    print('%s&%s&%s\\\\'%(name, round_to_n(dkl, 2), round_to_n(dkl2, 2)))


dkl(datadir+'human-viruses-uniref90_nohiv.fasta', 'viruses')

for name, row in pathogenproteomes.iterrows():
    path = datadir + row['path']
    dkl(path, name)


