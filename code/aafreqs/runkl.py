import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from lib import *

df = counter_to_df(count_kmers_proteome(human, 1), norm=True)
df2 = counter_to_df(count_kmers_proteome(human, 2), norm=True)

dfproteomes = pd.read_csv(datadir+'proteomes.csv', sep=',')
pathogenproteomes = dfproteomes[~dfproteomes['shortname'].isin(['Human'])]

round_to_n = lambda x, n: round(x, -int(np.floor(np.log10(x))) + (n - 1))

for idx, row in pathogenproteomes.iterrows():
    name = row['shortname']
    path = row['path']

    dfp = counter_to_df(count_kmers_proteome(datadir+path, 1), norm=True)
    dfmerged = pd.merge(df, dfp, on='seq', suffixes=['_human', '_pathogen'])

    h = np.asarray(dfmerged['freq_human'])
    p = np.asarray(dfmerged['freq_pathogen'])

    dkl = scipy.stats.entropy(p, qk=h, base=2)

    # calculate dkl between twomers
    dfp = counter_to_df(count_kmers_proteome(datadir+path, 2), norm=True)
    dfmerged = pd.merge(df2, dfp, on='seq', suffixes=['_human', '_pathogen'])

    h = np.asarray(dfmerged['freq_human'])
    p = np.asarray(dfmerged['freq_pathogen'])

    dkl2 = scipy.stats.entropy(p, qk=h, base=2)

    print('%s&%s&%s\\\\'%(name, round_to_n(dkl, 2), round_to_n(dkl2, 2)))

