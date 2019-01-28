import sys
sys.path.append('..')

import itertools, copy
import numpy as np
import scipy.misc
import pandas as pd
import seaborn as sns

from lib import *

df0 = counter_to_df(count_kmers_proteome(human, 1))
df0 = df0[~df0['seq'].str.contains('U|B|X|Z')]
df0 = df0.set_index('seq')

df1 = counter_to_df(count_kmers_proteome(human, 2, gap=0))
df1 = df1[~df1['seq'].str.contains('U|B|X|Z')]
df1 = df1.set_index('seq')

dfgap1 = counter_to_df(count_kmers_proteome(human, 2, gap=1))
dfgap1 = dfgap1[~dfgap1['seq'].str.contains('U|B|X|Z')]
dfgap1 = dfgap1.set_index('seq')

dfgap2 = counter_to_df(count_kmers_proteome(human, 2, gap=2))
dfgap2 = dfgap2[~dfgap2['seq'].str.contains('U|B|X|Z')]
dfgap2 = dfgap2.set_index('seq')

def fit_ising(f1, f2, niter=1, nmcmc=1e6, epsilon=0.1, Jk=None):
    h = f1
    q = len(aminoacids)
    N = 6
    if Jk is None:
        Jk = np.array((N-1, q, q))
    for i in range(niter):
        def jump(x):
            return np.random.randint(q, size=N)
        def energy(x):
            clib.energy(x, h, Jk)
        x0 = jump(None)
        samples = mcmcsampler(x0, energy, jump, nmcmc)
        for gap in range(len(f2s)):
            m = f2s[gap].merge(counter_to_df(count_kmers_iterable(samples, 2, gap=gap)), left_index=True, right_on='seq')
            m['logfold'] = np.log(m['freq_x']/m['freq_y'])
            print(i, gap, np.mean(np.abs(m['logfold'])))
            for idx, row in m.iterrows():
                logfold = row['logfold']
                aa1 = row['seq'][0]
                aa2 = row['seq'][1]
                Jk[gap][aa1][aa2] += logfold * epsilon
    return h, Jk

print('start fitting')
h, Jk = fit_ising(np.array(np.log(df0)), [df1, dfgap1, dfgap2], nmcmc=1e6, niter=100, epsilon=0.2)

dfh = pd.DataFrame(index=[key for key in h],
                   data=[h[key] for key in h],
                   columns=['h'])
doublets = [''.join(s) for s in itertools.product(list(aminoacids), repeat=2)]
dfJk = pd.DataFrame(index=doublets,
                    data=[Jk[0][s[0]][s[1]] for s in doublets],
                    columns=['J0'])
for i in range(1, len(Jk)):
    dfJk['J%g'%i] = [Jk[i][s[0]][s[1]] for s in doublets]

dfh.to_csv('h.csv')
dfJk.to_csv('Jk.csv')
