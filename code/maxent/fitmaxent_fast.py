import sys
sys.path.append('..')

import itertools, copy
import numpy as np
import scipy.misc
import pandas as pd
import seaborn as sns

from lib import *

import clib

aas_arr = np.array(list(aminoacids))
q = len(aminoacids)
N = 6

df0 = counter_to_df(count_kmers_proteome(human, 1))
df0 = df0.set_index('seq')
df0 = df0.sort_index()

df1 = counter_to_df(count_kmers_proteome(human, 2, gap=0))
df1 = df1.set_index('seq')

dfgap1 = counter_to_df(count_kmers_proteome(human, 2, gap=1))
dfgap1 = dfgap1.set_index('seq')

dfgap2 = counter_to_df(count_kmers_proteome(human, 2, gap=2))
dfgap2 = dfgap2.set_index('seq')

def fit_ising(f1, f2s, niter=1, nmcmc=1e6, epsilon=0.1, Jk=None, prng=None):
    if prng is None:
        prng = np.random
    h = np.array(np.log(f1['freq']))
    h -= np.mean(h)
    print(h)
    if Jk is None:
        Jk = np.zeros((N-1, q, q))
    for i in range(niter):
        def jump(x):
            return prng.randint(q, size=N)
        def energy(x):
            return clib.energy(x, h, Jk)
        x0 = jump(None)
        samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng)
        samples = [''.join(aas_arr[s]) for s in samples]
        for gap in range(len(f2s)):
            m = f2s[gap].merge(counter_to_df(count_kmers_iterable(samples, 2, gap=gap)), left_index=True, right_on='seq')
            #print(m)
            m['logfold'] = np.log(m['freq_x']/m['freq_y'])
            print(i, gap, np.mean(np.abs(m['logfold'])))
            for idx, row in m.iterrows():
                logfold = row['logfold']
                aa1 = aatonumber(row['seq'][0])
                aa2 = aatonumber(row['seq'][1])
                Jk[gap, aa1, aa2] += logfold * epsilon
    return h, Jk

print('start fitting')
seed = 1234
prng = np.random.RandomState(seed)
h, Jk = fit_ising(df0, [df1, dfgap1, dfgap2], nmcmc=1e6, niter=3, epsilon=0.2, prng=prng)

dfh = pd.DataFrame(index=aas_arr,
                   data=h,
                   columns=['h'])
doublets = [''.join(s) for s in itertools.product(list(aminoacids), repeat=2)]
dfJk = pd.DataFrame(index=doublets,
                    data=[Jk[0,aatonumber(s[0]),aatonumber(s[1])] for s in doublets],
                    columns=['J0'])
for i in range(1, len(Jk)):
    dfJk['J%g'%i] = [Jk[i,aatonumber(s[0]),aatonumber(s[1])] for s in doublets]

dfh.to_csv('data/h.csv')
dfJk.to_csv('data/Jk.csv')
