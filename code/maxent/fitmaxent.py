import sys
sys.path.append('..')

import itertools, copy
import numpy as np
import scipy.misc
import pandas as pd
import seaborn as sns

from lib import *

aas_arr = np.array(list(aminoacids))
q = len(aminoacids)
N = 6

df0 = counter_to_df(count_kmers_proteome(human, 1))
df0 = df0.set_index('seq')

df1 = counter_to_df(count_kmers_proteome(human, 2, gap=0))
df1 = df1.set_index('seq')

dfgap1 = counter_to_df(count_kmers_proteome(human, 2, gap=1))
dfgap1 = dfgap1.set_index('seq')

dfgap2 = counter_to_df(count_kmers_proteome(human, 2, gap=2))
dfgap2 = dfgap2.set_index('seq')

def fit_ising(f1, f2s, niter=1, nmcmc=1e6, epsilon=0.1, Jk=None, prng=None):
    if prng is None:
        prng = np.random
    h = np.log(f1['freq']).to_dict()
    print(h)
    if Jk is None:
        J0 = np.zeros((len(aminoacids), len(aminoacids)))
        J0 = pd.DataFrame(np.asarray(J0), index=list(aminoacids), columns=list(aminoacids)).to_dict()
        Jk = [J0]
        for gap in range(1, len(f2s)):
            Jk.append(copy.deepcopy(J0))
    for i in range(niter):
        def jump(x):
            return ''.join(np.random.choice(aas_arr, size=6))
            # the following is faster
            #return ''.join(aas_arr[prng.randint(q, size=6)])
        def energy(x):
            return energy_ising(x, h, Jk)
        x0 = jump(None)
        samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng)
        samples = [''.join(sample) for sample in samples]
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
seed = 1234
prng = np.random.RandomState(seed)
h, Jk = fit_ising(df0, [df1, dfgap1, dfgap2], nmcmc=1e6, niter=3, epsilon=0.2, prng=prng)

dfh = pd.DataFrame(index=[key for key in h],
                   data=[h[key] for key in h],
                   columns=['h'])
doublets = [''.join(s) for s in itertools.product(list(aminoacids), repeat=2)]
dfJk = pd.DataFrame(index=doublets,
                    data=[Jk[0][s[0]][s[1]] for s in doublets],
                    columns=['J0'])
for i in range(1, len(Jk)):
    dfJk['J%g'%i] = [Jk[i][s[0]][s[1]] for s in doublets]

dfh.to_csv('data/h.csv')
dfJk.to_csv('data/Jk.csv')
