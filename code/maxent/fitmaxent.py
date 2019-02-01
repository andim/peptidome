import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
import clib

output = True
aas_arr = np.array(list(aminoacids))
q = len(aminoacids)
N = 10
seed = 1234

humanseqs = [s for s in fasta_iter(human, returnheader=False)]
train, test = train_test_split(humanseqs, test_size=0.5)

def calc_logfold(df1, df2, **kwargs):
    default = dict(left_index=True, right_index=True)
    default.update(kwargs)
    m = df1.merge(df2, **default)
    m['logfold'] = np.log(m['freq_x']/m['freq_y'])
    jsd = calc_jsd(m['freq_x'], m['freq_y'])
    return m, jsd
     
def count(seqs, *args, **kwargs):
    df = counter_to_df(count_kmers_iterable(seqs, *args, **kwargs))
    df = df.set_index('seq')
    df = df.sort_index()
    return df

# evaluate empirical observables for fitting
df0 = count(train, 1)
df1 = count(train, 2, gap=0)
dfgap1 = count(train, 2, gap=1)
dfgap2 = count(train, 2, gap=2)

def fit_ising(f1, f2s, niter=1, nmcmc=1e6, epsilon=0.1, Jk=None, prng=None):
    if prng is None:
        prng = np.random
    h = np.array(np.log(f1['freq']))
    h -= np.mean(h)
    if output:
        print(h)
    if Jk is None:
        Jk = np.zeros((N-1, q, q))
    for i in range(niter):
        if output:
            print('iteration %g'%i)
        def jump(x):
            return prng.randint(q, size=N)
        def energy(x):
            return clib.energy(x, h, Jk)
        x0 = jump(None)
        samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng)
        samples = [''.join(aas_arr[s]) for s in samples]
        f1_model = count(samples, 1)
        m, jsd = calc_logfold(f1, f1_model)
        m = m.sort_values('seq')
        h += np.asarray(m['logfold'])*epsilon
        if output:
            print('f1', jsd)
        for gap in range(len(f2s)):
            f2_model = count(samples, 2, gap=gap)
            m, jsd = calc_logfold(f2s[gap], f2_model)
            if output:
                print('f2, gap', gap, jsd)
            for idx, row in m.iterrows():
                logfold = row['logfold']
                aa1 = aatonumber(idx[0])
                aa2 = aatonumber(idx[1])
                Jk[gap, aa1, aa2] += logfold * epsilon
    return h, Jk


print('start fitting')
prng = np.random.RandomState(seed)
h, Jk = fit_ising(df0, [df1, dfgap1, dfgap2], nmcmc=1e5, niter=20, epsilon=0.1, prng=prng)

df4 = count(train, 4)
df4_test = count(test, 4)
m, jsd_test = calc_logfold(df4, df4_test)
jsd_flat = calc_jsd(df4_test['freq'], np.ones_like(df4_test['freq']))

nmcmc = 1e6
prng = np.random
def jump(x):
    return prng.randint(q, size=N)
def energy(x):
    return clib.energy(x, h, Jk)
x0 = jump(None)
samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng)
samples = [''.join(aas_arr[s]) for s in samples]
df4_model = count(samples, 4)
m, jsd_model = calc_logfold(df4_test, df4_model)
print('4mer', 'test', jsd_test, 'model', jsd_model, 'flat', jsd_flat)

dfh = pd.DataFrame(index=aas_arr, data=h, columns=['h'])
doublets = [''.join(s) for s in itertools.product(list(aminoacids), repeat=2)]
dfJk = pd.DataFrame(index=doublets,
                    data=[Jk[0,aatonumber(s[0]),aatonumber(s[1])] for s in doublets],
                    columns=['J0'])
for i in range(1, len(Jk)):
    dfJk['J%g'%i] = [Jk[i,aatonumber(s[0]),aatonumber(s[1])] for s in doublets]

dfh.to_csv('data/h.csv')
dfJk.to_csv('data/Jk.csv')
