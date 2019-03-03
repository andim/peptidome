import itertools

import sys
sys.path.append('..')
from lib import *
import clib

import numpy as np
import pandas as pd

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

def fit_ising(f1, f2s, niter=1, nmcmc=1e6, epsilon=0.1, Jk=None, prng=None, output=False, N=10):
    if prng is None:
        prng = np.random
    h = np.array(np.log(f1['freq']))
    h -= np.mean(h)
    if output:
        print(h)
    q = len(aminoacids)
    aas_arr = np.array(list(aminoacids))
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

def save(h, Jk):
    aas_arr = np.array(list(aminoacids))
    dfh = pd.DataFrame(index=aas_arr, data=h, columns=['h'])
    dfJk = pd.DataFrame(data=Jk, columns=range(len(Jk)))
    #dfJk = pd.DataFrame(index=doublets,
    #                    data=[Jk[0,aatonumber(s[0]),aatonumber(s[1])] for s in doublets],
    #                    columns=['J0'])
    #for i in range(1, len(Jk)):
    #    dfJk['J%g'%i] = [Jk[i,aatonumber(s[0]),aatonumber(s[1])] for s in doublets]

    dfh.to_csv('data/h.csv')
    dfJk.to_csv('data/Jk.csv')
