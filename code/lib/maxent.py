import numpy as np
import pandas as pd
from . import *
from . import clib

def aacounts(seq):
    return aacounts_int(map_aatonumber(seq))

try:
    from .clib import aacounts_int
except:
    print('clib not found')
    def aacounts_int(seq):
        counter = np.zeros(len(aminoacids), dtype=int)
        for c in seq:
            counter[c] += 1
        return counter

def energy_global(counts, hks):
    energy = 0.0
    for i, c in enumerate(counts):
        energy += hks[i, c]
    return -energy

def prob_aa(aacountss, k, pseudocount=1.0):
    prob_aa_ks = []
    for i in range(len(aminoacids)):
        bincounts = np.bincount(np.array(aacountss)[:, i])
        prob_aa_k = np.ones(k+1)*pseudocount
        prob_aa_k[:len(bincounts)] += bincounts
        prob_aa_k /= np.sum(prob_aa_k)
        prob_aa_ks.append(prob_aa_k)
    prob_aa_ks = np.array(prob_aa_ks)
    return prob_aa_ks


def fit_global(fks, niter=1, nmcmc=1e6, epsilon=0.1, prng=None, output=False):
    N = len(fks[0])-1
    if prng is None:
        prng = np.random
    q = len(aminoacids)
    aas_arr = np.array(list(aminoacids))
    hks = np.zeros((q, N+1))
    for i in range(niter):
        if output:
            print('iteration %g'%i)
        def jump(x):
            return prng.randint(q, size=N)
        def energy(x):
            return energy_global(aacounts_int(x), hks)
        x0 = jump(None)
        samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng)
        aacountss = [aacounts_int(s) for s in samples]
        prob_aa_ks = prob_aa(aacountss, N)
        #print(fks, prob_aa_ks)
        hks += (fks - prob_aa_ks)*epsilon
        jsd = calc_jsd(fks, prob_aa_ks)
        #print(hks, jsd)
    return hks


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

def calc_logfold(df1, df2, **kwargs):
    default = dict(left_index=True, right_index=True)
    default.update(kwargs)
    m = df1.merge(df2, **default)
    m['logfold'] = np.log(m['freq_x']/m['freq_y'])
    jsd = calc_jsd(m['freq_x'], m['freq_y'])
    return m, jsd

def pseudocount_f1(iterable, factor=1.0):
    df = pd.DataFrame.from_dict(dict(seq=list(aminoacids), count=np.ones(len(aminoacids))*factor))
    df.set_index('seq', inplace=True)
    df_count = counter_to_df(count_kmers_iterable(iterable, 1), norm=False)
    df_count.set_index('seq', inplace=True)
    df = df.add(df_count, fill_value=0.0)
    df['freq'] = df['count'] / np.sum(df['count'])
    return df[['freq']]

def pseudocount_f2(iterable, k, gap, f1):
    "Use pseudocounts to regularize pair frequencies"
    kmers = [''.join(s) for s in itertools.product(aminoacids, repeat=k)]
    ind = np.array([float(f1.loc[s[0]] * f1.loc[s[1]]) for s in kmers])
    df = pd.DataFrame.from_dict(dict(seq=kmers, count=ind*len(aminoacids)**2))
    df.set_index('seq', inplace=True)
    df_count = counter_to_df(count_kmers_iterable(iterable, k, gap), norm=False)
    df_count.set_index('seq', inplace=True)
    df = df.add(df_count, fill_value=0.0)
    df['freq'] = df['count'] / np.sum(df['count'])
    return df[['freq']]

def count(seqs, *args, **kwargs):
    df = counter_to_df(count_kmers_iterable(seqs, *args, **kwargs))
    df = df.set_index('seq')
    df = df.sort_index()
    return df
