import itertools
import numpy as np
import pandas as pd
from . import *
#from . import clib

import numba
from numba import jit, njit

def aacounts_str(seq):
    return aacounts_int(map_aatonumber(seq))

@jit(nopython=True)
def aacounts_int_jit(seq):
    counter = np.zeros(len(aminoacids), dtype=np.int64)
    for c in seq:
        counter[c] += 1
    return counter

#try:
#    from .clib import aacounts_int
#except:
#    print('clib not found')
#    def aacounts_int(seq):
#        counter = np.zeros(len(aminoacids), dtype=int)
#        for c in seq:
#            counter[c] += 1
#        return counter

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


def global_jump(x, q, prng=None):
    if prng is None:
        prng = np.random
    return prng.randint(q, size=len(x))

def local_jump(x, q, prng=None):
    if prng is None:
        prng = np.random
    xnew = x.copy()
    index = prng.randint(len(x))
    xnew[index] = (x[index] + prng.randint(1, q))%q
    return xnew

@jit(nopython=True)
def local_jump_jit(x, q, seed=None):
    prng = np.random
    if not (seed is None):
        prng.seed(seed)
    xnew = x.copy()
    index = prng.randint(len(x))
    xnew[index] = (x[index] + prng.randint(1, q))%q
    return xnew

def fit_potts(f1, f2s, niter=1, nmcmc=1e6, epsilon=0.1, Jk=None, prng=None, output=False):
    N = len(f2s)+1
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
            return local_jump(x, q)
        def energy(x):
            return clib.energy(x, h, Jk)
        x0 = global_jump(np.zeros(N), q, prng=prng)
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

def fit_global(fks, niter=1, nmcmc=1e6, epsilon=0.1, prng=None, output=False):
    N = len(fks[0])-1
    if prng is None:
        prng = np.random
    q = len(aminoacids)
    aas_arr = np.array(list(aminoacids))
    f1 = np.sum(np.arange(fks.shape[1])*fks, axis=1)/(fks.shape[1]-1)
    h = np.array(np.log(f1))
    h -= np.mean(h)
    hks = h.reshape(20, 1)*np.arange(fks.shape[1])
    #hks = np.zeros((q, N+1))
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
        if output:
            print(jsd)
    return hks

def fit_full_potts(train_matrix, sampler, niter=1, epsilon=0.1,
        q=naminoacids,
        pseudocount=1.0, prng=None, output=False):
    """ sampler(x0, energy, jump, prng=prng): function returning samples from the distribution """

    # calculate empirical observables
    fi = frequencies(train_matrix, num_symbols=q, pseudocount=pseudocount)
    fij = pair_frequencies(train_matrix, num_symbols=q, fi=fi, pseudocount=pseudocount)
    fij = pair_frequencies_average(fij)

    if prng is None:
        prng = np.random
    hi = np.log(fi)
    hi -= np.mean(hi)
    if output:
        print(hi)
    Jij = np.zeros_like(fij)
    for iteration in range(niter):
        if output:
            print('iteration %g'%iteration)

        x0 = global_jump(np.zeros(len(fi)), q, prng=prng)

        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_potts(x, hi, Jij)

        samples = sampler(x0, energy, jump)

        fi_model = frequencies(samples, q, pseudocount=pseudocount)
        fij_model = pair_frequencies(samples, q, fi_model, pseudocount=pseudocount)
        fij_model = pair_frequencies_average(fij_model)

        hi -= np.log(fi_model/fi)*epsilon
        #hi += (fi-fi_model)*epsilon
        if output:
            print('f1', calc_jsd(fi_model[0], fi[0]))
        Jij -= np.log(fij_model/fij)*epsilon
        #Jij += (fij-fij_model)*epsilon
        if output:
            print('f2', calc_jsd(fij_model[0, 1], fij[0, 1]))
    return hi, Jij

def fit_ncov(train_matrix, sampler,
             q=naminoacids,
             niter=1, epsilon=0.1, pseudocount=1.0,
             prng=None, output=False):
    """ sampler(x0, energy, jump, prng=prng): function returning samples from the distribution """

    # calculate empirical observables
    _, L = train_matrix.shape
    aacounts = to_aacounts(train_matrix)
    n1 = calc_n1(aacounts)
    n2 = calc_n2(aacounts)

    if prng is None:
        prng = np.random
    h = np.log(n1/L)
    h -= np.mean(h)
    J = np.zeros_like(n2)
    for iteration in range(niter):
        if output:
            print('iteration %g/%g'%(iteration+1,niter))

        x0 = global_jump(np.zeros(L), q, prng=prng)
        
        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_ncov(x, h, J)

        samples = sampler(x0, energy, jump)
        aacounts = to_aacounts(samples)

        n1_model = calc_n1(aacounts)
        n2_model = calc_n2(aacounts)
 
        h -= np.log(n1_model/n1)*epsilon
        J -= np.log(n2_model/n2)*epsilon
    return h, J

### Global third-order couplings  ###

@njit
def energy_nskew(x, h, J, J2):
    counts = aacounts_int_jit(x)
    q = len(h)
    e = 0
    for alpha in range(q):
        e -= h[alpha]*counts[alpha]
        for beta in range(alpha, q):
            e -= J[alpha, beta]*counts[alpha]*counts[beta]
            for gamma in range(beta, q):
                e -= J2[alpha, beta, gamma]*counts[alpha]*counts[beta]*counts[gamma]
    return e

def fit_nskew(train_matrix, sampler, h=None, J=None, J2=None, q=naminoacids,
            niter=1, epsilon=0.1, pseudocount=1.0,
            prng=None, output=False):
    """ sampler(x0, energy, jump, prng=prng): function returning samples from the distribution """

    # calculate empirical observables
    _, L = train_matrix.shape
    aacounts = to_aacounts(train_matrix)
    n1 = calc_n1(aacounts)
    n2 = calc_n2(aacounts)
    n3 = calc_n3(aacounts)

    if prng is None:
        prng = np.random
    if h is None:
        h = np.log(n1/L)
        h -= np.mean(h)
    else:
        h = h.copy()
    if J is None:
        J = np.zeros_like(n2)
    else:
        J = J.copy()
    if J2 is None:
        J2 = np.zeros_like(n3)
    else:
        J2 = J2.copy()
    for iteration in range(niter):
        if output:
            print('iteration %g/%g'%(iteration+1,niter))

        x0 = global_jump(np.zeros(L), q, prng=prng)
        
        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_nskew(x, h, J, J2)

        samples = sampler(x0, energy, jump)
        aacounts = to_aacounts(samples)

        n1_model = calc_n1(aacounts)
        n2_model = calc_n2(aacounts)
        n3_model = calc_n3(aacounts)
 
        h -= np.log(n1_model/n1)*epsilon
        J -= np.log(n2_model/n2)*epsilon
        J2 -= np.log(n3_model/n3)*epsilon
    return h, J, J2

### Diagonal global third-order couplings  ###

@njit
def calc_n3_diag(matrix):
    N, q = matrix.shape
    n3 = np.zeros(q)
    for s in range(N):
        for alpha in range(q):
            n3[alpha] += matrix[s, alpha]**3
    n3 /= N
    return n3

@njit
def energy_nskewdiag(x, h, J, J2):
    counts = aacounts_int_jit(x)
    q = len(h)
    e = 0
    for alpha in range(q):
        e -= h[alpha]*counts[alpha] + J2[alpha]*counts[alpha]**3
        for beta in range(alpha, q):
            e -= J[alpha, beta]*counts[alpha]*counts[beta]
    return e

def fit_nskewdiag(train_matrix, sampler, h=None, J=None, J2=None, q=naminoacids,
            niter=1, epsilon=0.1, pseudocount=1.0,
            prng=None, output=False):
    """ sampler(x0, energy, jump, prng=prng): function returning samples from the distribution """

    # calculate empirical observables
    _, L = train_matrix.shape
    aacounts = to_aacounts(train_matrix)
    n1 = calc_n1(aacounts)
    n2 = calc_n2(aacounts)
    n3 = calc_n3_diag(aacounts)

    if prng is None:
        prng = np.random
    if h is None:
        h = np.log(n1/L)
        h -= np.mean(h)
    else:
        h = h.copy()
    if J is None:
        J = np.zeros_like(n2)
    else:
        J = J.copy()
    if J2 is None:
        J2 = np.zeros_like(n3)
    else:
        J2 = J2.copy()
    for iteration in range(niter):
        if output:
            print('iteration %g/%g'%(iteration+1,niter))

        x0 = global_jump(np.zeros(L), q, prng=prng)
        
        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_nskewdiag(x, h, J, J2)

        samples = sampler(x0, energy, jump)
        aacounts = to_aacounts(samples)

        n1_model = calc_n1(aacounts)
        n2_model = calc_n2(aacounts)
        n3_model = calc_n3_diag(aacounts)
 
        h -= np.log(n1_model/n1)*epsilon
        J -= np.log(n2_model/n2)*epsilon
        J2 -= np.log(n3_model/n3)*epsilon

    return h, J, J2

### Global third-order couplings + second order distance couplings ###

@njit
def energy_nskewfcov(x, h, J, J2, hi, Jij):
    return energy_potts(x, hi, Jij) + energy_nskew(x, h, J, J2)

def fit_nskewfcov(train_matrix, sampler, h=None, J=None, J2=None, q=naminoacids,
            niter=1, epsilon=0.1, pseudocount=1.0,
            prng=None, output=False):
    """ sampler(x0, energy, jump, prng=prng): function returning samples from the distribution """

    if prng is None:
        prng = np.random

    # calculate empirical observables
    _, L = train_matrix.shape
    aacounts = to_aacounts(train_matrix)
    n1 = calc_n1(aacounts)
    n2 = calc_n2(aacounts)
    n3 = calc_n3(aacounts)

    fi = frequencies(train_matrix, num_symbols=q, pseudocount=pseudocount)
    fij = pair_frequencies(train_matrix, num_symbols=q, fi=fi, pseudocount=pseudocount)



    # initialize Lagrange multipliers
    if h is None:
        h = np.log(n1/L)
        h -= np.mean(h)
    else:
        h = h.copy()
    if J is None:
        J = np.zeros_like(n2)
    else:
        J = J.copy()
    if J2 is None:
        J2 = np.zeros_like(n3)
    else:
        J2 = J2.copy()
    hi = np.zeros_like(fi)
    Jij = np.zeros_like(fij)
   
    for iteration in range(niter):
        if output:
            print('iteration %g/%g'%(iteration+1,niter))

        x0 = global_jump(np.zeros(L), q, prng=prng)
        
        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_nskewfcov(x, h, J, J2, hi, Jij)

        samples = sampler(x0, energy, jump)
        aacounts = to_aacounts(samples)

        n1_model = calc_n1(aacounts)
        n2_model = calc_n2(aacounts)
        n3_model = calc_n3(aacounts)

        fi_model = frequencies(samples, q, pseudocount=pseudocount)
        fij_model = pair_frequencies(samples, q, fi_model, pseudocount=pseudocount)
 
        h -= np.log(n1_model/n1)*epsilon
        J -= np.log(n2_model/n2)*epsilon
        J2 -= np.log(n3_model/n3)*epsilon

        hi -= np.log(fi_model/fi)*epsilon
        Jij -= np.log(fij_model/fij)*epsilon

    return h, J, J2, hi, Jij


### Other code ###


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

@njit
def energy_potts(x, hi, Jij):
    e = 0
    for i in range(len(x)):
        e -= hi[i, x[i]]
        for j in range(i+1, len(x)):
            e -= Jij[i, j, x[i], x[j]]
    return e

@njit
def frequencies(matrix, num_symbols, pseudocount=0, weights=None):
    """
    Calculate single-site frequencies

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
    num_symbols : int
        Number of different symbols
    weights: np.array
        Vector of length N of relative weights of different sequences

    Returns
    -------
    np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters
    """
    N, L = matrix.shape
    fi = pseudocount/num_symbols * np.ones((L, num_symbols))
    if weights is None:
        for s in range(N):
            for i in range(L):
                fi[i, matrix[s, i]] += 1.0
        return fi / (N+pseudocount)
    else:
        normalized_weights = N*weights/np.sum(weights)
        for s in range(N):
            for i in range(L):
                fi[i, matrix[s, i]] += normalized_weights[s]
    return fi / (N+pseudocount)

@njit
def pair_frequencies(matrix, num_symbols, fi, pseudocount=0, weights=None):
    """
    Calculate pairwise frequencies of symbols.

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
    num_symbols : int
        Number of different symbols
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters.
    weights: np.array
        Vector of length N of relative weights of different sequences

    Returns
    -------
    np.array
        Matrix of size L x L x num_symbols x num_symbols containing
        relative pairwise frequencies of all character combinations
    """
    N, L = matrix.shape
    fij = pseudocount/num_symbols**2 * np.ones((L, L, num_symbols, num_symbols))
    if weights is None:
        for s in range(N):
            for i in range(L):
                for j in range(i + 1, L):
                    fij[i, j, matrix[s, i], matrix[s, j]] += 1
    else:
        normalized_weights = N*weights/np.sum(weights)
        for s in range(N):
            for i in range(L):
                for j in range(i + 1, L):
                    fij[i, j, matrix[s, i], matrix[s, j]] += normalized_weights[s]

    # symmetrize matrix
    for i in range(L):
        for j in range(i + 1, L):
            for alpha in range(num_symbols):
                for beta in range(num_symbols):
                    fij[j, i, beta, alpha] = fij[i, j, alpha, beta]
 
    # normalize frequencies by the number
    # of sequences
    fij /= (N+pseudocount)

    # set the frequency of a pair (alpha, alpha)
    # in position i to the respective single-site
    # frequency of alpha in position i
    for i in range(L):
        for alpha in range(num_symbols):
            fij[i, i, alpha, alpha] = fi[i, alpha]

    return fij

def pair_frequencies_average(fij, copy=False):
    """ Average pair frequencies to make them translation invariant """
    if copy:
        fij = np.copy(fij)
    L = fij.shape[0]
    fij_d = np.array([np.mean(np.array([list(fij[i, i+d]) for i in range(0, L-d)]), axis=0) for d in range(0, L)])
    for i in range(L):
        for j in range(L):
            fij[i, j] = fij_d[np.abs(j-i)]
    return fij

def compute_covariance_matrix(fi, fij):
    cij = fij[:, :, :, :] - fi[:, np.newaxis, :, np.newaxis] * fi[np.newaxis, :, np.newaxis, :]
    return cij


@njit
def compute_flattened_covariance_matrix(fi, fij):
    """
    Compute the covariance matrix in a flat format for mean-field inversion.

    Parameters
    ----------
    fi : np.array
        Matrix of size L x num_symbols
        containing frequencies.
    fij : np.array
        Matrix of size L x L x num_symbols x
        num_symbols containing pair frequencies.

    Returns
    -------
    np.array
        Covariance matrix of size L x (num_symbols-1) x L x (num_symbols-1) 
        
    """
    L, num_symbols = fi.shape
    # The covariance values concerning the last symbol
    # are required to equal zero and are not represented
    # in the covariance matrix (important for taking the
    # inverse) - resulting in a matrix of size
    # (L * (num_symbols-1)) x (L * (num_symbols-1))
    # rather than (L * num_symbols) x (L * num_symbols).
    covariance_matrix = np.zeros((L * (num_symbols - 1),
                                  L * (num_symbols - 1)))
    for i in range(L):
        for j in range(L):
            for alpha in range(num_symbols - 1):
                for beta in range(num_symbols - 1):
                    covariance_matrix[
                        _flatten_index(i, alpha, num_symbols),
                        _flatten_index(j, beta, num_symbols),
                    ] = fij[i, j, alpha, beta] - fi[i, alpha] * fi[j, beta]
    return covariance_matrix

@njit
def _flatten_index(i, alpha, num_symbols):
    """
    Map position and symbol to index in
    the covariance matrix.

    Parameters
    ----------
    i : int, np.array of int
        The alignment column(s).
    alpha : int, np.array of int
        The symbol(s).
    num_symbols : int
        The number of symbols of the
        alphabet used.
    """
    return i * (num_symbols - 1) + alpha

@njit
def triplet_frequencies(matrix, num_symbols, pseudocount=0, weights=None):
    """
    Calculate triplet frequencies of symbols.

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
    num_symbols : int
        Number of different symbols
    weights: np.array
        Vector of length N of relative weights of different sequences

    Returns
    -------
    np.array
        Matrix of size L x L x L x num_symbols x num_symbols x num_symbols containing
        relative triplet frequencies of all character combinations
    """
    N, L = matrix.shape
    fijk = pseudocount/num_symbols**3*np.ones((L, L, L, num_symbols, num_symbols, num_symbols))
    if weights is None:
        for s in range(N):
            for i in range(L):
                for j in range(L):
                    for k in range(L):
                        fijk[i, j, k, matrix[s, i], matrix[s, j], matrix[s, k]] += 1
    else:
        normalized_weights = N*weights/np.sum(weights)
        for s in range(N):
            for i in range(L):
                for j in range(L):
                    for k in range(L):
                        fijk[i, j, k, matrix[s, i], matrix[s, j], matrix[s, k]] += normalized_weights[s]

    # normalize frequencies by the number
    # of sequences
    fijk /= (N+pseudocount)

    return fijk

@njit
def quadruplet_frequencies(matrix, num_symbols, pseudocount=0):
    """
    Calculate quadruplet frequencies of symbols.

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
    num_symbols : int
        Number of different symbols

    Returns
    -------
    np.array
        Matrix of size L x L x L x L x num_symbols x num_symbols
        x num_symbols x num_symbols containing
        relative frequencies of character combinations
    """
    N, L = matrix.shape
    fijkl = pseudocount/num_symbols**4 * np.ones((L, L, L, L,
                                            num_symbols, num_symbols,
                                            num_symbols, num_symbols))
    for s in range(N):
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    for l in range(L):
                        fijkl[i, j, k, l, matrix[s, i], matrix[s, j], matrix[s, k], matrix[s, l]] += 1

    # normalize frequencies by the number
    # of sequences
    fijkl /= (N+pseudocount)

    return fijkl

def zero_sum_gauge(J_ij):
    J_ij_0 = (J_ij
              - np.mean(J_ij, axis=2)[:, :, np.newaxis, :]
              - np.mean(J_ij, axis=3)[:, :, :, np.newaxis]
              + np.mean(J_ij, axis=(2,3))[:, :, np.newaxis, np.newaxis])
    return J_ij_0

def compute_cijk(fijk, fij, fi):
    #https://en.wikipedia.org/wiki/Ursell_function
    # fijk - fi fjk - fj fik - fk fij + 2*fi fj fk
    return (fijk
            - (fi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
               * fij[np.newaxis, :, :, np.newaxis, :, :])
            - (fi[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
               * fij[:, np.newaxis, :, :, np.newaxis, :])
            - (fi[np.newaxis, np.newaxis :, np.newaxis, np.newaxis, :]
                * fij[:, :, np.newaxis, :, :, np.newaxis])
            + (2*fi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] *
                fi[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis] *
                fi[np.newaxis, np.newaxis :, np.newaxis, np.newaxis, :]))

def compute_fold_ijk(fijk, fi):
    return fijk / (fi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] *
                fi[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis] *
                fi[np.newaxis, np.newaxis :, np.newaxis, np.newaxis, :])

def flatten_ij(cij):
    mask = ~np.eye(cij.shape[0], dtype=bool)
    return cij[mask].flatten()

def flatten_ijk(cijk):
    L = cijk.shape[0]
    num_symbols = cijk.shape[3]
    flattened = []
    for i in range(L):
        for j in range(i+1, L):
            for k in range(j+1, L):
                for alpha in range(num_symbols):
                    for beta in range(num_symbols):
                        for gamma in range(num_symbols):
                            flattened.append(cijk[i, j, k, alpha, beta, gamma])
    return np.array(flattened)

def Fpotts_thermodynamic_integration(hi, Jij, integration_intervals=1, mcmc_kwargs=dict()):
    prng = np.random
    L, q = hi.shape

    def Fprime(alpha):
        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_potts(x, hi, alpha*Jij)
        x0 = prng.randint(q, size=L)
        matrix = mcmcsampler(x0, energy, jump, **mcmc_kwargs)
        return np.mean([energy_potts(x, np.zeros_like(hi), Jij) for x in matrix])

    xs = np.linspace(0, 1, integration_intervals+1)
    Fprimes = [Fprime(x) for x in xs]
    Fint = scipy.integrate.simps(Fprimes, xs)
    F0 = -np.sum(np.log(np.sum(np.exp(hi), axis=1)))
    return F0 + Fint

#### Code for global covariance model ####

def to_aacounts(matrix):
    return np.array([list(aacounts_int_jit(seq)) for seq in matrix])

@njit
def energy_ncov(x, h, J):
    counts = aacounts_int_jit(x)
    q = len(h)
    e = 0
    for alpha in range(q):
        e -= h[alpha]*counts[alpha]
        for beta in range(alpha, q):
            e -= J[alpha, beta]*counts[alpha]*counts[beta]
    return e

def calc_n1(aacounts):
    return np.mean(aacounts, axis=0)

@njit
def calc_n2(matrix):
    N, q = matrix.shape
    n2 = np.zeros((q, q))
    for s in range(N):
        for alpha in range(q):
            for beta in range(q):
                n2[alpha, beta] += matrix[s, alpha]*matrix[s, beta]     
    n2 /= N
    return n2

@njit
def calc_n3(matrix):
    N, q = matrix.shape
    n3 = np.zeros((q, q, q))
    for s in range(N):
        for alpha in range(q):
            for beta in range(q):
                for gamma in range(q):
                    n3[alpha, beta, gamma] += matrix[s, alpha]*matrix[s, beta]*matrix[s, gamma]
    n3 /= N
    return n3


def Fcov_thermodynamic_integration(h, J, L, integration_intervals=1, mcmc_kwargs=dict()):
    prng = np.random
    q = h.shape[0]

    def Fprime(alpha):
        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_ncov(x, h, alpha*J)
        x0 = prng.randint(q, size=L)
        matrix = mcmcsampler(x0, energy, jump, **mcmc_kwargs)
        return np.mean([energy_ncov(x, np.zeros_like(h), J) for x in matrix])

    xs = np.linspace(0, 1, integration_intervals+1)
    Fprimes = [Fprime(x) for x in xs]
    Fint = scipy.integrate.simps(Fprimes, xs)
    F0 = -L*np.log(np.sum(np.exp(h)))
    return F0 + Fint
