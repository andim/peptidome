import random, re
import gzip
from mimetypes import guess_type
from functools import partial
from collections import defaultdict
import os.path
from itertools import groupby
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
from Bio import SeqIO

from numba import jit, njit


from . import nsb

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
aminoacids_set = set(aminoacids)
naminoacids = len(aminoacids)

def unique_amino_acids(proteome):
    "returns an array of all unique amino acids used within a proteome"
    return np.unique(list(''.join([seq for h, seq in proteome])))

def entropy_nsb(n, base=None):
    """"
    Estimate the entropy of a discrete distribution from counts per category

    n: array of counts 
    base: base in which to measure the entropy (default: nats)
    """
    N = np.sum(n)
    K = len(n)
    nxkx = nsb.make_nxkx(n, K)
    entropy = nsb.S(nxkx, N, K)
    if base:
        entropy /= np.log(base)
    return entropy

def entropy_grassberger(n, base=None):
    """"
    Estimate the entropy of a discrete distribution from counts per category

    n: array of counts 
    base: base in which to measure the entropy (default: nats)
    """
    N = np.sum(n)
    entropy = np.log(N) - np.sum(n*scipy.special.digamma(n+1e-20))/N
    if base:
        entropy /= np.log(base)
    return entropy

def calc_jsd(p, q, base=np.e):
    "Calculate Jensen Shannon Divergence between p and q"
    p, q = np.array(p, copy=True), np.array(q, copy=True)
    p /= np.sum(p)
    q /= np.sum(q)
    m = 0.5*(p + q)
    return 0.5*(scipy.stats.entropy(p, m, base=base)
                + scipy.stats.entropy(q, m, base=base))

def calc_mi(df2_or_seqs, gap=0):
    """Calculate the mutual information between
       residues from a count of pairs of amino acids.
       Uses the Treves-Panzeri correction for finite size
    """
    try:
        df2 = df2_or_seqs
        strcolumn_to_charcolumns(df2, 'seq')
    except:
        df2 = Counter(df2_or_seqs, k=2, gap=gap).to_df(norm=False, clean=True)
        strcolumn_to_charcolumns(df2, 'seq')
    df11 = df2.groupby('aa1').agg(np.sum)['count']
    df11 /= np.sum(df11)
    df12 = df2.groupby('aa2').agg(np.sum)['count']
    df12 /= np.sum(df12)
    df2['theory'] = [float(df11.loc[s[0]] * df12.loc[s[1]]) for s in df2['seq']]
    df2['freq'] = df2['count']/np.sum(df2['count'])
    mi = np.sum(df2['freq']*np.log2(df2['freq']/df2['theory']))
    micorr = mi - (len(aminoacids)-1)**2/(2*np.log(2)*np.sum(df2['count']))
    return micorr

def calc_mi_std(seqs, gap):
    mis = []
    for i in range(30):
        df2 = Counter(random.sample(seqs, int(len(seqs)/2)), k=2, gap=gap).to_df(norm=False, clean=True)
        mis.append(calc_mi(df2))
    return np.mean(mis), np.std(mis, ddof=1)/2**.5

def strcolumn_to_charcolumns(df, column, prefix='aa'):
    """Build columns of chars from a column of strings of fixed length."""
    k = len(df[column][0]) 
    for i in range(1, k+1):
        df[prefix+str(i)] = [s[i-1] for s in df[column]]
    return df

def scrambled(iterable):
    for s in iterable:
        l = list(s)
        random.shuffle(l)
        shuffled = ''.join(l)
        yield shuffled

def to_kmers(seqs, k, return_index=False):
    """Generator yielding all possible kmers from a set of sequences.

    seqs: list of sequences
    k: length of kmer
    return_index: if true yield tuple seq, index
    """
    if return_index:
        for index, seq in enumerate(seqs):
                for i in range(len(seq)-k+1):
                    s = seq[i:i+k]
                    if isvalidaa(s):
                        yield s, index
    else:
        for seq in seqs:
            for i in range(len(seq)-k+1):
                s = seq[i:i+k]
                if isvalidaa(s):
                    yield s

_aatonumber = {c: i for i, c in enumerate(aminoacids)}
_numbertoaa = {i: c for i, c in enumerate(aminoacids)}

def map_aatonumber(seq):
    """
    Map sequence to array of number
    """
    seq = np.array(list(seq))
    return np.vectorize(_aatonumber.__getitem__)(seq)

def map_numbertoaa(seq):
    """
    Map integer to amino acid sequence
    """
    seq = list(seq)
    return np.vectorize(_numbertoaa.__getitem__)(seq)


def aatonumber(char):
    return _aatonumber[char]


def map_matrix(matrix, map_=_aatonumber):
    """
    Remap elements in a numpy array 

    Parameters
    ----------
    array : np.array
        Matrix to be remapped
    map_ : dict
        Map to be applied to matrix elements

    Returns
    -------
    np.array
        Remapped matrix
    """
    return np.vectorize(map_.__getitem__)(matrix)

def kmers_to_matrix(kmers):
    """"
    Map a list of str kmers to an integer numpy array.

    Parameters
    ----------
    kmers : iterable of strings
        kmers to be converted
    Returns
    -------
    np.array
        Mapped array
    """
    matrix_str =  np.array([list(kmer) for kmer in kmers])
    matrix = map_matrix(matrix_str)
    return matrix


class Counter(defaultdict):

    def __init__(self, iterable, k, gap=0, **kwargs):
        """
        Counter class

        iterable: sequences or proteome filename
        k: int, kmer length
        gap: int, gap between first and subsequent letters
        """
        super(Counter, self).__init__(int)
        self.k = k
        self.gap = gap
        if isinstance(iterable, str):
            iterable = fasta_iter(iterable, returnheader=False)
        self.count(iterable, **kwargs)

    def count(self, iterable, **kwargs):
        for seq in iterable:
            count_kmers(seq, self.k, gap=self.gap, counter=self, **kwargs)

    def clean(self):
        "keep only kmers composed of standard amino acids"
        keys = list(self.keys())
        for key in keys:
            if not isvalidaa(key):
                del self[key]

    def to_df(self, norm=True, clean=True):
        """Convert a (kmer, count) dict to a pandas DataFrame
        
        clean: only accept counts responding to valid amino acid letters 
        """
        if clean:
            self.clean()
        if norm:
            df = pd.DataFrame(dict(seq=list(self.keys()), freq=normalize(self)))
        else:
            arr = np.array(list(self.values()), dtype=np.float)
            df = pd.DataFrame(dict(seq=list(self.keys()), count=arr))
        df.sort_values('seq', inplace=True)
        return df

def count_kmers_iterable(iterable, k, clean=False, **kwargs):
    """
    Count number of kmers in all strings of an iterable
    """
    counter = defaultdict(int)
    for seq in iterable:
        count_kmers(seq, k, counter=counter, **kwargs)
    if clean:
        counter = {k:counter[k] for k in counter.keys() if isvalidaa(k)}
    return counter

def calc_tripletmodelparams(proteome):
    df = Counter(proteome, 1).to_df(norm=True)
#    df = counter_to_df(count_kmers_proteome(proteome, 1), norm=True)
    df = df.set_index('seq')
    charlogp = np.log10(df['freq']).to_dict()

    #df1 = counter_to_df(count_kmers_proteome(proteome, 2), norm=False)
    df1 = Counter(proteome, 2).to_df(norm=False)
    strcolumn_to_charcolumns(df1, 'seq')
    count = df1.pivot(columns='aa1', index='aa2')['count']
    count /= np.sum(count, axis=0)
    count[count.isna()] = 1e-10
    doubletlogp = np.log10(count).to_dict()

    df2 = Counter(proteome, 3).to_df(norm=False)
    df2['aa12'] = [s[:2] for s in df2['seq']]
    df2['aa3'] = [s[2] for s in df2['seq']]
    count = df2.pivot(columns='aa12', index='aa3')['count']
    count /= np.sum(count, axis=0)
    count[count.isna()] = 1e-10
    tripletlogp = np.log10(count).to_dict()

    modelparams = dict(charlogp=charlogp, doubletlogp=doubletlogp, tripletlogp=tripletlogp)
    return modelparams


def iscontained(string, strings):
    "Is one of the strings contained in string?"
    for s in strings:
        if s in string:
            return True
    return False

try:
    from .clib import count_kmers
except ImportError:
    print('clib not found')
    def count_kmers(string, k, counter=None, gap=0):
        """
        Count occurrence of kmers in a given string.
        """
        if counter is None:
            counter = defaultdict(int)
        for i in range(len(string)-k-gap+1):
            if gap:
                counter[string[i]+string[i+gap+1:i+k+gap]] += 1
            else:
                counter[string[i:i+k]] += 1
        return counter

def plot_sorted(data, ax=None, normalize=True, scalex=1.0, scaley=1.0, **kwargs):
    if ax is None:
        ax = plt.gca()
    sorted_data = np.sort(data)  # Or data.sort(), if data can be modified
    # Cumulative counts:
    if normalize:
        norm = sorted_data.size
    else:
        norm = 1
    #ax.step(sorted_data, np.arange(sorted_data.size)/norm)  # From 0 to the number of data points-1
    ax.set_xscale('log')
    ax.set_yscale('log')
    return ax.step(sorted_data[::-1]*scalex, scaley*np.arange(sorted_data.size)/norm, **kwargs)


def normalize(counter):
    "Given a (kmer, count) dict returns a normalized array of frequencies"
    arr = np.array(list(counter.values()), dtype=np.float)
    arr /= np.sum(arr)
    return arr

def isvalidaa(string):
    "returns true if string is composed only of characters from the standard amino acid alphabet"
    return all(c in aminoacids_set for c in string)

def counter_to_df(counter, norm=True, clean=True):
    """Convert a (kmer, count) dict to a pandas DataFrame
    
    clean: only accept counts responding to valid amino acid letters 
    """
    if clean:
        counter = {k:counter[k] for k in counter.keys() if isvalidaa(k)}
    if norm:
        return pd.DataFrame(dict(seq=list(counter.keys()), freq=normalize(counter)))
    arr = np.array(list(counter.values()), dtype=np.float)
    return pd.DataFrame(dict(seq=list(counter.keys()), count=arr))

def loglikelihood_independent(string, charlogp=None, k=None, **kwargs):
    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    for c in string:
        try:
            logp += charlogp[c]
        except KeyError:
            logp = np.nan
    return logp

def loglikelihood_mc(string, charlogp=None, doubletlogp=None, k=None, **kwargs):
    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    cold = None
    for c in string:
        try:
            if not cold:
                logp += charlogp[c]
            else:
                logp += doubletlogp[cold][c]
        except KeyError:
            logp = np.nan
        cold = c
    return logp

def loglikelihood_triplet(string, charlogp=None, doubletlogp=None, tripletlogp=None, k=None):
    """ Calculate the loglikelihood of a given string given a triplet model.

    charlogp: log probabilities of different characters log P(c)
    doubletlogp: conditional frequency of character given previous character log P(c_i | c_i-1)
    tripletlogp: conditional frequency of character given previous two characters log P(c_i | c_i-1, c_i-2)
    """

    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    cm1, cm2 = None, None
    for c in string:
        try:
            if (not cm1) and (not cm2):
                logp += charlogp[c]
            elif not cm2:
                logp += doubletlogp[cm1][c]
            else:
                logp += tripletlogp[cm2+cm1][c]
        except KeyError:
            logp = np.nan
        cm2 = cm1
        cm1 = c
    return logp

def likelihoods_epitopes(epitopes, likelihood, k):
    epitopes = list(epitopes)
    likelihoods = np.array([likelihood(seq[i:i+k], k) for seq in epitopes for i in range(len(seq)-k+1)])
    weights = np.array([1.0/(len(seq)-k+1) for seq in epitopes for i in range(len(seq)-k+1)])
    return likelihoods, weights


@jit
def mcmcsampler(x0, energy, jump, nsteps=1000, nburnin=0, nsample=1):
    """Markov chain Monte carlo sampler (JIT enabled).

    x0: starting position (array)
    energy(x): function for calculating energy
    jump(x): function for calculating a proposed new position
    nburnin: burnin period in which states are not saved
    nsample: sample interval for saving states
    
    returns array of states
    """
    prng = np.random
    nsteps, nburnin, nsample = int(nsteps), int(nburnin), int(nsample)
    x = x0
    Ex = energy(x)
    samples = np.zeros(((nsteps-nburnin)//nsample, x0.shape[0]), dtype=np.int64)
    counter = 0
    for i in range(1, nsteps+1):
        xp = jump(x)
        Exp = energy(xp)
        if (Exp < Ex) or (prng.rand() < np.exp(-Exp+Ex)):
            x = xp
            Ex = Exp
        if (i > nburnin) and ((i-nburnin) % nsample == 0):
            samples[counter] = x
            counter += 1
    return samples

def energy_ising(s, h, Jk):
    "energy of a translation invariant ising model"
    energy = sum(h[c] for c in s)
    for k, J in enumerate(Jk):
        for i in range(len(s)-1-k):
            energy += J[s[i]][s[i+k+1]]
    return -energy


def falling_factorial(x, n):
    "returns x (x-1) ... (x-n+1)"
    return scipy.special.factorial(x)/scipy.special.factorial(x-n+1)

codon_map = {"UUU":"F", "UUC":"F", "UUA":"L", "UUG":"L",
    "UCU":"S", "UCC":"S", "UCA":"S", "UCG":"S",
    "UAU":"Y", "UAC":"Y", "UAA":"STOP", "UAG":"STOP",
    "UGU":"C", "UGC":"C", "UGA":"STOP", "UGG":"W",
    "CUU":"L", "CUC":"L", "CUA":"L", "CUG":"L",
    "CCU":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAU":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGU":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "AUU":"I", "AUC":"I", "AUA":"I", "AUG":"M",
    "ACU":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAU":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGU":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GUU":"V", "GUC":"V", "GUA":"V", "GUG":"V",
    "GCU":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAU":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGU":"G", "GGC":"G", "GGA":"G", "GGG":"G",}
nt_to_ind = {
    'A' : 0,
    'C' : 1,
    'G' : 2,
    'U' : 3
    }
def ntfreq_to_aafreq(ntfreq):
    frequencies = {aa:0 for aa in aminoacids}
    for nts, aa in codon_map.items():
        if not aa == 'STOP':
            frequencies[aa] += np.prod([ntfreq[nt_to_ind[nt]] for nt in nts])
    sum_ = sum(frequencies.values())
    for aa in aminoacids:
        frequencies[aa] /= sum_
    return frequencies

def dict_to_array(dict_):
    "return an array from a dictionary by sorting the keys"
    keys = sorted(dict_.keys())
    return np.array([dict_[key] for key in keys])

from operator import ne
def disthamming(seqa, seqb):
    """Calculate Hamming distance between two sequences."""
    return sum(map(ne, seqa, seqb))

@jit(nopython=True)
def hammingdist_jit(seqa, seqb):
    return np.sum(seqa != seqb)

def pairwise_distances(data, N=100, distance=disthamming, data2=None,
                       weights=None, weights2=None,
                       warning=True, prng=np.random):
    """Pairwise distances between N randomly picked pairs from data."""
    N = int(N)
    data = np.asarray(data)
    if data2 is None:
        data2 = data
    else:
        data2 = np.asarray(data2)
    distances = np.zeros(N)
    if weights is not None:
        weights = np.asarray(weights)
        dweights = np.zeros(N)
        if weights2 is None:
            weights2 = weights
        else:
            weights2 = np.asarray(weights2)
    if warning and (len(data)*len(data2) < 10 * N):
        print('warning: low amount of data, %g vs. %g', (len(data)*len(data2), N))
    randints1 = prng.randint(len(data), size=N)
    randints2 = prng.randint(len(data2), size=N)
    for i in range(N):
        inda, indb = randints1[i], randints2[i]
        while inda == indb:
            inda, indb = prng.randint(len(data)), prng.randint(len(data2))
        seqa, seqb = data[inda], data2[indb]
        distances[i] = distance(seqa, seqb)
        if weights is not None:
            dweights[i] = weights[inda] * weights2[indb]
    if weights is not None:
        return distances, dweights
    return distances
    
@njit
def pairwise_distances_jit(data, N=100, data2=None, normalize=False):
    N = int(N)
    data = np.asarray(data)
    if data2 is None:
        data2 = data
    else:
        data2 = np.asarray(data2)
    num_rows = 2*int(N**.5)
    indices = np.random.choice(data.shape[0], num_rows, replace=False)
    data = data[indices[:num_rows//2]]
    data2 = data2[indices[num_rows//2:]]
    L = data.shape[1]
    hist = np.zeros(L+1)
    for i in range(len(data)):
        for j in range(len(data2)):
            dist = hammingdist_jit(data[i, :], data2[j, :])
            hist[dist] += 1
    if normalize:
        hist /= len(data)*len(data2)
    return hist
