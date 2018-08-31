from itertools import groupby
from collections import defaultdict
import numpy as np
import pandas as pd

def fasta_iter(fasta_name):
    """
    Given a fasta file return a iterator over tuples of header, complete sequence.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line[0] == ">"))
    for header in faiter:
        # drop the ">"
        header = next(header)[1:].strip()
        # join all sequence lines together
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq

def unique_amino_acids(proteome):
    "returns an array of all unique amino acids used within a proteome"
    return np.unique(list(''.join([seq for h, seq in proteome])))

def count_kmers_proteome(proteome, k):
    counter = defaultdict(int)
    for header, sequence in fasta_iter(proteome):
        count_kmers(sequence, k, counter=counter)
    return counter

def count_kmers(string, k, counter=None):
    """
    Count number of kmers in a given string.
    """
    if counter is None:
        counter = defaultdict(int)
    for i in range(len(string)-k+1):
        counter[string[i:i+k]] += 1
    return counter

def normalize(counter):
    arr = np.array(list(counter.values()), dtype=np.float)
    arr /= np.sum(arr)
    return arr
 
def counter_to_df(counter, norm=True):
    if norm:
        return pd.DataFrame(dict(seq=list(counter.keys()), freq=normalize(counter)))
    arr = np.array(list(counter.values()), dtype=np.float)
    return pd.DataFrame(dict(seq=list(counter.keys()), count=arr))

def loglikelihood_independent(string, charprobdict, k=None):
    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    for c in string:
        try:
            logp += charprobdict[c]
        except KeyError:
            logp = np.nan
    return logp

def loglikelihood_mc(string, charprobdict, doubletprobdict, k=None):
    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    cold = None
    for c in string:
        try:
            if not cold:
                logp += charprobdict[c]
            else:
                logp += doubletprobdict[cold][c]
        except KeyError:
            logp = np.nan
        cold = c
    return logp
