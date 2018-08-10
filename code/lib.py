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
 
def counter_to_df(counter):
    return pd.DataFrame(dict(seq=list(counter.keys()), freq=normalize(counter)))
