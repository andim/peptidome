import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from lib import *

min_length = 100

aas =  aminoacids

def aa_frequencies(proteome, min_length=1):
    proteome = proteome_path(proteome)
    n = sum([1 for h, seq in fasta_iter(proteome) if len(seq)>=min_length])
    array = np.zeros((n, len(aas)))
    i = 0
    for h, seq in fasta_iter(proteome):
        seq = seq.replace('X', '')
        seq = seq.replace('U', '')
        if len(seq) < min_length:
            continue
        counter = {}
        for aa in aas:
            counter[aa] = 1
        count_kmers(seq, 1, counter=counter)
        sum_ = np.sum(list(counter.values()))
        for j, aa in enumerate(aas):
            array[i, j] = counter[aa]/sum_
        i += 1
    return array


aa_human = aa_frequencies('Human', min_length=min_length)
aa_malaria = aa_frequencies('Malaria', min_length=min_length)
aa_cmv = aa_frequencies('CMV', min_length=min_length)

np.savez('data/data.npz', human=aa_human, malaria=aa_malaria, cmv=aa_cmv)
