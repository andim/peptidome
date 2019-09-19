import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from lib import *

min_length = 200

aas =  aminoacids
def aa_frequencies(proteome, min_length=1):
    seqs = []
    for h, seq in fasta_iter(proteome):
        for sym in 'XUBZ':
            seq = seq.replace(sym, '')
        if len(seq) < min_length:
            continue
        seqs.append(seq)
    array = np.zeros((len(seqs), len(aas)))
    i = 0
    for seq in seqs:
        counter = {}
        for aa in aas:
            counter[aa] = 1
        count_kmers(seq, 1, counter=counter)
        sum_ = np.sum(list(counter.values()))
        for j, aa in enumerate(aas):
            array[i, j] = counter[aa]/sum_
        i += 1
    return array

aa_human = aa_frequencies(proteome_path('Human'), min_length=min_length)
aa_malaria = aa_frequencies(proteome_path('Malaria'), min_length=min_length)
aa_cmv = aa_frequencies(proteome_path('CMV'), min_length=min_length)
aa_viruses = aa_frequencies(datadir+'human-viruses-uniref90.fasta', min_length=min_length)

np.savez('data/data.npz', human=aa_human, malaria=aa_malaria, cmv=aa_cmv, viruses=aa_viruses)
