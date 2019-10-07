import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from lib import *

# Also stores indices of proteins within the proteome to allow their reidentification

min_length = 100

aas =  aminoacids
def aa_frequencies(proteome, min_length=1):
    seqs = []
    indices = []
    for i, (h, seq) in enumerate(fasta_iter(proteome)):
        for sym in 'XUBZ':
            seq = seq.replace(sym, '')
        if len(seq) < min_length:
            continue
        seqs.append(seq)
        indices.append(i)
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
    return array, indices

aa_human, indices_human = aa_frequencies(proteome_path('Human'), min_length=min_length)
aa_malaria, indices_malaria = aa_frequencies(proteome_path('Malaria'), min_length=min_length)
aa_cmv, indices_cmv = aa_frequencies(proteome_path('CMV'), min_length=min_length)
aa_viruses, indices_viruses = aa_frequencies(datadir+'human-viruses-uniref90.fasta', min_length=min_length)

np.savez('data/data.npz',
        human=aa_human, malaria=aa_malaria, cmv=aa_cmv, viruses=aa_viruses,
        indices_human=indices_human,
        indices_malaria=indices_malaria,
        indices_cmv=indices_cmv,
        indices_viruses=indices_viruses
        )
