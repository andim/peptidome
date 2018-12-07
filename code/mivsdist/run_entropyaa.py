import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from lib import *

name = 'human'
proteome = human
#name = 'mouse'
#proteome = mouse
#name = 'yeast'
#proteome = yeast
#name = 'malaria'
#proteome = malaria

entropyestimator = entropy_grassberger

aa_counts = np.zeros((20, 100))
for seq in fasta_iter(proteome, returnheader=False):
    try:
        seq = map_aatonumber(seq[:aa_counts.shape[1]])
    except KeyError:
        continue
    for pos, number in enumerate(seq):
        aa_counts[number, pos] += 1

entropies = [entropyestimator(aa_counts[:, i], base=2) for i in range(aa_counts.shape[1])]
df = pd.DataFrame.from_dict(dict(entropy=entropies,
        position=np.arange(1, aa_counts.shape[1]+1)))
df.to_csv('data/entropyaa-%s.csv'%name)
