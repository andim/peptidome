import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

L = 9
seed = 12345

prng = np.random.RandomState(seed=seed)
seqs = np.array(pd.read_csv('../pfam/data/human_nozf.csv')['Sequence'])
train, test = train_test_split(seqs, test_size=0.5, random_state=prng)

for label, data in [('train', train), ('test', test)]:
    matrix = kmers_to_matrix(to_kmers(data, k=L))
    np.savetxt('data/%s_matrix_L%i.csv.gz'%(label,L), matrix, fmt='%i')
