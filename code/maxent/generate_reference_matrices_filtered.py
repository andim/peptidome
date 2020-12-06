import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

L = 9
filterlength = 12
seed = 12345

prng = np.random.RandomState(seed=seed)

df = load_proteome_as_df('Human')
df.drop_duplicates(subset=['Sequence'], inplace=True)

seqs = df['Sequence']
train, test = train_test_split(seqs, test_size=0.5, random_state=prng)


def filterseqs(seqs, k, filterlength):
    counter = count_kmers_iterable(seqs, filterlength, clean=True)
    count_df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
    count_series = count_df.T.squeeze()
    filtered = np.asarray(count_series.index.str[:k])
    return filtered


for label, data in [('train', train), ('test', test)]:
    matrix = kmers_to_matrix(filterseqs(data, k=L, filterlength=filterlength))
    np.savetxt('data/filtered_%s_matrix_L%i.csv.gz'%(label,L), matrix, fmt='%i')
