import numpy as np
import pandas as pd


import sys
sys.path.append('..')
sys.path.append('../../../code')
from lib import *


datadir = '../../../code/maxent/data'
figdatadir = './'

reference = set(count_kmers_proteome(human, 9, clean=True))

def mat_to_dist(A, size=100000):
    sample = matrix_to_kmers(A[np.random.choice(A.shape[0], size=size, replace=False), :])
    ns = nndist_hamming_distribution(sample, reference)
    return ns

if __name__ == "__main__":
    data_key = sys.argv[1]
    sample_matrix = load_matrix(f'{datadir}/{data_key}_k9_matrix.csv.gz')
    n_values = mat_to_dist(sample_matrix)
    ps_values = n_values / np.sum(n_values)
    df = pd.DataFrame([n_values,ps_values],columns=['0','1','2','3+'],index=['N','P'])
    df.to_csv(f'{datadir}/{data_key}_distance_to_self.csv')