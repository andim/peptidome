import numpy as np
import pandas as pd


import sys
sys.path.append('..')
sys.path.append('../../../code')
from lib import *


datadir = '../../../code/maxent/data'

def mat_to_dist(A, reference, size=100000):
    sample = matrix_to_kmers(A[np.random.choice(A.shape[0], size=size, replace=False), :])
    ns = nndist_hamming_distribution(sample, reference)
    return ns

if __name__ == "__main__":
    
    keys = ['Humanviruses_nskewfcov','Human_nskewfcov', 'Malaria_nskewfcov']
    for reference_key in keys:

        ref_matrix = load_matrix(f'{datadir}/{reference_key}_k9_matrix.csv.gz')
        reference = set(matrix_to_kmers(ref_matrix))

        for sample_key in keys:
            if reference_key==reference_key:
                continue

            sample_matrix = load_matrix(f'{datadir}/{sample_key}_k9_matrix.csv.gz')
            n_values = mat_to_dist(sample_matrix, reference)
            ps_values = n_values / np.sum(n_values)
            df = pd.DataFrame([n_values,ps_values],columns=['0','1','2','3+'],index=['N','P'])
            df.to_csv(f'{datadir}/{sample_key}_distance_to_{reference_key}.csv')



        samples = ['Humanviruses_train', 'Human_train', 'Malaria_train']
        for sample_key in samples:

            sample_matrix = load_matrix(f'{datadir}/{sample_key}_k9_matrix.csv.gz')

            n_values = mat_to_dist(sample_matrix, reference)

            ps_values = n_values / np.sum(n_values)
            df = pd.DataFrame([n_values,ps_values],columns=['0','1','2','3+'],index=['N','P'])
            df.to_csv(f'{datadir}/{sample_key}_distance_to_{reference_key}.csv')


        sample_matrix = np.random.choice(np.arange(0, len(aminoacids), 1), size=(1000000, 9))

        n_values = mat_to_dist(sample_matrix, reference)

        ps_values = n_values / np.sum(n_values)
        df = pd.DataFrame([n_values,ps_values],columns=['0','1','2','3+'],index=['N','P'])
        df.to_csv(f'{datadir}/uniform_distance_to_{reference_key}.csv')