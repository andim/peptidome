import os.path
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

L = 9

for dataset in ['train', 'test', 'model', 'model_ncov', 'model_nskew', 'model_nskewdiag']:
    print(dataset)
    path = 'data/%s_observables.npz'%dataset
    if not os.path.exists(path):
        matrix = load_matrix('data/%s_matrix_L%i.csv.gz'%(dataset, L))
        fi = frequencies(matrix, num_symbols=naminoacids)
        fij = pair_frequencies(matrix, num_symbols=naminoacids, fi=fi)
        cij = compute_covariance_matrix(fi, fij)
        fijk = triplet_frequencies(matrix, num_symbols=naminoacids)
        cijk = compute_cijk(fijk, fij, fi)
        fold_ijk = compute_fold_ijk(fijk, fi)
        np.savez_compressed(path,
                 fi=fi, fij=fij, cij=cij,
                 cijk=cijk, fijk=fijk, fold_ijk=fold_ijk)
