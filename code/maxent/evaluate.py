import os.path
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

for dataset in ['train', 'test', 'model', 'model_global']:
    print(dataset)
    path = 'data/%s_observables.npz'%dataset
    if not os.path.exists(path):
        matrix = np.loadtxt('data/%s_matrix.csv.gz'%dataset, dtype=np.int64)
        fi = frequencies(matrix, num_symbols=naminoacids)
        fij = pair_frequencies(matrix, num_symbols=naminoacids, fi=fi)
        cij = compute_covariance_matrix(fi, fij)
        fijk = triplet_frequencies(matrix, num_symbols=naminoacids)
        cijk = compute_cijk(fijk, fij, fi)
        fold_ijk = compute_fold_ijk(fijk, fi)
        np.savez_compressed(path,
                 fi=fi, fij=fij, cij=cij,
                 cijk=cijk, fijk=fijk, fold_ijk=fold_ijk)
