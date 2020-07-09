import itertools, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

for dataset in ['train', 'test', 'model']:
    matrix = np.loadtxt('data/%s_matrix.csv.gz'%dataset).astype(int)
    fi = frequencies(matrix, num_symbols=naminoacids)
    fij = pair_frequencies(matrix, num_symbols=naminoacids, fi=fi)
    cij = compute_covariance_matrix(fi, fij)
    fijk = triplet_frequencies(matrix, num_symbols=naminoacids)
    cijk = compute_cijk(fijk, fij, fi)
    fold_ijk = compute_fold_ijk(fijk, fi)
    np.savez('data/%s_observables.npz'%dataset, fi=fi, cij=cij, cijk=cijk, fold_ijk=fold_ijk)
