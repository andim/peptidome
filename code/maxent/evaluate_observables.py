import os.path
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

matrix = load_matrix(snakemake.input[0])
fi = frequencies(matrix, num_symbols=naminoacids)
fij = pair_frequencies(matrix, num_symbols=naminoacids, fi=fi)
cij = compute_covariance_matrix(fi, fij)
fijk = triplet_frequencies(matrix, num_symbols=naminoacids)
cijk = compute_cijk(fijk, fij, fi)
fold_ijk = compute_fold_ijk(fijk, fi)
np.savez_compressed(snakemake.output[0],
         fi=fi, fij=fij, cij=cij,
         cijk=cijk, fijk=fijk, fold_ijk=fold_ijk)
