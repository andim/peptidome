import itertools, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

from numba import njit

L = 9
pseudocount = 1.0

matrix = load_matrix('data/train_matrix_L%i.csv.gz'%L)

flat = np.random.choice(np.arange(0, len(aminoacids), 1), size=matrix.shape)
np.savetxt('data/flat_matrix_L%i.csv.gz'%L, flat, fmt='%i')

fi = frequencies(matrix, num_symbols=len(aminoacids), pseudocount=pseudocount)
f = fi.mean(axis=0)
independent = np.random.choice(np.arange(0, 20, 1), size=matrix.shape, p=f)
np.savetxt('data/independent_matrix_L%i.csv.gz'%L, independent, fmt='%i')
