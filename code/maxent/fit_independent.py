import itertools, json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *


k = int(snakemake.wildcards.k)
q = len(aminoacids)
pseudocount = 1.0

matrix = load_matrix(snakemake.input[0])

fi = frequencies(matrix, num_symbols=q, pseudocount=pseudocount)
f = fi.mean(axis=0)
model_matrix = np.random.choice(np.arange(0, q, 1), size=matrix.shape, p=f)
np.savetxt(snakemake.output[0], model_matrix, fmt='%i')
np.savez(snakemake.output[1], f=f)
