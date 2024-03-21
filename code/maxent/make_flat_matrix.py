import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

k = 9

flat = np.random.choice(np.arange(0, len(aminoacids), 1), size=(1000000, k))
np.savetxt(f'data/flat_matrix_k{k}.csv.gz', flat, fmt='%i')
