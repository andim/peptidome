import os.path
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

matrix = load_matrix(snakemake.input[0])

params = np.load(snakemake.input[1])
h = params['h']
J = params['J']
J2 = params['J2']
hi = params['hi']
Jij = params['Jij']
@njit
def energy(x):
    return energy_nskewfcov(x, h=h, J=J, J2=J2, hi=hi, Jij=Jij)

F = np.float(pd.read_csv(snakemake.input[2], header=None, index_col=0).loc['F'])

energies = np.array([energy(x) for x in matrix])

np.savez_compressed(snakemake.output[0], energies=energies-F)
