import itertools, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
import clib
from base import *

output = True
aas_arr = np.array(list(aminoacids))
N = 9

proteome = proteome_path('Human')

seed = 1234
prng = np.random.RandomState(seed)

seqs = [s for s in fasta_iter(proteome, returnheader=False)]
train, test = train_test_split(seqs, test_size=0.5, random_state=prng)
#train, test = seqs, seqs

# evaluate empirical observables for fitting
#df1 = pseudocount_f1(train)
#df2s = [pseudocount_f2(train, 2, gap, df1)  for gap in range(0, N-1)]
df1 = count(train, 1)
df2s = [count(train, 2, gap=gap)  for gap in range(0, N-1)]
print('fit')
h, Jk = fit_potts(df1, df2s, nmcmc=1e6, niter=10, epsilon=0.1, prng=prng, output=output)
print(Jk)

save('Human_N%g'%N, h, Jk)
np.savez('data/Human_N%g.npz'%N, h=h, Jk=Jk)
