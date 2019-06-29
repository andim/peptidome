import sys
sys.path.append('..')
import itertools
import numpy as np
import pandas as pd
import scipy.stats

from lib import *

k = 9

df = pd.read_csv('data/freq1.csv')
df = df.set_index('seq')
df.sort_index(inplace=True)
charlogp = np.log10(df['freq']).to_dict()

loglikelihood = lambda seq, k: loglikelihood_independent(seq, charlogp, k=k)

#phuman = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1) ])
#np.save('data/loglikelihood-k9-human.npy', phuman)

prandom = np.array([loglikelihood(''.join(np.random.choice(np.array(list(aminoacids)), size=k, p=np.asarray(df['freq']))), k) for i in range(1000000)])
np.save('data/loglikelihood-k9-random.npy', prandom)

#p4mers = np.array([loglikelihood(''.join(s), 4) for s in itertools.product(aminoacids, repeat=4)])
#np.save('data/loglikelihood-k4-all.npy', p4mers)

