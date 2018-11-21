import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import scipy.stats

from lib import *

k = 9

df = pd.read_csv('data/freq1.csv')
df = df.set_index('seq')
charlogp = np.log10(df['freq']).to_dict()

loglikelihood = lambda seq, k: loglikelihood_independent(seq, charlogp, k=k)

phuman = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(human) for i in range(len(seq)-k+1) ])
np.save('data/loglikelihood-k9-human.npy', phuman)
