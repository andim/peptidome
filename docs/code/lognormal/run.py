import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import scipy.stats

from lib import *

for k in [1, 4]:
    counter = count_kmers_proteome(human, k)
    df = counter_to_df(counter, norm=False)
    df = df[~df['seq'].str.contains('U|B|X|Z')]
    df['freq'] = df['count'] / np.sum(df['count'])
    df.to_csv('data/freq%g.csv'%k)
