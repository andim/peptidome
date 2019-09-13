import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

name = sys.argv[1]
proteome = proteome_path(name)

df1 = Counter(proteome, k=1).to_df(norm=True, clean=True)
df1.set_index('seq', inplace=True)

gaps = np.arange(0, 4, 1)
for gap in gaps:
    df2 = counter_to_df(count_kmers_proteome(proteome, k=2, gap=gap), norm=True, clean=True)
    strcolumn_to_charcolumns(df2, 'seq')
    df2['theory'] = [float(df1.loc[s[0]] * df1.loc[s[1]]) for s in df2['seq']]
    df2['fold'] = np.log2(df2['freq']/df2['theory'])
    print(np.abs(df2['fold']).max(), np.abs(df2['fold']).mean())
    dfmat = df2.pivot(columns='aa1', index='aa2')['fold']
    print(dfmat)
    dfmat.to_csv('data/doubletfoldenrichment-%g-%s.csv'%(gap, name))
