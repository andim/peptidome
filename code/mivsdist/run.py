import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *


proteomes = load_proteomes()

name = sys.argv[1]
if name == 'Viruses':
    proteome = datadir + 'human-viruses-uniref90_nohiv.fasta'
else:
    proteome = proteomes.loc[name]['path']
print(name, proteome)

entropyestimator = entropy_grassberger

df1 = counter_to_df(count_kmers_proteome(proteome, k=1), clean=True)
entropy1 = entropyestimator(df1['freq'], base=2)

mutualinformation = []
gaps = np.arange(0, 201, 1)
for gap in gaps:
    df2 = counter_to_df(count_kmers_proteome(proteome, k=2, gap=gap), norm=False, clean=True)
    entropy2 = entropyestimator(df2['count'], base=2)
    df = strcolumn_to_charcolumns(df2, 'seq')
    e1 = entropyestimator(df.groupby('aa1').agg(np.sum)['count'], base=2)
    e2 = entropyestimator(df.groupby('aa2').agg(np.sum)['count'], base=2)
    mi = e1 + e2 - entropy2
    print(gap, mi)
    mutualinformation.append(mi)

df = pd.DataFrame.from_dict(dict(gaps=gaps, mutualinformation=mutualinformation))
df.to_csv('data/mutualinformation-%s.csv'%name)
