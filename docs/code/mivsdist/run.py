import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

name = sys.argv[1]
if name == 'Viruses':
    proteome = datadir + 'human-viruses-uniref90_nohiv.fasta'
else:
    proteome = proteome_path(name)
print(name, proteome)

entropyestimator = lambda x: entropy_grassberger(x, base=2)

mutualinformation = []
gaps = np.arange(0, 201, 1)
for gap in gaps:
    df2 = Counter(proteome, k=2, gap=gap).to_df(norm=False, clean=True)
    entropy2 = entropyestimator(df2['count'])
    df = strcolumn_to_charcolumns(df2, 'seq')
    e1 = entropyestimator(df.groupby('aa1').agg(np.sum)['count'])
    e2 = entropyestimator(df.groupby('aa2').agg(np.sum)['count'])
    mi = e1 + e2 - entropy2
    print(gap, mi)
    mutualinformation.append(mi)

df = pd.DataFrame.from_dict(dict(gaps=gaps, mutualinformation=mutualinformation))
df.to_csv('data/mutualinformation-%s.csv'%name)
