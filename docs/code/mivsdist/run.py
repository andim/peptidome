import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from lib import *

#name = 'human'
#proteome = human
#name = 'mouse'
#proteome = mouse
#name = 'yeast'
#proteome = yeast
#name = 'malaria'
#proteome = malaria
name = 'viruses'
proteome = datadir + 'human-viruses-uniref90_nohiv.fasta'

entropyestimator = entropy_grassberger

df1 = counter_to_df(count_kmers_proteome(proteome, k=1))
df1 = df1[~df1['seq'].str.contains('U|B|X|Z')]
df1.set_index('seq', inplace=True)
entropy1 = entropyestimator(df1['freq'], base=2)

meanabsfoldchanges = []
mutualinformations = []
gaps = np.arange(0, 201, 1)
for gap in gaps:
    df2 = counter_to_df(count_kmers_proteome(proteome, k=2, gap=gap), norm=False)
    df2 = df2[~df2['seq'].str.contains('U|B|X|Z')]
    entropy2 = entropyestimator(df2['count'], base=2)
    df = strcolumn_to_charcolumns(df2, 'seq')
    e1 = entropyestimator(df.groupby('aa1').agg(np.sum)['count'], base=2)
    e2 = entropyestimator(df.groupby('aa2').agg(np.sum)['count'], base=2)
    mi = e1 + e2 - entropy2
    print(gap, mi)
    mutualinformations.append(mi)

df = pd.DataFrame.from_dict(dict(gaps=gaps, mutualinformation=mutualinformations))
df.to_csv('data/mutualinformation-%s.csv'%name)
