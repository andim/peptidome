import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

filtering = 'notop20'
seqs = list(pd.read_csv('data/human_%s.csv'%filtering)['Sequence'])

mutualinformation = []
mutualinformation_std = []
gaps = np.arange(0, 201, 1)
for gap in gaps:
    df2 = Counter(seqs, k=2, gap=gap).to_df(norm=False, clean=True)
    mi = calc_mi(df2)
    mutualinformation.append(mi)
    mi_std = calc_mi_std(seqs, gap)[1]
    mutualinformation_std.append(mi_std)
    print(gap, mi, mi_std)

df = pd.DataFrame.from_dict(dict(gaps=gaps, mutualinformation=mutualinformation,
                                 mutualinformation_std=mutualinformation_std))
df.to_csv('data/mutualinformation_%s.csv'%filtering)
