import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *

run = int(sys.argv[1])-1
names = ['Human', 'Mouse', 'Yeast', 'Viruses']
name = names[run]
if name == 'Viruses':
    proteome = datadir + 'human-viruses-uniref90_nohiv.fasta'
else:
    proteome = proteome_path(name)
print(name, proteome)

def calc_mi_std(seqs, gap):
    mis = []
    for i in range(30):
        df2 = Counter(random.sample(seqs, int(len(seqs)/2)), k=2, gap=gap).to_df(norm=False, clean=True)
        mis.append(calc_mi(df2))
    return np.std(mis, ddof=1)/2**.5

mutualinformation = []
mutualinformation_std = []
shuffled_mutualinformation = []
gaps = np.arange(0, 201, 1)
for gap in gaps:
    seqs = [s for s in fasta_iter(proteome, returnheader=False)]
    df2 = Counter(seqs, k=2, gap=gap).to_df(norm=False, clean=True)
    mi = calc_mi(df2)
    mutualinformation.append(mi)
    mi_std = calc_mi_std(seqs, gap)
    mutualinformation_std.append(mi_std)

    # calculate shuffled mi
    iterable = scrambled(fasta_iter(proteome, returnheader=False))
    df2 = Counter(iterable, k=2, gap=gap).to_df(norm=False, clean=True)
    shuffledmi = calc_mi(df2)
    shuffled_mutualinformation.append(shuffledmi)
    print(gap, mi, mi_std, shuffledmi)

df = pd.DataFrame.from_dict(dict(gaps=gaps, mutualinformation=mutualinformation,
                                 mutualinformation_std=mutualinformation_std,
                                 shuffledmutualinformation=shuffled_mutualinformation))
df.to_csv('data/mutualinformation-%s.csv'%name)
