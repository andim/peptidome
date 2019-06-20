import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib import *

ks = np.arange(1, 6)

proteomes = load_proteomes()
dfdict = {'k' : ks}
for name in ['Human', 'Yeast', 'Ecoli']:
    if name == 'Viruses':
        proteome = datadir+'human-viruses-uniref90_nohiv.fasta'
    else:
        proteome = datadir + proteomes.ix[name]['path']
    entropies = []
    for k in ks:
        df = counter_to_df(count_kmers_proteome(proteome, k=k), norm=False)
        entropies.append(entropy_grassberger(df['count'], base=2))
    dfdict[name] = entropies

df = pd.DataFrame.from_dict(dfdict)
print(df)
df.to_csv('data/entropy.csv', index=False)
