import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from lib import *

proteinatlas = pd.read_csv('../../data/proteinatlas.tsv.zip', sep='\t')
tissues = [c for c in proteinatlas.columns if 'Tissue' in c]
proteinatlas.fillna(0, inplace=True)

humanproteome = load_proteome_as_df('Human')
humanproteome = humanproteome[~(humanproteome['Gene'] == '')]
humanproteome = humanproteome.set_index('Gene')
humanproteome = humanproteome.squeeze()

missing = set(proteinatlas['Gene'].unique()) - set(humanproteome.index)

proteinatlas = proteinatlas[~proteinatlas['Gene'].isin(missing)]
proteinatlas[tissues] = proteinatlas[tissues].div(proteinatlas[tissues].sum(axis=0), axis=1)

def generator(N, k, p):
    counter = 0
    while counter < N:
        sequence = humanproteome.loc[np.random.choice(proteinatlas['Gene'], p=p)]
        if len(sequence)>k:
            startindex = np.random.randint(0, len(sequence)-k)
            counter += 1
            yield sequence[startindex:startindex+k]

freqs = {}
df = Counter(proteome_path('Human'), 1).to_df()
df.set_index('seq', inplace=True)
df = df.squeeze()
freqs['All'] = df
for tissue in tissues:
    df = Counter(generator(5000, 9, np.array(proteinatlas[tissue])), 1).to_df()
    df.set_index('seq', inplace=True)
    df = df.squeeze()
    freqs[tissue] = df
pd.DataFrame(freqs).to_csv('freqs.csv', index=True)
