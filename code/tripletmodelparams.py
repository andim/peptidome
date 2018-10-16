# Calculate parameters for triplet model

import numpy as np
import pandas as pd

import json

from lib import *

name = 'malaria'
proteome = malaria

df = counter_to_df(count_kmers_proteome(proteome, 1), norm=True)
df = df.set_index('seq')
charlogp = np.log10(df['freq']).to_dict()

df1 = counter_to_df(count_kmers_proteome(proteome, 2), norm=False)
strcolumn_to_charcolumns(df1, 'seq')
count = df1.pivot(columns='aa1', index='aa2')['count']
count /= np.sum(count, axis=0)
count[count.isna()] = 1e-10
doubletlogp = np.log10(count).to_dict()

df2 = counter_to_df(count_kmers_proteome(proteome, 3), norm=False)
df2['aa12'] = [s[:2] for s in df2['seq']]
df2['aa3'] = [s[2] for s in df2['seq']]
count = df2.pivot(columns='aa12', index='aa3')['count']
count /= np.sum(count, axis=0)
count[count.isna()] = 1e-10
tripletlogp = np.log10(count).to_dict()

modelparams = dict(charlogp=charlogp, doubletlogp=doubletlogp, tripletlogp=tripletlogp)
with open('../data/triplet-%s.json'%name, 'w') as f:
    json.dump(modelparams, f, indent=4)
