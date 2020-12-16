import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

k = int(snakemake.wildcards.k)
proteome = snakemake.wildcards.proteome

filterlength = 12
seed = 12345

prng = np.random.RandomState(seed=seed)

if proteome == 'Humannozf':
    seqs = np.array(pd.read_csv('../pfam/data/human_nozf.csv')['Sequence'])
else:
    if proteome == 'Humanviruses':
        df = load_unirefproteome_as_df_path(datadir+'human-viruses-uniref90-filtered.fasta')
    else:
        df = load_proteome_as_df(proteome)
    df.drop_duplicates(subset=['Sequence'], inplace=True)
    seqs = df['Sequence']

train, test = train_test_split(seqs, test_size=0.5, random_state=prng)

for i, (label, data) in enumerate([('train', train), ('test', test)]):
    if proteome == 'Humannozf':
        matrix = kmers_to_matrix(to_kmers(data, k=k))
    else:
        matrix = kmers_to_matrix(filter_unique(data, k=k, filterlength=filterlength))
    np.savetxt(snakemake.output[i], matrix, fmt='%i')
