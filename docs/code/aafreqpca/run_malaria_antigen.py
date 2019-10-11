import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from lib import *

df_t = load_iedb_tcellepitopes(human_only=True)

iedbname = 'Plasmodium falciparum'
epitope_proteins = [s.split('/')[-1] for s in df_t[df_t['Epitope', 'Organism Name'] == iedbname]['Epitope', 'Parent Protein IRI'].unique() if type(s) == type('')]

epitope_proteins_indices = [i
                            for i, (h, seq) in enumerate(fasta_iter(proteome_path('Malaria')))
                            if iscontained(h, epitope_proteins)]
pd.Series(epitope_proteins_indices).to_csv('data/malaria_antigens.csv', index=False, header=False)
