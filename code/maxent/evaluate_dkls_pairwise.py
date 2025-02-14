import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *


k = 9
model = 'nskewfcov'
q = naminoacids

for proteome in ['Human','Mouse','Chicken','Zebrafish','Humanviruses','Malaria','Tuberculosis','Listeria','StrepA']:

    print(f'For {proteome}')
    entropy = pd.read_csv(f'data/{proteome}_{model}_k{k}_entropy.csv',header=None, index_col=0)
    params = np.load(f'data/{proteome}_{model}_k{k}_params.npz')
    matrix = load_matrix(f'data/{proteome}_{model}_k{k}_matrix.csv.gz')
    energy = make_energy(params)
    energies = np.array([energy(x) for x in matrix])


    df_data = []

    for reference in ['Human','Mouse','Chicken','Zebrafish','Humanviruses','Malaria','Tuberculosis','Listeria','StrepA']:

        print(f'Reference: {reference}')
        entropy_reference = pd.read_csv(f'data/{reference}_{model}_k{k}_entropy.csv',header=None, index_col=0)
        params_reference = np.load(f'data/{reference}_{model}_k{k}_params.npz')

        
        energy_reference = make_energy(params_reference)
        energies_reference = np.array([energy_reference(x) for x in matrix])
        DKL = float(entropy.loc['F']) - np.mean(energies) + np.mean(energies_reference) - float(entropy_reference.loc['F'])

        df_data.append({
            'reference': reference,
            'DKL': DKL
        })

    df = pd.DataFrame(df_data)
    df.to_csv(f'data/{proteome}_{model}_k{k}_dkls_all.csv')
