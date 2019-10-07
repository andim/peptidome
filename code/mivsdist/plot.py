import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

for species in ['Human', 'Mouse', 'Yeast', 'Viruses']:
    fig, ax = plt.subplots()
    df = pd.read_csv('data/mutualinformation-%s.csv'%species)
    ax.errorbar(df['gaps']+1, df['mutualinformation'],
            2*df['mutualinformation_std'], label='data')
    ax.plot(df['gaps']+1, df['shuffledmutualinformation'], label='shuffled')
    ax.legend()
    ax.set_ylim(0.0, 0.0145)
    ax.set_xlim(0.95, 201.0)
    ax.set_xscale('log')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Mutual information in bits')
    fig.tight_layout()
    if species == 'Human':
        fig.savefig('main.png')
        plt.show()
    else:
        fig.savefig('%s.png'%species)
