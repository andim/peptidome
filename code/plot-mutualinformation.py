import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt

from lib import *
mishuffled = 0.0014149224105342029

df = pd.read_csv('mutualinformation-human.csv')
fig = plt.figure(figsize=(5, 3))
plt.plot(df['gaps'], df['mutualinformation'], label='data')
plt.axhline(mishuffled, color='k', label='shuffled within proteins')
plt.legend()
plt.ylim(0.0)
plt.xlim(0.0)
plt.xlabel('Gap')
plt.ylabel('Mutual information in bits')
fig.tight_layout()
fig.savefig('plots/doublet-mutualinformation.png', dpi=300)
plt.show()
