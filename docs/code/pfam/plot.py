import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

fig, ax = plt.subplots()
df = pd.read_csv('../mivsdist/data/mutualinformation-Human.csv')
ax.plot(df['gaps']+1, df['mutualinformation'], label='all')
df = pd.read_csv('data/mutualinformation_nozf.csv')
ax.plot(df['gaps']+1, df['mutualinformation'], label='excluding zincfinger')
df = pd.read_csv('data/mutualinformation_uniquedomains.csv')
ax.plot(df['gaps']+1, df['mutualinformation'], label='unique domains')
#df = pd.read_csv('data/mutualinformation_replaced.csv')
#ax.plot(df['gaps']+1, df['mutualinformation'], label='replaced')
ax.legend()
ax.set_ylim(0.0)
ax.set_xlim(1.0, 200.0)
ax.set_xscale('log')
ax.set_xlabel('Distance')
ax.set_ylabel('Mutual information in bits')
fig.tight_layout()
fig.savefig('main.png')
plt.show()
