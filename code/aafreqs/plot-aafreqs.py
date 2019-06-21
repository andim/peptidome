import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

from lib import *

df = counter_to_df(count_kmers_proteome(human, 1), norm=True)
df = df.set_index('seq')
df = df.sort_index()

fig, ax = plt.subplots()
df.plot(kind='bar', ax=ax, legend=False)
ax.set_xlabel('amino acid')
ax.set_ylabel('frequency')
fig.tight_layout()
plt.show()
fig.savefig('main.png')
