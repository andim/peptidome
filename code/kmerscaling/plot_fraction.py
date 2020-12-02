import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.stats
import sklearn.decomposition
import sklearn.manifold
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

df = load_proteome_as_df('Human')
print('multiple seqs', len(df['Sequence'])-len(df['Sequence'].unique()))
df.drop_duplicates(subset=['Sequence'], inplace=True)

def counter_to_series(counter):
    count_df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
    count_series = count_df.T.squeeze()
    return count_series

ks = np.arange(1, 20)
fractions = []
for k in ks:
    counter = count_kmers_iterable(df['Sequence'], k)
    count_series = counter_to_series(counter)
    if k == 1:
        n = np.sum(count_series)
    fraction = np.sum(count_series[count_series>1])/np.sum(count_series)
    print(k, fraction)
    fractions.append(fraction)

p0 = 1/20**ks

fig, ax = plt.subplots()
ax.plot(ks, fractions, 'o')
ax.plot(ks, 1 - p0*n/(np.exp(p0*n)-1), label=r'$1-\frac{N p_0}{e^{N p_0}-1}$')
ax.legend()
ax.set_ylabel('fraction of kmers seen\nmore than once')
ax.set_ylim(0.0)
ax.set_xlabel('k')
fig.tight_layout()
fig.savefig('kmer_multiples.png')
plt.show()


