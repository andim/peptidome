---
layout: post
title: Entropy as a function of position
---

The entropy of the amino acids is more restricted for the first few amino acids within a protein.


{% include post-image-gallery.html filter="entropyvsposition/" %}

### Code 
#### run.py

```python
import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from lib import *

name = sys.argv[1]
proteome = proteome_path(name)

entropyestimator = entropy_grassberger

aa_counts = np.zeros((20, 100))
for seq in fasta_iter(proteome, returnheader=False):
    try:
        seq = map_aatonumber(seq[:aa_counts.shape[1]])
    except KeyError:
        continue
    for pos, number in enumerate(seq):
        aa_counts[number, pos] += 1

print(aa_counts)

entropies = [entropyestimator(aa_counts[:, i], base=2) for i in range(aa_counts.shape[1])]
df = pd.DataFrame.from_dict(dict(entropy=entropies,
        position=np.arange(1, aa_counts.shape[1]+1)))
df.to_csv('data/entropyaa-%s.csv'%name)

```
#### plot.py

```python
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import brokenaxes
plt.style.use('../peptidome.mplstyle')

name = sys.argv[1]

fig = plt.figure(figsize=(3.5, 2.5))
if name == 'Human':
    ax = brokenaxes.brokenaxes(ylims=((0.0, 0.2), (3.7, 4.22)), bottom=0.2, left=0.2)
else:
    ax = fig.add_subplot(111)
df = pd.read_csv('data/entropyaa-%s.csv'%name)
#ax = brokenaxes.brokenaxes(ylims=((2.2, 2.4), (3.7, 4.22)), bottom=0.2, left=0.2)
#df = pd.read_csv('data/entropyaa-viruses.csv')
ax.plot(df['position'], df['entropy'], 'o')
print(df['position'])
ax.set_xlim(0, 51)
ax.set_ylabel('entropy in bits', labelpad=6)
ax.set_xlabel('position', labelpad=5)
fig.tight_layout()
if name == 'Human':
    fig.savefig('../../paper/images/entropyaa.pdf')
    fig.savefig('main.png')
plt.show()

```
