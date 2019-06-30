---
layout: post
title: k-mer entropy
---

Entropy of the kmer distributions for different proteomes.

{% include image.html
   url="/code/kmerentropy/main.png"
   description="Entropy of the kmer distributions for various proteomes. The black line shows the entropy for a uniform random distribution (maximal possible entropy)."
%}



{% include image-gallery.html filter="kmerentropy/" %}

### Code 
#### kmerentropy.ipynb



```python
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.stats
import sklearn.decomposition
import sklearn.manifold
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('../custom.mplstyle')

from lib import *
```


```python
counters = [count_kmers_proteome(human, k, clean=True) for k in range(1, 10)]
```


```python
entropies = np.array([entropy(normalize(c), base=2) for i, c in enumerate(counters)])

```


```python
entropies_gb = np.array([entropy_grassberger(np.asarray(list(c.values())), base=2) for i, c in enumerate(counters)])

```


```python
entropies_nsb = np.array([entropy_nsb(np.asarray(list(c.values())), base=2) for i, c in enumerate(counters)])

```


```python
def coincidence_prob(counter, size=int(1e7)):
    s = np.random.choice(np.asarray(list(counter.keys())), size=size,
                 p=normalize(counter))
    return np.mean(s[:len(s)//2] == s[len(s)//2:])
```


```python
min_entropies = [-np.log2(coincidence_prob(c)) for c in counters]
```

    /home/amayer/.conda/envs/py3/lib/python3.6/site-packages/ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in log2
      """Entry point for launching an IPython kernel.



```python
ks = np.arange(1, len(entropies)+1)
plt.plot(ks, np.log2(20)*ks, label='upper bound')
plt.plot(ks, entropies, label='MLE')
plt.plot(ks, entropies_gb, label='GB')
plt.plot(ks, entropies_nsb, label='NSB')
plt.plot(ks, min_entropies, label='lower bound')
plt.xlabel('k')
plt.ylim(0, 30)
plt.ylabel('Entropy in bits')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f619045a780>




![png](notebook_files/kmerentropy_7_1.png)



```python
plt.plot(ks, np.ones_like(ks)*np.log2(20), label='upper bound')
plt.plot(ks, entropies/ks, label='MLE')
plt.plot(ks, entropies_gb/ks, label='GB')
plt.plot(ks, entropies_nsb/ks, label='NSB')
plt.plot(ks, min_entropies/ks, label='lower bound')
plt.xlabel('k')
plt.ylabel('Entropy in bits')
plt.legend()
plt.ylim(4.1, 4.2)
```




    (4.1, 4.2)




![png](notebook_files/kmerentropy_8_1.png)



```python

```
#### run.py

```python
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib import *

ks = np.arange(1, 6)

proteomes = load_proteomes()
dfdict = {'k' : ks}
for name in ['Human', 'Yeast', 'Ecoli']:
    if name == 'Viruses':
        proteome = datadir+'human-viruses-uniref90_nohiv.fasta'
    else:
        proteome = datadir + proteomes.ix[name]['path']
    entropies = []
    for k in ks:
        df = counter_to_df(count_kmers_proteome(proteome, k=k), norm=False)
        entropies.append(entropy_grassberger(df['count'], base=2))
    dfdict[name] = entropies

df = pd.DataFrame.from_dict(dfdict)
print(df)
df.to_csv('data/entropy.csv', index=False)

```
#### plot.py

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

df = pd.read_csv('data/entropy.csv')

fig, ax = plt.subplots()
for column in df.columns[1:4]:
    print(df[column])
    ax.plot(df['k'], df[column]/df['k'], 'o-', label=column)
ax.axhline(np.log2(20), c='k', label='random')
ax.legend(loc='lower left', ncol=2)
ax.set_xlabel('k')
ax.set_ylabel('entropy of kmers in bit / k');
fig.tight_layout()
fig.savefig('../../paper/images/entropykmer.pdf')
fig.savefig('main.png')
plt.show()

```
