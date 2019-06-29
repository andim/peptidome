---
layout: post
title: Jensen-Shannon divergence between doublet frequencies in different proteomes
---

No statistical modeling, only direct sampling. Are there clusters?

{% include image-gallery.html filter="jsd_doublet/" %}

### Code 
#### dkls-comparison.ipynb


{::nomarkdown}
{% jupyter_notebook "/code/jsd_doublet/dkls-comparison.ipynb"%}
{:/nomarkdown}
#### runkl.py

```python
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from lib import *

proteomes = load_proteomes()
human = datadir + proteomes.ix['Human']['path']


df = counter_to_df(count_kmers_proteome(human, 1), norm=True)
df2 = counter_to_df(count_kmers_proteome(human, 2), norm=True)


round_to_n = lambda x, n: round(x, -int(np.floor(np.log10(x))) + (n - 1))

def dkl(path, name): 
    dfp = counter_to_df(count_kmers_proteome(path, 1), norm=True)
    dfmerged = pd.merge(df, dfp, on='seq', suffixes=['_human', '_pathogen'])

    h = np.asarray(dfmerged['freq_human'])
    p = np.asarray(dfmerged['freq_pathogen'])

    dkl = scipy.stats.entropy(p, qk=h, base=2)

    # calculate dkl between twomers
    dfp = counter_to_df(count_kmers_proteome(path, 2), norm=True)
    dfmerged = pd.merge(df2, dfp, on='seq', suffixes=['_human', '_pathogen'])

    h = np.asarray(dfmerged['freq_human'])
    p = np.asarray(dfmerged['freq_pathogen'])

    dkl2 = scipy.stats.entropy(p, qk=h, base=2)

    print('%s&%s&%s\\\\'%(name, round_to_n(dkl, 2), round_to_n(dkl2, 2)))


dkl(datadir+'human-viruses-uniref90_nohiv.fasta', 'viruses')

pathogenproteomes = proteomes[~proteomes.index.isin(['Human'])]
for name, row in pathogenproteomes.iterrows():
    path = datadir + row['path']
    dkl(path, name)



```
#### run.py

```python
import sys, copy
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import sklearn.manifold
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from lib import *
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

proteomes = load_proteomes()
human = datadir + proteomes.loc['Human']['path']
names = ['Human', 'Mouse', 'Vaccinia', 'InfluenzaB', 'InfluenzaA', 'CMV', 'HCV', 'HSV1',
       'DENV', 'HIV', 'EBV', 'Ebola', 'Ecoli', 'Tuberculosis', 'Listeria',
       'Burkholderia', 'Meningococcus', 'StrepA', 'Hpylori',
       'Lyme', 'Tetanus', 'Leprosy', 'Malaria', 'Chagas',
       'OnchocercaVolvulus']

dfs = {name: counter_to_df(count_kmers_proteome(datadir + proteomes.loc[name]['path'], 2), norm=False, clean=True) for name in proteomes.index}

N = len(names)
distances_uniform = np.zeros(N)
distances = np.zeros((N, N))
for i, namei in enumerate(names):
    df1 = dfs[namei]
    f1 = np.asarray(list(df1['count']))
    f1 += np.ones_like(f1)
    f2 = np.ones_like(f1)
    distances_uniform[i] = calc_jsd(f1, f2)
    for j, namej in enumerate(names):
#        if i == j:
#            f1 = np.asarray(df1['count'])
#            f1 += np.ones_like(f1)
#            f2 = np.ones_like(f1)
        df2 = dfs[namej]
        dfm = pd.merge(df1, df2, on='seq', suffixes=['_1', '_2'])
        f1, f2 = np.asarray(dfm['count_1']), np.asarray(dfm['count_2'])
        f1 += np.ones_like(f1)
        f2 += np.ones_like(f2)
        distances[i, j] = calc_jsd(f1, f2, base=2)

fullnames = list(proteomes.loc[names]['fullname'])

df = pd.DataFrame(distances, index=names, columns=names, copy=True)
df['Uniform'] = distances_uniform
df = df.append(pd.Series(distances_uniform, name='Uniform', index=names))
df.iloc[-1, -1] = 0.0
df.to_csv('data/jsds.csv', index=True)

```
#### plot.py

```python
import sys, copy
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import sklearn.manifold
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
plt.style.use('../peptidome.mplstyle')

from lib import *

df = pd.read_csv('data/jsds.csv', index_col=0)
print(df)
names = ['Human', 'Mouse', 'Vaccinia', 'InfluenzaB', 'InfluenzaA', 'CMV', 'HCV', 'HSV1',
       'DENV', 'HIV', 'EBV', 'Ebola', 'Ecoli', 'Tuberculosis', 'Listeria',
       'Burkholderia', 'Meningococcus', 'StrepA', 'Hpylori',
       'Lyme', 'Tetanus', 'Leprosy', 'Malaria', 'Chagas',
       'OnchocercaVolvulus']


colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
type_to_color = {'virus' : colors[0],
                 'bacterium' : colors[1],
                 'parasite' : colors[2],
                 'vertebrate' : colors[3]
                }
typecolors = np.array([type_to_color[proteomes.loc[name]['type']] for  name in names])


ys = df['Uniform'][names]
xs = df['Human'][names]
fig, ax = plt.subplots(figsize=(3.42, 3.42))
ax.scatter(xs, ys, color=typecolors)
#offset=0.00
#for i, name in enumerate(names):
#    ax.text(xs[i], ys[i]+offset, name, ha='center', color=typecolors[i])
ax.plot([0, 0.15], [0, 0.15], 'k-')
ax.set_xlabel('JSD(proteome, human)')
ax.set_ylabel('JSD(proteome, uniform)')
ax.set_aspect('equal')
ax.set_xlim(-0.0, 0.15)
ax.set_ylim(-0.0, 0.15)
for i, (type_, color) in enumerate(type_to_color.items()):
    ax.text(0.8, 0.2-i*0.04, type_, color=color, transform=ax.transAxes)
#sns.despine(fig)
fig.tight_layout()
fig.savefig('main.png')
plt.show()

```
