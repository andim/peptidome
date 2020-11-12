---
layout: post
title: Distribution of likelihoods
---

Likelihoods of peptides under different models.


{% include image.html
   url="/code/likelihoodprofile/main.png"
   description="Distribution of likelihoods under a triplet model learned on the human proteomes for random peptides (9-mers) from the human proteome and IEDB T cell epitopes."
%}

{% include post-image-gallery.html filter="likelihoodprofile/" %}

### Code 
#### likelihood-maxent.ipynb

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *
from lib.maxent import *
from lib.plotting import *
plt.style.use('../peptidome.mplstyle')

%load_ext autoreload
%autoreload 2
```


```python
k = 9
ref = 'human'
```


```python
matrix = load_matrix('../maxent/data/test_matrix.csv.gz')
```


```python
params = np.load('../maxent/data/Human_reference_9.npz')
hi = params['hi']
Jij = params['Jij']
```


```python
likelihood_human = np.array([-energy_potts(x, hi, Jij) for x in matrix])
```


```python
with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: -energy_potts(map_aatonumber(seq.upper()), hi, Jij)
likelihoodname = 'maxent'
```


```python
df_ts = load_iedb_tcellepitopes()
```


```python
host = 'Homo sapiens'
#host = 'Mus musculus'
mask_host_host = df_ts['Host', 'Name'].str.contains(host, na=False)
# no host epitopes or epitopes of unknown provenance
mask_epitope_host = df_ts['Epitope', 'Parent Species'].str.contains(host, na=True)
df_ts = df_ts[mask_host_host & (~mask_epitope_host)]
```


```python
# uniquify epitopes by keeping only the first one
df_ts = df_ts.groupby(('Epitope', 'Description')).apply(lambda x: x.iloc[0])
```


```python
plt.hist(df_ts['Epitope', 'Description'].str.len(), bins=np.arange(1, 25, 1)-0.5)
```




    (array([0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 7.0000e+00,
            1.1000e+01, 1.6000e+01, 3.6900e+02, 8.5280e+03, 4.5240e+03,
            3.8900e+02, 2.0290e+03, 8.9100e+02, 1.0910e+03, 4.3846e+04,
            1.4740e+03, 2.2780e+03, 4.0540e+03, 5.3900e+02, 5.2120e+03,
            2.8400e+02, 1.2700e+02, 9.4000e+01]),
     array([ 0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5, 10.5,
            11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5,
            22.5, 23.5]),
     <a list of 23 Patch objects>)




![png](notebook_files/likelihood-maxent_9_1.png)



```python
xmin = likelihood_human.min()
xmax = likelihood_human.max()
hists_pos = {}
hists_neg = {}
lengths = [9, 15]
for length in lengths:
    positive = ~(df_ts['Assay', 'Qualitative Measure'] == 'Negative')
    lengthmask = np.abs(df_ts['Epitope', 'Description'].str.len()-length)<2
    likelihoods_t, weights_t = likelihoods_epitopes(df_ts[positive & lengthmask]['Epitope', 'Description'], loglikelihood, k)
    likelihoods_t_neg, weights_t_neg = likelihoods_epitopes(df_ts[(~positive) & lengthmask]['Epitope', 'Description'], loglikelihood, k)

    bins = list(np.linspace(xmin+3, xmax-8, 25))
    bins.extend([xmin-0.1, xmax+0.1])
    bins = np.array(sorted(bins))
    binmids = (bins[1:]+bins[:-1])*0.5
    hists_pos[length] = np.histogram(likelihoods_t, bins=bins, weights=weights_t)[0]
    hists_neg[length]= np.histogram(likelihoods_t_neg, bins=bins, weights=weights_t_neg)[0]
```


```python
fig, ax = plt.subplots()
ps = [likelihood_human, likelihoods_t, likelihoods_t_neg]#, likelihoods_b]
labels = ['Human proteins', 'IEDB$^+$ T cell epitopes', 'IEDB$^-$ T cell epitopes']#, 'B epitopes']
weights = [np.ones(len(likelihood_human)), weights_t, weights_t_neg]#, weights_b]

plot_histograms(ps, labels, weights=weights, xmin=xmin, xmax=xmax, ax=ax, nbins=30)
ax.set_xlim(xmin, xmax)
ax.set_ylabel('Frequency')
ax.set_xlabel('Likelihood')
ax.set_yscale('log')
ax.legend(title='Peptide', loc='lower center')
fig.tight_layout()
```


![png](notebook_files/likelihood-maxent_11_0.png)



```python
fig, ax = plt.subplots()
for length in lengths:
    hist_pos, hist_neg = hists_pos[length], hists_neg[length]
    plot_proportion(binmids, hist_pos, hist_pos+hist_neg, ls='-', marker='.', ax=ax, label=length)
ax.set_xlabel('Likelihood')
ax.set_ylabel('Fraction positive')
ax.legend()
ax.set_ylim(0.0, 1.0)
fig.savefig('maxent_iedb_fraction_positive.png')
```


![png](notebook_files/likelihood-maxent_12_0.png)



```python

```
#### resample.ipynb

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import sys
sys.path.append('..')

from lib import *
```


```python
k = 9
ref = 'human'
name = 'Malaria'

likelihoods_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']

with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

df_ts = load_iedb_tcellepitopes(human_only=True)
df_bs = load_iedb_bcellepitopes(human_only=True)
```


```python
pathogenproteomes = load_proteomes(only_pathogens=True)
row = pathogenproteomes.loc[name]
iedbname = row['iedbname']
path = datadir + row['path']

likelihoods_pathogen = np.asarray(pd.read_csv('data/proteome-ref%s-k%i-%s.zip'%(ref, k, name))['likelihoods'])

df_t = df_ts[df_ts['Epitope', 'Parent Species'].str.contains(iedbname, na=False)]
epi_t = df_t['Epitope', 'Description'].unique()
likelihoods_t, weights_t = likelihoods_epitopes(epi_t, loglikelihood, k)
df_b = df_bs[df_bs['Epitope', 'Parent Species'].str.contains(iedbname, na=False)]
epi_b = df_b['Epitope', 'Description'].unique()
likelihoods_b, weights_b = likelihoods_epitopes(epi_b, loglikelihood, k)
len(epi_t), len(epi_b)
```




    (968, 984)




```python
fig, ax = plt.subplots()
ps = [likelihoods_human, likelihoods_pathogen]
labels = ['human', 'pathogen']
weights = [np.ones(len(likelihoods_human)), np.ones(len(likelihoods_pathogen))]
if len(likelihoods_t) > 100:
    ps.append(likelihoods_t)
    labels.append('T epitopes')
    weights.append(weights_t)
if len(likelihoods_b) > 100:
    ps.append(likelihoods_b)
    labels.append('B epitopes')
    weights.append(weights_b)
for i in range(20):
    values = np.random.choice(likelihoods_pathogen, size=len(epi_t))
    xmin=-14.1
    xmax=-8.9
    nbins=30
    bins = np.linspace(xmin, xmax, nbins)
    counts, bins = np.histogram(values, bins=bins)
    counts = counts/len(values)
    ax.plot(0.5*(bins[:-1]+bins[1:]), counts, '.8')
plot_histograms(ps, labels, weights=weights, xmin=-14.1, xmax=-8.9, ax=ax, nbins=30)
ax.set_xlim(-14, -9)
ax.set_ylabel('probability density')
ax.set_xlabel('$log_2$ likelihood')
plt.title(name)
fig.tight_layout()
```


![png](notebook_files/resample_3_0.png)



```python
means = sorted([np.mean(np.random.choice(likelihoods_pathogen, size=len(epi_t))) for i in range(100000)])
```


```python
mt, mb = np.mean(likelihoods_t), np.mean(likelihoods_b)
```


```python
np.searchsorted(means, mt), np.searchsorted(means, mb)
```




    (100000, 100000)




```python
means[-1]
```




    -11.508079545454548




```python

```
#### __init__.py

```python
from .main import *

```
#### resample.py

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'
name = 'Malaria'

likelihoods_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']

with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

pathogenproteomes = load_proteomes(only_pathogens=True)
row = pathogenproteomes.ix[name]

df_ts = load_iedb_tcellepitopes(human_only=True)
df_bs = load_iedb_bcellepitopes(human_only=True)

iedbname = row['iedbname']
path = datadir + row['path']

likelihoods_pathogen = pd.read_csv('data/proteome-ref%s-k%i-%s.zip'%(ref, k, name))['likelihoods']

df_t = df_ts[df_ts['Epitope', 'Parent Species'].str.contains(iedbname, na=False)]
epi_t = df_t['Epitope', 'Description'].unique()
likelihoods_t, weights_t = likelihoods_epitopes(epi_t, loglikelihood, k)
df_b = df_bs[df_bs['Epitope', 'Parent Species'].str.contains(iedbname, na=False)]
likelihoods_b, weights_b = likelihoods_epitopes(df_b['Epitope', 'Description'].unique(), loglikelihood, k)

print(epi_t)

if (len(likelihoods_t) > 100) or (len(likelihoods_b) > 100):
    fig, ax = plt.subplots()
    ps = [likelihoods_human, likelihoods_pathogen]
    labels = ['human', 'pathogen']
    weights = [np.ones(len(likelihoods_human)), np.ones(len(likelihoods_pathogen))]
    if len(likelihoods_t) > 100:
        ps.append(likelihoods_t)
        labels.append('T epitopes')
        weights.append(weights_t)
    if len(likelihoods_b) > 100:
        ps.append(likelihoods_b)
        labels.append('B epitopes')
        weights.append(weights_b)
    plot_histograms(ps, labels, weights=weights, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
    ax.set_xlim(-14, -9)
    ax.set_ylabel('probability density')
    ax.set_xlabel('$log_2$ likelihood')
    plt.title(name)
    fig.tight_layout()
    fig.savefig('plots/likelihoodprofile-%s-%s-k%i.png' % (name, likelihoodname, k), dpi=300)
    plt.close()

```
#### plot-cov.py

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'

likelihood_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']
likelihood_virus = pd.read_csv('data/proteome-ref%s-k%i-Viruses.zip'%(ref, k))['likelihoods']
likelihood_cov2 = pd.read_csv('data/proteome-ref%s-k%i-SARSCoV2.zip'%(ref, k))['likelihoods']
likelihood_flua = pd.read_csv('data/proteome-ref%s-k%i-InfluenzaA.zip'%(ref, k))['likelihoods']

fig, ax = plt.subplots(figsize=(3.4, 2.0))
ps = [likelihood_human, likelihood_virus, likelihood_cov2, likelihood_flua]
labels = ['human', 'viruses (averaged)', 'SARS-CoV-2', 'InfluenzaA']
plot_histograms(ps, labels, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
ax.set_xlim(-14, -9)
ax.set_ylim(0.0)
ax.set_ylabel('probability density')
ax.set_xlabel('$log_2$ likelihood')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-SARSCoV2-%s-k%i.png' % (ref, k), dpi=300)

```
#### plot-iedb-tcell-epitopes.py

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *
plt.style.use('../peptidome.mplstyle')


k = 9
ref = 'human'

likelihood_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']

with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'


df_ts = load_iedb_tcellepitopes(human_only=True)
mask = ~(df_ts['Assay', 'Qualitative Measure'] == 'Negative')
print(mask.shape, np.sum(mask))
likelihoods_t, weights_t = likelihoods_epitopes(df_ts[mask]['Epitope', 'Description'], loglikelihood, k)
likelihoods_t_neg, weights_t_neg = likelihoods_epitopes(df_ts[~mask]['Epitope', 'Description'], loglikelihood, k)
#df_bs = load_iedb_bcellepitopes(human_only=True)
#df_b = df_bs[~df_bs['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)]
#likelihoods_b, weights_b = likelihoods_epitopes(df_b['Epitope', 'Description'], loglikelihood, k)

print(len(likelihood_human), len(likelihoods_t))#, len(likelihoods_b))

fig, ax = plt.subplots()
ps = [likelihood_human, likelihoods_t]#, likelihoods_t_neg]#, likelihoods_b]
labels = ['Human proteins', 'IEDB+ T cell epitopes']#, 'IEDB negative']#, 'B epitopes']
weights = [np.ones(len(likelihood_human)), weights_t]#, weights_t_neg]#, weights_b]
plot_histograms(ps, labels, weights=weights, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
ax.set_xlim(-14, -9)
ax.set_ylabel('Frequency')
ax.set_xlabel('$log_{10}$ Likelihood under human proteome statistics')
ax.legend(title='Peptide')
#plt.title('all')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-all-%s-k%i.png' % (likelihoodname, k), dpi=300)
fig.savefig('main.png')
fig.savefig('../../paper/images/likelihoodprofile-iedb-tcell.pdf')

```
#### run-likelihoods.py

```python
import os.path
import json
import glob
import numpy as np
import pandas as pd

import sys
sys.path.append('..')

from lib import *
from lib.maxent import *

k = 9
ref = 'human'
#with open(datadir+ 'triplet-%s.json'%ref, 'r') as f:
#    tripletparams = json.load(f)
#loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
#likelihoodname = 'triplet'

params = np.load('../maxent/data/Human_reference_9.npz')
hi = params['hi']
Jij = params['Jij']
loglikelihood = lambda seq, k: -energy_potts(map_aatonumber(seq.upper()), hi, Jij) if isvalidaa(seq) else np.nan
likelihoodname = 'maxent'

def run(name, path, pathout, proteinname=True, sequence=False):
    print(name)
    likelihoods = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    if sequence:
        sequence = np.array([seq[i:i+k] for h, seq in fasta_iter(path) for i in range(len(seq)-k+1)])
    if proteinname:
        protein = np.array([h.split('|')[1] for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    else:
        protein = np.array([ind for ind, (h, seq) in enumerate(fasta_iter(path)) for i in range(len(seq)-k+1) ])
    if sequence:
        df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, protein=protein, sequence=sequence))
    else:
        df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, protein=protein))
    df.dropna(inplace=True)
    df.to_csv(pathout, compression='zip', index=False, float_format='%.4f')

# All viruses
path = datadir+'human-viruses-swissprot.fasta'
pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, 'Viruses')
if not os.path.exists(pathout):
    run('Viruses', path, pathout, proteinname=False)

# Cancer datasets
filenames = ['frameshifts.fasta.gz']
for filename in filenames:
    name = filename.split('.')[0]
    path = datadir+'cancer/' + filename
    pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, name)
    if not os.path.exists(pathout):
        run(name, path, pathout, proteinname=False)

# Ufo datasets
filenames = glob.glob(datadir + 'ufos/*.csv')
for filename in filenames:
    name = filename.split('/')[-1].split('.')[0]
    print(name)
    df_in = pd.read_csv(filename, sep='\t')
    sequences = np.array([seq[i:i+k] for seq in df_in['AA_seq'] for i in range(len(seq)-k+1)])
    likelihoods = np.array([loglikelihood(seq, k) for seq in sequences])
    df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, sequence=sequences))
    df.dropna(inplace=True)
    pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, name)
    df.to_csv(pathout, compression='zip', index=False, float_format='%.4f')

    # only middle part
    sequences = np.array([seq[i:i+k] for seq in df_in['AA_seq'] for i in range(10, min(len(seq)-k+1, 51))])
    likelihoods = np.array([loglikelihood(seq, k) for seq in sequences])
    df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, sequence=sequences))
    df.dropna(inplace=True)
    df.to_csv('data/proteome-ref%s-k%i-%s-middle.zip'%(ref, k, name), compression='zip', index=False, float_format='%.4f')



# SARS CoV 2 dataset
filenames = ['SARSCoV2.fasta']
for filename in filenames:
    name = filename.split('.')[0]
    path = datadir + filename
    pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, name)
    if not os.path.exists(pathout):
        run(name, path, pathout, proteinname=False)

# Proteomes
proteomes = load_proteomes()
for name, row in proteomes.iterrows():
    path = datadir + row['path']
    pathout = 'data/proteome-ref%s-%s-k%i-%s.zip'%(ref, likelihoodname, k, name)
    if not os.path.exists(pathout):
        run(name, path, pathout, proteinname=False)

```
#### plot-ufo.py

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'


likelihood_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']
likelihood_virus = pd.read_csv('data/proteome-ref%s-k%i-Viruses.zip'%(ref, k))['likelihoods']
likelihood_ufo = pd.read_csv('data/proteome-ref%s-k%i-ufo.zip'%(ref, k))['likelihoods']
likelihood_ufo_middle = pd.read_csv('data/proteome-ref%s-k%i-ufo-middle.zip'%(ref, k))['likelihoods']
likelihood_ext = pd.read_csv('data/proteome-ref%s-k%i-ext.zip'%(ref, k))['likelihoods']

#df_ext = pd.read_csv('data/proteome-ref%s-k%i-ext.zip'%(ref, k))
#likelihood_ext_noM = df_ext[~df_ext['sequence'].str.startswith('M')]['likelihoods']
#print(df_ext)

fig, ax = plt.subplots(figsize=(3.4, 2.0))
ps = [likelihood_human, likelihood_virus, likelihood_ufo, likelihood_ufo_middle, likelihood_ext]
labels = ['human', 'viruses', 'ufo', 'ufo [10:50]', 'ext']
if k == 9:
    xmin, xmax, nbins = -14.1, -8.9, 35
elif k == 5:
    xmin, xmax, nbins = -8.1, -4.1, 30
plot_histograms(ps, labels, xmin=xmin, xmax=xmax, ax=ax, nbins=nbins)
ax.set_xlim(xmin, xmax)
ax.set_ylim(0.0)
ax.set_ylabel('probability density')
ax.set_xlabel('$log_{10}$ likelihood')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-Ufo-%s-k%i.png' % (ref, k), dpi=300)

```
#### plot-likelihoodprofiles.py

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *
plt.style.use('../peptidome.mplstyle')

k = 9
ref = 'human'
likelihoodname = 'maxent'

likelihoods_human = pd.read_csv('data/proteome-ref%s-%s-k%i-Human.zip'%(ref, likelihoodname, k))['likelihoods']

with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

pathogenproteomes = load_proteomes(only_pathogens=True)

df_ts = load_iedb_tcellepitopes(human_only=True)
df_ts['length'] = [len(d) for d in df_ts['Epitope', 'Description']]
df_bs = load_iedb_bcellepitopes(human_only=True)
df_bs['length'] = [len(d) for d in df_bs['Epitope', 'Description']]


for name, row in pathogenproteomes.iterrows():
    iedbname = row['iedbname']
    path = datadir + row['path']
    print(name)

    likelihoods_pathogen = pd.read_csv('data/proteome-ref%s-k%i-%s.zip'%(ref, k, name))['likelihoods']

    df_t = df_ts[df_ts['Epitope', 'Parent Species'].str.contains(iedbname, na=False)]
    likelihoods_t, weights_t = likelihoods_epitopes(df_t['Epitope', 'Description'].unique(), loglikelihood, k)
    df_b = df_bs[df_bs['Epitope', 'Parent Species'].str.contains(iedbname, na=False)]
    likelihoods_b, weights_b = likelihoods_epitopes(df_b['Epitope', 'Description'].unique(), loglikelihood, k)


    epitope_proteins = [s.split('/')[-1] for s in df_t[df_t['Epitope', 'Organism Name'] == iedbname]['Epitope', 'Parent Protein IRI'].unique() if type(s) == type('')]
    epitope_proteins_aa = [s for h, s in fasta_iter(path, returnheader=True) if iscontained(h, epitope_proteins)]
    likelihoods_epi, weights_epi = likelihoods_epitopes(epitope_proteins_aa, loglikelihood, k)

    if (len(likelihoods_t) > 100) or (len(likelihoods_b) > 100):
        fig, ax = plt.subplots()
        ps = [likelihoods_human, likelihoods_pathogen]
        labels = ['Human proteins', name+' proteins']
        weights = [np.ones(len(likelihoods_human)), np.ones(len(likelihoods_pathogen))]
        if len(likelihoods_t) > 100:
            ps.append(likelihoods_t)
            labels.append('T cell epitopes')
            weights.append(weights_t)
        if len(likelihoods_b) > 100:
            ps.append(likelihoods_b)
            labels.append('B epitopes')
            weights.append(weights_b)
        if len(likelihoods_epi) > 1:
            ps.append(likelihoods_epi)
            labels.append('T cell antigens')
            weights.append(weights_epi)

        plot_histograms(ps, labels, weights=weights, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
        ax.set_xlim(-14, -9)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('$log_{10}$ Likelihood under human proteome statistics')
        plt.title(name)
        fig.tight_layout()
        fig.savefig('plots/likelihoodprofile-%s-%s-k%i.png' % (name, likelihoodname, k), dpi=300)
        plt.close()

```
#### tripletmodelparams.py

```python
# Calculate parameters for triplet model

import numpy as np
import pandas as pd

import json

from lib import *

name = 'malaria'
proteome = malaria

df = counter_to_df(count_kmers_proteome(proteome, 1), norm=True)
df = df.set_index('seq')
charlogp = np.log10(df['freq']).to_dict()

df1 = counter_to_df(count_kmers_proteome(proteome, 2), norm=False)
strcolumn_to_charcolumns(df1, 'seq')
count = df1.pivot(columns='aa1', index='aa2')['count']
count /= np.sum(count, axis=0)
count[count.isna()] = 1e-10
doubletlogp = np.log10(count).to_dict()

df2 = counter_to_df(count_kmers_proteome(proteome, 3), norm=False)
df2['aa12'] = [s[:2] for s in df2['seq']]
df2['aa3'] = [s[2] for s in df2['seq']]
count = df2.pivot(columns='aa12', index='aa3')['count']
count /= np.sum(count, axis=0)
count[count.isna()] = 1e-10
tripletlogp = np.log10(count).to_dict()

modelparams = dict(charlogp=charlogp, doubletlogp=doubletlogp, tripletlogp=tripletlogp)
with open('../data/triplet-%s.json'%name, 'w') as f:
    json.dump(modelparams, f, indent=4)

```
#### plot-chicken.py

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'

likelihood_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']
likelihood_flua = pd.read_csv('data/proteome-ref%s-k%i-InfluenzaA.zip'%(ref, k))['likelihoods']
likelihood_flub = pd.read_csv('data/proteome-ref%s-k%i-InfluenzaB.zip'%(ref, k))['likelihoods']
likelihood_chicken = pd.read_csv('data/proteome-ref%s-k%i-Chicken.zip'%(ref, k))['likelihoods']

fig, ax = plt.subplots(figsize=(3.4, 2.0))
ps = [likelihood_human, likelihood_flua, likelihood_flub, likelihood_chicken]
labels = ['human', 'Influenza A', 'Influenza B', 'Chicken']
plot_histograms(ps, labels, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
ax.set_xlim(-14, -9)
ax.set_ylim(0.0)
ax.set_ylabel('probability density')
ax.set_xlabel('$log_{10}$ likelihood')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-Chicken-%s-k%i.png' % (ref, k), dpi=300)

```
#### plot-cancer.py

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'

likelihoodname = 'maxent'

likelihood_human = pd.read_csv('data/proteome-ref%s-%s-k%i-Human.zip'%(ref, likelihoodname, k))['likelihoods']
likelihood_virus = pd.read_csv('data/proteome-ref%s-%s-k%i-Viruses.zip'%(ref, likelihoodname, k))['likelihoods']
likelihood_frameshifts = pd.read_csv('data/proteome-ref%s-%s-k%i-frameshifts.zip'%(ref, likelihoodname, k))['likelihoods']
likelihood_pb1ufo = pd.read_csv('data/proteome-ref%s-%s-k%i-pb1ufo.zip'%(ref, likelihoodname, k))['likelihoods']

fig, ax = plt.subplots(figsize=(3.4, 2.0))
ps = [likelihood_human, likelihood_virus, likelihood_frameshifts, likelihood_pb1ufo]
labels = ['human', 'viruses', 'frameshifts', 'pb1 ufo']
plot_histograms(ps, labels, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
ax.set_xlim(-14, -9)
ax.set_ylim(0.0)
ax.set_ylabel('probability density')
ax.set_xlabel('$log_10$ likelihood')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-Cancer-%s-k%i.png' % (ref, k), dpi=300)
fig.savefig('../../paper/images/cancer.pdf', dpi=300)

```
#### plot-viruses.py

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')

from lib import *
from lib.plotting import *
from lib.maxent import *

k = 9
ref = 'human'
likelihoodname = 'maxent'

likelihood_human = pd.read_csv('data/proteome-ref%s-%s-k%i-Human.zip'%(ref, likelihoodname, k))['likelihoods']
likelihood_virus = pd.read_csv('data/proteome-ref%s-%s-k%i-Viruses.zip'%(ref, likelihoodname, k))['likelihoods']

#with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
#    tripletparams = json.load(f)
#loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
#likelihoodname = 'triplet'

params = np.load('../maxent/data/Human_9.npz')
hi = params['hi']
Jij = params['Jij']
loglikelihood = lambda seq, k: -energy_potts(map_aatonumber(seq.upper()), hi, Jij) if isvalidaa(seq) else np.nan
likelihoodname = 'maxent'


df_ts = load_iedb_tcellepitopes(human_only=True)
mask = ~df_ts['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)
mask &= df_ts['Epitope', 'Parent Species'].str.contains('virus', case=False, na=False)
#mask &= ~df_ts['Epitope', 'Parent Species'].str.contains('Human immunodeficiency virus 1', case=False, na=False)
df_t = df_ts[mask]
likelihoods_t, weights_t = likelihoods_epitopes(df_t['Epitope', 'Description'].unique(), loglikelihood, k)
df_bs = load_iedb_bcellepitopes(human_only=True)
mask = ~df_bs['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)
mask &= df_bs['Epitope', 'Parent Species'].str.contains('virus', case=False, na=False)
#mask &= ~df_bs['Epitope', 'Parent Species'].str.contains('Human immunodeficiency virus 1', case=False, na=False)
df_b = df_bs[mask]
likelihoods_b, weights_b = likelihoods_epitopes(df_b['Epitope', 'Description'].unique(), loglikelihood, k)

print(len(likelihood_human), len(likelihood_virus), len(likelihoods_t), len(likelihoods_b))

xmin, xmax = likelihood_human.min(), likelihood_human.max()

fig, ax = plt.subplots(figsize=(3.4, 2.0))
ps = [likelihood_human, likelihood_virus, likelihoods_t, likelihoods_b]
labels = ['human', 'viruses', 'T cell epitopes', 'B cell epitopes']
weights = [np.ones(len(likelihood_human)), np.ones(len(likelihood_virus)), weights_t, weights_b]
plot_histograms(ps, labels, weights=weights, xmin=xmin, xmax=xmax, ax=ax, nbins=100, step=False)
ax.set_xlim(xmin, xmax)
#ax.set_yscale('log')
ax.set_ylim(0.0)
ax.set_ylabel('probability density')
ax.set_xlabel('loglikelihood')
fig.tight_layout()
plt.show()
#fig.savefig('plots/likelihoodprofile-Viruses-%s-k%i.png' % (likelihoodname, k), dpi=300)
#fig.savefig('../../paper/images/viruses.pdf', dpi=300)

```
