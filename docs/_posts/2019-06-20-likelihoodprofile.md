---
layout: post
title: Distribution of likelihoods
---

Likelihoods of peptides under different models.


{% include image.html
   url="/code/likelihoodprofile/main.png"
   description="Distribution of likelihoods under a triplet model learned on the human proteomes for random peptides (9-mers) from the human proteome and IEDB T cell epitopes."
%}

{% include image-gallery.html filter="likelihoodprofile/" %}

### Code 
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
#### run-likelihoods.py

```python
import os.path
import json
import numpy as np
import pandas as pd

import sys
sys.path.append('..')

from lib import *

k = 9
ref = 'human'
with open(datadir+ 'triplet-%s.json'%ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

def run(name, path, proteinname=True):
    print(name)
    likelihoods = np.array([loglikelihood(seq[i:i+k], k) for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    if proteinname:
        protein = np.array([h.split('|')[1] for h, seq in fasta_iter(path) for i in range(len(seq)-k+1) ])
    else:
        protein = np.array([ind for ind, (h, seq) in enumerate(fasta_iter(path)) for i in range(len(seq)-k+1) ])
    df = pd.DataFrame.from_dict(dict(likelihoods=likelihoods, protein=protein))
    df.to_csv('data/proteome-ref%s-k%i-%s.zip'%(ref, k, name), compression='zip', index=False, float_format='%.4f')

path = datadir+'human-viruses-uniref90_nohiv.fasta'
pathout = 'data/proteome-ref%s-k%i-%s.zip'%(ref, k, 'Viruses')
if not os.path.exists(pathout):
    run('Viruses', pathin, proteinname=False)

proteomes = load_proteomes()
for name, row in proteomes.iterrows():
    path = datadir + row['path']
    pathout = 'data/proteome-ref%s-k%i-%s.zip'%(ref, k, name)
    if not os.path.exists(pathout):
        run(name, path)

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

likelihoods_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']

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
#### plot-all.py

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

k = 9
ref = 'human'

likelihood_human = pd.read_csv('data/proteome-ref%s-k%i-Human.zip'%(ref, k))['likelihoods']
likelihood_virus = pd.read_csv('data/proteome-ref%s-k%i-Viruses.zip'%(ref, k))['likelihoods']

with open(datadir+ 'triplet-%s.json' % ref, 'r') as f:
    tripletparams = json.load(f)
loglikelihood = lambda seq, k: loglikelihood_triplet(seq, **tripletparams, k=k)
likelihoodname = 'triplet'

df_ts = load_iedb_tcellepitopes(human_only=True)
mask = ~df_ts['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)
mask &= df_ts['Epitope', 'Parent Species'].str.contains('virus', case=False, na=False)
mask &= ~df_ts['Epitope', 'Parent Species'].str.contains('Human immunodeficiency virus 1', case=False, na=False)
df_t = df_ts[mask]
likelihoods_t, weights_t = likelihoods_epitopes(df_t['Epitope', 'Description'].unique(), loglikelihood, k)
df_bs = load_iedb_bcellepitopes(human_only=True)
mask = ~df_bs['Epitope', 'Parent Species'].str.contains('Homo sapiens', na=False)
mask &= df_bs['Epitope', 'Parent Species'].str.contains('virus', case=False, na=False)
mask &= ~df_bs['Epitope', 'Parent Species'].str.contains('Human immunodeficiency virus 1', case=False, na=False)
df_b = df_bs[mask]
likelihoods_b, weights_b = likelihoods_epitopes(df_b['Epitope', 'Description'].unique(), loglikelihood, k)

print(len(likelihood_human), len(likelihood_virus), len(likelihoods_t), len(likelihoods_b))

fig, ax = plt.subplots(figsize=(3.4, 2.0))
ps = [likelihood_human, likelihood_virus, likelihoods_t, likelihoods_b]
labels = ['human', 'viruses', 'T cell epitopes', 'B cell epitopes']
weights = [np.ones(len(likelihood_human)), np.ones(len(likelihood_virus)), weights_t, weights_b]
plot_histograms(ps, labels, weights=weights, xmin=-14.1, xmax=-8.9, ax=ax, nbins=35)
ax.set_xlim(-14, -9)
ax.set_ylabel('probability density')
ax.set_xlabel('$log_2$ likelihood')
fig.tight_layout()
plt.show()
fig.savefig('plots/likelihoodprofile-Viruses-%s-k%i.png' % (likelihoodname, k), dpi=300)
fig.savefig('../../paper/images/viruses.pdf', dpi=300)

```
