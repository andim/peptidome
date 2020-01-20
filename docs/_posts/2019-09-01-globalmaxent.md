---
layout: post
title: Inference of global maxent models
---

Infering and benchmarking of Maxent models with global couplings.

{% include post-image-gallery.html filter="globalmaxent/" %}

### Code 
#### globalmaxent.ipynb



```python
import sys
sys.path.append('..')
import itertools, copy
import re
import json
import numpy as np
import scipy.misc
from scipy.stats import binom
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

plt.style.use('../peptidome.mplstyle')

from lib import *
from lib.maxent import *

%load_ext autoreload
%autoreload 2
```


```python
humanseqs = [s for s in fasta_iter(human, returnheader=False)]

```


```python
train, test = train_test_split(humanseqs, test_size=0.5)

```


```python
k = 3

```


```python
kmers = [''.join(s) for s in itertools.product(aminoacids, repeat=k)]
df = pd.DataFrame.from_dict(dict(seq=kmers, freq=np.zeros(len(kmers))))
df.set_index('seq', inplace=True)
df['freq'] = df.add(count(train, k), fill_value=0.0)
df['freq_test'] = np.zeros(len(kmers))
df['freq_test'] = df['freq_test'].add(count(test, k)['freq'], fill_value=0.0)
jsd_test = calc_jsd(df['freq_test'], df['freq'])
jsd_flat = calc_jsd(df['freq_test'], np.ones_like(df['freq']), base=2)


#df_test.set_index('seq', inplace=True)
#df_count = counter_to_df(count_kmers_iterable(test, k), norm=False)
#df_count.set_index('seq', inplace=True)
#df_test = df_test.add(df_count, fill_value=0.0)
#df_test['freq'] = df_test['count'] / np.sum(df_test['count'])


tripletparams = calc_tripletmodelparams(train)
#kmers = df.index
df['freq_ind'] = np.array([10**(loglikelihood_independent(s, **tripletparams)) for s in kmers])
df['freq_mc'] = np.array([10**(loglikelihood_mc(s, **tripletparams)) for s in kmers])
df['freq_tri'] = np.array([10**(loglikelihood_triplet(s, **tripletparams)) for s in kmers])
jsd_ind = calc_jsd(df['freq_test'], df['freq_ind'], base=2)
jsd_mc = calc_jsd(df['freq_test'], df['freq_mc'], base=2)
jsd_tri = calc_jsd(df['freq_test'], df['freq_tri'], base=2)
```


```python
aacountss = []
#for seq in np.random.choice(humanseqs, 1000, replace=False):
for seq in train:
    seq = seq.strip('X')
    if not isvalidaa(seq):
        seq = re.sub('X|U', '', seq)
        if not isvalidaa(seq):
            print(seq)
    seq = map_aatonumber(seq)
    for i in range(len(seq)-k):
        #if isvalidaa(seq[i:i+k]):
        #print(seq[i:i+k], aacounts(seq[i:i+k]))
        aacountss.append(aacounts_int(seq[i:i+k]))
```


```python
pseudocount = 1e-2
aafreqs = np.mean(np.asarray(aacountss), axis=0)/k
prob_aa_ks = prob_aa(aacountss, k, pseudocount=pseudocount)
for i in range(len(aminoacids)):
    aa = map_numbertoaa([i])[0]
    prob_aa_k = prob_aa_ks[i]
    print(aa, aafreqs[i], prob_aa_k)
    if i >16:
        l, = plt.plot(prob_aa_k, label=aa)
        x = np.arange(0, k+1, 1)
        n = k
        p = aafreqs[i]
        plt.plot(x, binom.pmf(x, n, p), '--', c=l.get_color())
plt.legend()
plt.yscale('log')
fks = np.asarray(prob_aa_ks)
```

    A 0.07018196319094387 [0.80916578 0.17231273 0.0173313  0.00119019]
    C 0.0230518411129529 [9.33056429e-01 6.47747237e-02 2.12573121e-03 4.31157313e-05]
    D 0.04751265640647439 [8.65330503e-01 1.27051639e-01 7.36723431e-03 2.50623661e-04]
    E 0.07104289911013449 [0.80831139 0.17162396 0.01868921 0.00137544]
    F 0.03642254924194759 [8.95360441e-01 1.00090155e-01 4.47071103e-03 7.86935267e-05]
    G 0.0651804495280772 [8.19793028e-01 1.65643908e-01 1.37917429e-02 7.71321347e-04]
    H 0.026185023932931942 [9.24073474e-01 7.33913854e-02 2.44172513e-03 9.34153731e-05]
    I 0.043559198005213164 [8.75960026e-01 1.17522398e-01 6.39752031e-03 1.20054905e-04]
    K 0.05764134520755409 [8.40720483e-01 1.46186534e-01 1.25414375e-02 5.51545212e-04]
    L 0.09976095693871588 [0.73159094 0.23897938 0.02798553 0.00144414]
    M 0.02024943247866004 [9.40738078e-01 5.77927128e-02 1.45203148e-03 1.71772400e-05]
    N 0.03617338783220694 [8.95955799e-01 9.96479737e-02 4.31648216e-03 7.97450872e-05]
    P 0.06315935031790425 [0.82867275 0.15436119 0.0157813  0.00118476]
    Q 0.04760332428640815 [8.66062564e-01 1.25559124e-01 7.88407627e-03 4.94235167e-04]
    R 0.05632414888740814 [8.43751431e-01 1.44027681e-01 1.17178904e-02 5.02998171e-04]
    S 0.08343197554677932 [0.77624353 0.19872057 0.02353235 0.00150356]
    T 0.05390725400291096 [8.48410194e-01 1.41818527e-01 9.41059153e-03 3.60686989e-04]
    V 0.0597070188973594 [8.32553539e-01 1.56062440e-01 1.10934388e-02 2.90582959e-04]
    W 0.012168423999460666 [9.64093587e-01 3.53129792e-02 5.87999307e-04 5.43481496e-06]
    Y 0.026736801075956666 [9.22503319e-01 7.48229097e-02 2.63381017e-03 3.99610499e-05]



![png](notebook_files/globalmaxent_6_1.png)



```python
for i, aa in enumerate(aminoacids):
    dist = scipy.stats.rv_discrete(values=(np.arange(k+1), prob_aa_ks[i]))
    print(aa, dist.var()/(dist.mean()*(1-dist.mean()/k)))
```

    A 1.0625771941793762
    C 1.0195649023022253
    D 1.0198392910670318
    E 1.077522672501814
    F 1.0138094683680006
    G 1.036764862445943
    H 1.0173861616040063
    I 1.0170495603845016
    K 1.0518972712591002
    L 1.0182700744915758
    M 1.0091885943746781
    N 1.0120498490975285
    P 1.0830176561123857
    Q 1.0377695511885143
    R 1.0465291072434337
    S 1.0624230391457272
    T 1.0231977957502412
    V 1.0150855582618494
    W 1.0088789308278159
    Y 1.0156053703525478



```python
# evaluate empirical observables for fitting
df0 = count(train, 1)
df1 = count(train, 2, gap=0)
dfgap1 = count(train, 2, gap=1)
if k == 4:
    dfgap2 = count(train, 2, gap=2)
```


```python
args = [df1, dfgap1, dfgap2] if k == 4 else [df1, dfgap1]
h, Jk = fit_ising(df0, args, nmcmc=5e5, niter=10, epsilon=0.2, N=k, output=True)
```

    [ 0.45372895 -0.65964851  0.06305901  0.46478129 -0.20277892  0.37814671
     -0.53260975 -0.0245608   0.25711263  0.80430225 -0.73222319 -0.20974447
      0.34726576  0.0645158   0.23295415  0.62676144  0.18877261  0.29108636
     -1.29886505 -0.51205627]
    iteration 0
    f1 6.4331098127172665e-06
    f2, gap 0 0.002384177887086923
    f2, gap 1 0.001963444279899073
    iteration 1
    f1 5.311625918338817e-06
    f2, gap 0 0.0017391698645377244
    f2, gap 1 0.0013212536337933634
    iteration 2
    f1 1.1357785609720052e-05
    f2, gap 0 0.0011039446554275954
    f2, gap 1 0.0009683778291910818
    iteration 3
    f1 1.2602404368973092e-05
    f2, gap 0 0.0007024574157963314
    f2, gap 1 0.0007356535850437341
    iteration 4
    f1 8.006796881827585e-06
    f2, gap 0 0.0005425770284775988
    f2, gap 1 0.0005897652759192061
    iteration 5



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-9-09a14c965d89> in <module>
          1 args = [df1, dfgap1, dfgap2] if k == 4 else [df1, dfgap1]
    ----> 2 h, Jk = fit_ising(df0, args, nmcmc=5e5, niter=10, epsilon=0.2, N=k, output=True)
    

    ~/repos/peptidome/code/lib/maxent.py in fit_ising(f1, f2s, niter, nmcmc, epsilon, Jk, prng, output, N)
         84             return clib.energy(x, h, Jk)
         85         x0 = jump(None)
    ---> 86         samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng)
         87         samples = [''.join(aas_arr[s]) for s in samples]
         88         f1_model = count(samples, 1)


    ~/repos/peptidome/code/lib/main.py in mcmcsampler(x0, energy, jump, nsteps, nburnin, nsample, prng)
        392     for i in range(nsteps):
        393         xp = jump(x)
    --> 394         Exp = energy(xp)
        395         if prng.rand() < np.exp(-Exp+Ex):
        396             x = xp


    ~/repos/peptidome/code/lib/maxent.py in energy(x)
         81         def jump(x):
         82             return prng.randint(q, size=N)
    ---> 83         def energy(x):
         84             return clib.energy(x, h, Jk)
         85         x0 = jump(None)


    KeyboardInterrupt: 



```python
Z = np.exp(scipy.special.logsumexp([-clib.energy(np.array(s), h, Jk) for s in itertools.product(range(naminoacids), repeat=k)]))
df['freq_maxent'] = np.exp([-clib.energy(map_aatonumber(s), h, Jk) for s in kmers])/Z
jsd_maxent = calc_jsd(df_test['freq'], df['freq_maxent'], base=2)
```


```python
hks = fit_global(fks, niter=10, nmcmc=1e6, epsilon=0.1, output=True)
```


```python
pseudocount = 1e-2
niter=10
nmcmc=1e6
epsilon=0.5
prng=None
output=False

N = len(fks[0])-1
if prng is None:
    prng = np.random
q = len(aminoacids)
aas_arr = np.array(list(aminoacids))
f1 = np.sum(np.arange(fks.shape[1])*fks, axis=1)/(fks.shape[1]-1)
h = np.array(np.log(f1))
h -= np.mean(h)
hks = h.reshape(20, 1)*np.arange(fks.shape[1])
#hks = np.zeros((q, N+1))
for i in range(niter):
    if output:
        print('iteration %g'%i)
#    def jump(x):
#        xp = x.copy()
#        i = np.random.randint(0, len(x))
#        xp[i] = (xp[i]+np.random.randint(0, q-1))%q
#        return xp
    def jump(x):
        return prng.randint(q, size=N)
    def energy(x):
        return energy_global(aacounts_int(x), hks)
    x0 = jump(prng.randint(q, size=N))
    samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng, nburnin=1e3)
    aacountss = [aacounts_int(s) for s in samples]
    prob_aa_ks = prob_aa(aacountss, N, pseudocount=pseudocount)
    #Z = np.exp(scipy.special.logsumexp([-energy_global(aacounts_int(np.array(s)), hks) for s in itertools.product(range(naminoacids), repeat=k)]))
    #probs = np.exp([-energy_global(aacounts(s), hks) for s in kmers])/Z
    if i == 0:
        prob_aa_ks0 = prob_aa_ks
    hks += np.log(fks/prob_aa_ks)*epsilon
    #hks += (fks-prob_aa_ks)*epsilon
    jsd = calc_jsd(fks, prob_aa_ks, base=2)
    print(jsd)
```


```python
plt.plot(fks.flatten(), prob_aa_ks.flatten(), 'o')
x = [1e-6, 1e0]
plt.plot(x, x, 'k')
plt.xscale('log')
plt.yscale('log')
```


```python
plt.plot(fks.flatten(), prob_aa_ks0.flatten()/fks.flatten(), 'o')
plt.plot(fks.flatten(), prob_aa_ks.flatten()/fks.flatten(), 'o')
#x = [1e-6, 1e0]
#plt.plot(x, x, 'k')
plt.axhline(1.0, c='k')
#plt.ylim(1e-1)
plt.xscale('log')
plt.yscale('log')
```


```python
Z = np.exp(scipy.special.logsumexp(
           [-energy_global(aacounts_int(np.array(s)), hks) for s in itertools.product(range(naminoacids), repeat=k)]
           ))
df['freq_maxentglobal'] = np.exp([-energy_global(aacounts(s), hks) for s in kmers])/Z
jsd_maxentglobal = calc_jsd(df['freq_test'], df['freq_maxentglobal'], base=2)
```


```python
entropies = {}
Smax = np.log2(20)*k
for column in df.filter(regex='freq'):
    f3 = np.array(df[column])
    entropy = scipy.stats.entropy(f3, base=2)
    print(column, Smax-entropy)
    entropies[column] = entropy
```


```python
print('test', jsd_test, 'maxent', jsd_maxent, 'maxentglobal', jsd_maxentglobal,
              'flat', jsd_flat, 'ind', jsd_ind, 'mc', jsd_mc, 'tri', jsd_tri)
```


```python
from scipy.stats import gaussian_kde
def scatterplot(x, y, ax=None):
    if ax is None:
        ax = plt.gca()
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    ax.scatter(x, y, c=z, s=1, edgecolor='')
```


```python
scatter = lambda x, y, ax: plotting.density_scatter(x, y, ax=ax, s=1, bins=100,
                                           trans=lambda x: np.log(x+1e-8),
                                           norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=50 if k ==3 else 400),
                                           cmap='viridis')
#scatter = lambda x, y, ax: ax.scatter(x, y, s=1, alpha=1, edgecolor=None)

fig, axes = plt.subplots(figsize=(7.2, 2.0), ncols=5, sharex=True, sharey=True)
axes[0].set_ylabel('test set')

for ax, column, xlabel in [(axes[0], 'freq_ind','independent prediction'),
                           (axes[1], 'freq_maxentglobal', 'global maxent prediction'),
                           (axes[2], 'freq_mc', 'mc'),
                           (axes[3], 'freq_maxent', 'maxent prediction'),
                           (axes[4], 'freq', 'training set')
                            ]:
    scatter(df[column], df['freq_test'], ax)
    ax.set_xlabel(xlabel)
    jsd = calc_jsd(df['freq_test'], df[column], base=2)
    entropy = entropies[column]
    ax.set_title('JSD = {:.4f}\nH = {:.2f}'.format(jsd, entropy))
    
if k == 3:
    x = np.logspace(-5.7, -2.7)
elif k == 4:
    x = np.logspace(-7.7, -2.9)
for ax in axes:
    ax.plot(x, x, 'k', lw=0.8)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(x), max(x))
    ax.set_xscale('log')
    ax.set_yscale('log')
fig.tight_layout()
fig.savefig('main.png' if k == 3 else 'comparison_k4.png', dpi=600)
```


```python
2**entropies['freq'], 2**entropies['freq_maxent'], 2**entropies['freq_ind'], 20**k
```


```python
bins = np.linspace(-6, -2)
kwargs = dict(bins=bins, histtype='step')
plt.hist(np.log10(df['freq_ind']), **kwargs)
plt.hist(np.log10(df['freq_maxent']),**kwargs)
plt.hist(np.log10(df['freq']),**kwargs)
plt.yscale('log')
```


```python

```
#### benchmark-aacounting.ipynb



```python
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

%load_ext Cython

%load_ext autoreload
%autoreload 2
```


```python
def aacounts(seq):
    counter = np.zeros(len(aminoacids), dtype=int)
    for c in seq:
        counter[c] += 1
    return counter
```


```python
seq.dtype
```




    dtype('int64')




```python
seq = map_aatonumber('AWCYAE')
```


```python
%timeit aacounts(seq)
```

    4.14 µs ± 61.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)



```cython
%%cython -a
import cython
import numpy as np
cimport numpy as np

cdef int naminoacids = 20
@cython.boundscheck(False)
@cython.wraparound(False)
def aacounts_cython(np.ndarray[np.int64_t, ndim=1] seq):
    cdef np.ndarray[np.int64_t, cast=True] counter = np.zeros(naminoacids, dtype=np.int64)
    cdef Py_ssize_t i
    for i in range(seq.shape[0]):
        counter[seq[i]] += 1
    return counter
```




<!DOCTYPE html>
<!-- Generated by Cython 0.29.6 -->
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>Cython: _cython_magic_d8cff81e055fca030227c03a77197d1e.pyx</title>
    <style type="text/css">
    
body.cython { font-family: courier; font-size: 12; }

.cython.tag  {  }
.cython.line { margin: 0em }
.cython.code { font-size: 9; color: #444444; display: none; margin: 0px 0px 0px 8px; border-left: 8px none; }

.cython.line .run { background-color: #B0FFB0; }
.cython.line .mis { background-color: #FFB0B0; }
.cython.code.run  { border-left: 8px solid #B0FFB0; }
.cython.code.mis  { border-left: 8px solid #FFB0B0; }

.cython.code .py_c_api  { color: red; }
.cython.code .py_macro_api  { color: #FF7000; }
.cython.code .pyx_c_api  { color: #FF3000; }
.cython.code .pyx_macro_api  { color: #FF7000; }
.cython.code .refnanny  { color: #FFA000; }
.cython.code .trace  { color: #FFA000; }
.cython.code .error_goto  { color: #FFA000; }

.cython.code .coerce  { color: #008000; border: 1px dotted #008000 }
.cython.code .py_attr { color: #FF0000; font-weight: bold; }
.cython.code .c_attr  { color: #0000FF; }
.cython.code .py_call { color: #FF0000; font-weight: bold; }
.cython.code .c_call  { color: #0000FF; }

.cython.score-0 {background-color: #FFFFff;}
.cython.score-1 {background-color: #FFFFe7;}
.cython.score-2 {background-color: #FFFFd4;}
.cython.score-3 {background-color: #FFFFc4;}
.cython.score-4 {background-color: #FFFFb6;}
.cython.score-5 {background-color: #FFFFaa;}
.cython.score-6 {background-color: #FFFF9f;}
.cython.score-7 {background-color: #FFFF96;}
.cython.score-8 {background-color: #FFFF8d;}
.cython.score-9 {background-color: #FFFF86;}
.cython.score-10 {background-color: #FFFF7f;}
.cython.score-11 {background-color: #FFFF79;}
.cython.score-12 {background-color: #FFFF73;}
.cython.score-13 {background-color: #FFFF6e;}
.cython.score-14 {background-color: #FFFF6a;}
.cython.score-15 {background-color: #FFFF66;}
.cython.score-16 {background-color: #FFFF62;}
.cython.score-17 {background-color: #FFFF5e;}
.cython.score-18 {background-color: #FFFF5b;}
.cython.score-19 {background-color: #FFFF57;}
.cython.score-20 {background-color: #FFFF55;}
.cython.score-21 {background-color: #FFFF52;}
.cython.score-22 {background-color: #FFFF4f;}
.cython.score-23 {background-color: #FFFF4d;}
.cython.score-24 {background-color: #FFFF4b;}
.cython.score-25 {background-color: #FFFF48;}
.cython.score-26 {background-color: #FFFF46;}
.cython.score-27 {background-color: #FFFF44;}
.cython.score-28 {background-color: #FFFF43;}
.cython.score-29 {background-color: #FFFF41;}
.cython.score-30 {background-color: #FFFF3f;}
.cython.score-31 {background-color: #FFFF3e;}
.cython.score-32 {background-color: #FFFF3c;}
.cython.score-33 {background-color: #FFFF3b;}
.cython.score-34 {background-color: #FFFF39;}
.cython.score-35 {background-color: #FFFF38;}
.cython.score-36 {background-color: #FFFF37;}
.cython.score-37 {background-color: #FFFF36;}
.cython.score-38 {background-color: #FFFF35;}
.cython.score-39 {background-color: #FFFF34;}
.cython.score-40 {background-color: #FFFF33;}
.cython.score-41 {background-color: #FFFF32;}
.cython.score-42 {background-color: #FFFF31;}
.cython.score-43 {background-color: #FFFF30;}
.cython.score-44 {background-color: #FFFF2f;}
.cython.score-45 {background-color: #FFFF2e;}
.cython.score-46 {background-color: #FFFF2d;}
.cython.score-47 {background-color: #FFFF2c;}
.cython.score-48 {background-color: #FFFF2b;}
.cython.score-49 {background-color: #FFFF2b;}
.cython.score-50 {background-color: #FFFF2a;}
.cython.score-51 {background-color: #FFFF29;}
.cython.score-52 {background-color: #FFFF29;}
.cython.score-53 {background-color: #FFFF28;}
.cython.score-54 {background-color: #FFFF27;}
.cython.score-55 {background-color: #FFFF27;}
.cython.score-56 {background-color: #FFFF26;}
.cython.score-57 {background-color: #FFFF26;}
.cython.score-58 {background-color: #FFFF25;}
.cython.score-59 {background-color: #FFFF24;}
.cython.score-60 {background-color: #FFFF24;}
.cython.score-61 {background-color: #FFFF23;}
.cython.score-62 {background-color: #FFFF23;}
.cython.score-63 {background-color: #FFFF22;}
.cython.score-64 {background-color: #FFFF22;}
.cython.score-65 {background-color: #FFFF22;}
.cython.score-66 {background-color: #FFFF21;}
.cython.score-67 {background-color: #FFFF21;}
.cython.score-68 {background-color: #FFFF20;}
.cython.score-69 {background-color: #FFFF20;}
.cython.score-70 {background-color: #FFFF1f;}
.cython.score-71 {background-color: #FFFF1f;}
.cython.score-72 {background-color: #FFFF1f;}
.cython.score-73 {background-color: #FFFF1e;}
.cython.score-74 {background-color: #FFFF1e;}
.cython.score-75 {background-color: #FFFF1e;}
.cython.score-76 {background-color: #FFFF1d;}
.cython.score-77 {background-color: #FFFF1d;}
.cython.score-78 {background-color: #FFFF1c;}
.cython.score-79 {background-color: #FFFF1c;}
.cython.score-80 {background-color: #FFFF1c;}
.cython.score-81 {background-color: #FFFF1c;}
.cython.score-82 {background-color: #FFFF1b;}
.cython.score-83 {background-color: #FFFF1b;}
.cython.score-84 {background-color: #FFFF1b;}
.cython.score-85 {background-color: #FFFF1a;}
.cython.score-86 {background-color: #FFFF1a;}
.cython.score-87 {background-color: #FFFF1a;}
.cython.score-88 {background-color: #FFFF1a;}
.cython.score-89 {background-color: #FFFF19;}
.cython.score-90 {background-color: #FFFF19;}
.cython.score-91 {background-color: #FFFF19;}
.cython.score-92 {background-color: #FFFF19;}
.cython.score-93 {background-color: #FFFF18;}
.cython.score-94 {background-color: #FFFF18;}
.cython.score-95 {background-color: #FFFF18;}
.cython.score-96 {background-color: #FFFF18;}
.cython.score-97 {background-color: #FFFF17;}
.cython.score-98 {background-color: #FFFF17;}
.cython.score-99 {background-color: #FFFF17;}
.cython.score-100 {background-color: #FFFF17;}
.cython.score-101 {background-color: #FFFF16;}
.cython.score-102 {background-color: #FFFF16;}
.cython.score-103 {background-color: #FFFF16;}
.cython.score-104 {background-color: #FFFF16;}
.cython.score-105 {background-color: #FFFF16;}
.cython.score-106 {background-color: #FFFF15;}
.cython.score-107 {background-color: #FFFF15;}
.cython.score-108 {background-color: #FFFF15;}
.cython.score-109 {background-color: #FFFF15;}
.cython.score-110 {background-color: #FFFF15;}
.cython.score-111 {background-color: #FFFF15;}
.cython.score-112 {background-color: #FFFF14;}
.cython.score-113 {background-color: #FFFF14;}
.cython.score-114 {background-color: #FFFF14;}
.cython.score-115 {background-color: #FFFF14;}
.cython.score-116 {background-color: #FFFF14;}
.cython.score-117 {background-color: #FFFF14;}
.cython.score-118 {background-color: #FFFF13;}
.cython.score-119 {background-color: #FFFF13;}
.cython.score-120 {background-color: #FFFF13;}
.cython.score-121 {background-color: #FFFF13;}
.cython.score-122 {background-color: #FFFF13;}
.cython.score-123 {background-color: #FFFF13;}
.cython.score-124 {background-color: #FFFF13;}
.cython.score-125 {background-color: #FFFF12;}
.cython.score-126 {background-color: #FFFF12;}
.cython.score-127 {background-color: #FFFF12;}
.cython.score-128 {background-color: #FFFF12;}
.cython.score-129 {background-color: #FFFF12;}
.cython.score-130 {background-color: #FFFF12;}
.cython.score-131 {background-color: #FFFF12;}
.cython.score-132 {background-color: #FFFF11;}
.cython.score-133 {background-color: #FFFF11;}
.cython.score-134 {background-color: #FFFF11;}
.cython.score-135 {background-color: #FFFF11;}
.cython.score-136 {background-color: #FFFF11;}
.cython.score-137 {background-color: #FFFF11;}
.cython.score-138 {background-color: #FFFF11;}
.cython.score-139 {background-color: #FFFF11;}
.cython.score-140 {background-color: #FFFF11;}
.cython.score-141 {background-color: #FFFF10;}
.cython.score-142 {background-color: #FFFF10;}
.cython.score-143 {background-color: #FFFF10;}
.cython.score-144 {background-color: #FFFF10;}
.cython.score-145 {background-color: #FFFF10;}
.cython.score-146 {background-color: #FFFF10;}
.cython.score-147 {background-color: #FFFF10;}
.cython.score-148 {background-color: #FFFF10;}
.cython.score-149 {background-color: #FFFF10;}
.cython.score-150 {background-color: #FFFF0f;}
.cython.score-151 {background-color: #FFFF0f;}
.cython.score-152 {background-color: #FFFF0f;}
.cython.score-153 {background-color: #FFFF0f;}
.cython.score-154 {background-color: #FFFF0f;}
.cython.score-155 {background-color: #FFFF0f;}
.cython.score-156 {background-color: #FFFF0f;}
.cython.score-157 {background-color: #FFFF0f;}
.cython.score-158 {background-color: #FFFF0f;}
.cython.score-159 {background-color: #FFFF0f;}
.cython.score-160 {background-color: #FFFF0f;}
.cython.score-161 {background-color: #FFFF0e;}
.cython.score-162 {background-color: #FFFF0e;}
.cython.score-163 {background-color: #FFFF0e;}
.cython.score-164 {background-color: #FFFF0e;}
.cython.score-165 {background-color: #FFFF0e;}
.cython.score-166 {background-color: #FFFF0e;}
.cython.score-167 {background-color: #FFFF0e;}
.cython.score-168 {background-color: #FFFF0e;}
.cython.score-169 {background-color: #FFFF0e;}
.cython.score-170 {background-color: #FFFF0e;}
.cython.score-171 {background-color: #FFFF0e;}
.cython.score-172 {background-color: #FFFF0e;}
.cython.score-173 {background-color: #FFFF0d;}
.cython.score-174 {background-color: #FFFF0d;}
.cython.score-175 {background-color: #FFFF0d;}
.cython.score-176 {background-color: #FFFF0d;}
.cython.score-177 {background-color: #FFFF0d;}
.cython.score-178 {background-color: #FFFF0d;}
.cython.score-179 {background-color: #FFFF0d;}
.cython.score-180 {background-color: #FFFF0d;}
.cython.score-181 {background-color: #FFFF0d;}
.cython.score-182 {background-color: #FFFF0d;}
.cython.score-183 {background-color: #FFFF0d;}
.cython.score-184 {background-color: #FFFF0d;}
.cython.score-185 {background-color: #FFFF0d;}
.cython.score-186 {background-color: #FFFF0d;}
.cython.score-187 {background-color: #FFFF0c;}
.cython.score-188 {background-color: #FFFF0c;}
.cython.score-189 {background-color: #FFFF0c;}
.cython.score-190 {background-color: #FFFF0c;}
.cython.score-191 {background-color: #FFFF0c;}
.cython.score-192 {background-color: #FFFF0c;}
.cython.score-193 {background-color: #FFFF0c;}
.cython.score-194 {background-color: #FFFF0c;}
.cython.score-195 {background-color: #FFFF0c;}
.cython.score-196 {background-color: #FFFF0c;}
.cython.score-197 {background-color: #FFFF0c;}
.cython.score-198 {background-color: #FFFF0c;}
.cython.score-199 {background-color: #FFFF0c;}
.cython.score-200 {background-color: #FFFF0c;}
.cython.score-201 {background-color: #FFFF0c;}
.cython.score-202 {background-color: #FFFF0c;}
.cython.score-203 {background-color: #FFFF0b;}
.cython.score-204 {background-color: #FFFF0b;}
.cython.score-205 {background-color: #FFFF0b;}
.cython.score-206 {background-color: #FFFF0b;}
.cython.score-207 {background-color: #FFFF0b;}
.cython.score-208 {background-color: #FFFF0b;}
.cython.score-209 {background-color: #FFFF0b;}
.cython.score-210 {background-color: #FFFF0b;}
.cython.score-211 {background-color: #FFFF0b;}
.cython.score-212 {background-color: #FFFF0b;}
.cython.score-213 {background-color: #FFFF0b;}
.cython.score-214 {background-color: #FFFF0b;}
.cython.score-215 {background-color: #FFFF0b;}
.cython.score-216 {background-color: #FFFF0b;}
.cython.score-217 {background-color: #FFFF0b;}
.cython.score-218 {background-color: #FFFF0b;}
.cython.score-219 {background-color: #FFFF0b;}
.cython.score-220 {background-color: #FFFF0b;}
.cython.score-221 {background-color: #FFFF0b;}
.cython.score-222 {background-color: #FFFF0a;}
.cython.score-223 {background-color: #FFFF0a;}
.cython.score-224 {background-color: #FFFF0a;}
.cython.score-225 {background-color: #FFFF0a;}
.cython.score-226 {background-color: #FFFF0a;}
.cython.score-227 {background-color: #FFFF0a;}
.cython.score-228 {background-color: #FFFF0a;}
.cython.score-229 {background-color: #FFFF0a;}
.cython.score-230 {background-color: #FFFF0a;}
.cython.score-231 {background-color: #FFFF0a;}
.cython.score-232 {background-color: #FFFF0a;}
.cython.score-233 {background-color: #FFFF0a;}
.cython.score-234 {background-color: #FFFF0a;}
.cython.score-235 {background-color: #FFFF0a;}
.cython.score-236 {background-color: #FFFF0a;}
.cython.score-237 {background-color: #FFFF0a;}
.cython.score-238 {background-color: #FFFF0a;}
.cython.score-239 {background-color: #FFFF0a;}
.cython.score-240 {background-color: #FFFF0a;}
.cython.score-241 {background-color: #FFFF0a;}
.cython.score-242 {background-color: #FFFF0a;}
.cython.score-243 {background-color: #FFFF0a;}
.cython.score-244 {background-color: #FFFF0a;}
.cython.score-245 {background-color: #FFFF0a;}
.cython.score-246 {background-color: #FFFF09;}
.cython.score-247 {background-color: #FFFF09;}
.cython.score-248 {background-color: #FFFF09;}
.cython.score-249 {background-color: #FFFF09;}
.cython.score-250 {background-color: #FFFF09;}
.cython.score-251 {background-color: #FFFF09;}
.cython.score-252 {background-color: #FFFF09;}
.cython.score-253 {background-color: #FFFF09;}
.cython.score-254 {background-color: #FFFF09;}
.cython .hll { background-color: #ffffcc }
.cython  { background: #f8f8f8; }
.cython .c { color: #408080; font-style: italic } /* Comment */
.cython .err { border: 1px solid #FF0000 } /* Error */
.cython .k { color: #008000; font-weight: bold } /* Keyword */
.cython .o { color: #666666 } /* Operator */
.cython .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.cython .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.cython .cp { color: #BC7A00 } /* Comment.Preproc */
.cython .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.cython .c1 { color: #408080; font-style: italic } /* Comment.Single */
.cython .cs { color: #408080; font-style: italic } /* Comment.Special */
.cython .gd { color: #A00000 } /* Generic.Deleted */
.cython .ge { font-style: italic } /* Generic.Emph */
.cython .gr { color: #FF0000 } /* Generic.Error */
.cython .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.cython .gi { color: #00A000 } /* Generic.Inserted */
.cython .go { color: #888888 } /* Generic.Output */
.cython .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.cython .gs { font-weight: bold } /* Generic.Strong */
.cython .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.cython .gt { color: #0044DD } /* Generic.Traceback */
.cython .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.cython .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.cython .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.cython .kp { color: #008000 } /* Keyword.Pseudo */
.cython .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.cython .kt { color: #B00040 } /* Keyword.Type */
.cython .m { color: #666666 } /* Literal.Number */
.cython .s { color: #BA2121 } /* Literal.String */
.cython .na { color: #7D9029 } /* Name.Attribute */
.cython .nb { color: #008000 } /* Name.Builtin */
.cython .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.cython .no { color: #880000 } /* Name.Constant */
.cython .nd { color: #AA22FF } /* Name.Decorator */
.cython .ni { color: #999999; font-weight: bold } /* Name.Entity */
.cython .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.cython .nf { color: #0000FF } /* Name.Function */
.cython .nl { color: #A0A000 } /* Name.Label */
.cython .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.cython .nt { color: #008000; font-weight: bold } /* Name.Tag */
.cython .nv { color: #19177C } /* Name.Variable */
.cython .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.cython .w { color: #bbbbbb } /* Text.Whitespace */
.cython .mb { color: #666666 } /* Literal.Number.Bin */
.cython .mf { color: #666666 } /* Literal.Number.Float */
.cython .mh { color: #666666 } /* Literal.Number.Hex */
.cython .mi { color: #666666 } /* Literal.Number.Integer */
.cython .mo { color: #666666 } /* Literal.Number.Oct */
.cython .sa { color: #BA2121 } /* Literal.String.Affix */
.cython .sb { color: #BA2121 } /* Literal.String.Backtick */
.cython .sc { color: #BA2121 } /* Literal.String.Char */
.cython .dl { color: #BA2121 } /* Literal.String.Delimiter */
.cython .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.cython .s2 { color: #BA2121 } /* Literal.String.Double */
.cython .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.cython .sh { color: #BA2121 } /* Literal.String.Heredoc */
.cython .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.cython .sx { color: #008000 } /* Literal.String.Other */
.cython .sr { color: #BB6688 } /* Literal.String.Regex */
.cython .s1 { color: #BA2121 } /* Literal.String.Single */
.cython .ss { color: #19177C } /* Literal.String.Symbol */
.cython .bp { color: #008000 } /* Name.Builtin.Pseudo */
.cython .fm { color: #0000FF } /* Name.Function.Magic */
.cython .vc { color: #19177C } /* Name.Variable.Class */
.cython .vg { color: #19177C } /* Name.Variable.Global */
.cython .vi { color: #19177C } /* Name.Variable.Instance */
.cython .vm { color: #19177C } /* Name.Variable.Magic */
.cython .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
</head>
<body class="cython">
<p><span style="border-bottom: solid 1px grey;">Generated by Cython 0.29.6</span></p>
<p>
    <span style="background-color: #FFFF00">Yellow lines</span> hint at Python interaction.<br />
    Click on a line that starts with a "<code>+</code>" to see the C code that Cython generated for it.
</p>
<div class="cython"><pre class="cython line score-8" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">01</span>: <span class="k">import</span> <span class="nn">cython</span></pre>
<pre class='cython code score-8 '>  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(0);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 1, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_test, __pyx_t_1) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 1, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
</pre><pre class="cython line score-8" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">02</span>: <span class="k">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></pre>
<pre class='cython code score-8 '>  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_Import</span>(__pyx_n_s_numpy, 0, 0);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 2, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_np, __pyx_t_1) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 2, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">03</span>: <span class="k">cimport</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">04</span>: </pre>
<pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">05</span>: <span class="k">cdef</span> <span class="kt">int</span> <span class="nf">naminoacids</span> <span class="o">=</span> <span class="mf">20</span></pre>
<pre class='cython code score-0 '>  __pyx_v_46_cython_magic_d8cff81e055fca030227c03a77197d1e_naminoacids = 20;
</pre><pre class="cython line score-0">&#xA0;<span class="">06</span>: <span class="nd">@cython</span><span class="o">.</span><span class="n">boundscheck</span><span class="p">(</span><span class="bp">False</span><span class="p">)</span></pre>
<pre class="cython line score-0">&#xA0;<span class="">07</span>: <span class="nd">@cython</span><span class="o">.</span><span class="n">wraparound</span><span class="p">(</span><span class="bp">False</span><span class="p">)</span></pre>
<pre class="cython line score-35" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">08</span>: <span class="k">def</span> <span class="nf">aacounts_cython</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">int64_t</span><span class="p">,</span> <span class="n">ndim</span><span class="o">=</span><span class="mf">1</span><span class="p">]</span> <span class="n">seq</span><span class="p">):</span></pre>
<pre class='cython code score-35 '>/* Python wrapper */
static PyObject *__pyx_pw_46_cython_magic_d8cff81e055fca030227c03a77197d1e_1aacounts_cython(PyObject *__pyx_self, PyObject *__pyx_v_seq); /*proto*/
static PyMethodDef __pyx_mdef_46_cython_magic_d8cff81e055fca030227c03a77197d1e_1aacounts_cython = {"aacounts_cython", (PyCFunction)__pyx_pw_46_cython_magic_d8cff81e055fca030227c03a77197d1e_1aacounts_cython, METH_O, 0};
static PyObject *__pyx_pw_46_cython_magic_d8cff81e055fca030227c03a77197d1e_1aacounts_cython(PyObject *__pyx_self, PyObject *__pyx_v_seq) {
  PyObject *__pyx_r = 0;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("aacounts_cython (wrapper)", 0);
  if (unlikely(!<span class='pyx_c_api'>__Pyx_ArgTypeTest</span>(((PyObject *)__pyx_v_seq), __pyx_ptype_5numpy_ndarray, 1, "seq", 0))) <span class='error_goto'>__PYX_ERR(0, 8, __pyx_L1_error)</span>
  __pyx_r = __pyx_pf_46_cython_magic_d8cff81e055fca030227c03a77197d1e_aacounts_cython(__pyx_self, ((PyArrayObject *)__pyx_v_seq));

  /* function exit code */
  goto __pyx_L0;
  __pyx_L1_error:;
  __pyx_r = NULL;
  __pyx_L0:;
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}

static PyObject *__pyx_pf_46_cython_magic_d8cff81e055fca030227c03a77197d1e_aacounts_cython(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_seq) {
  PyArrayObject *__pyx_v_counter = 0;
  Py_ssize_t __pyx_v_i;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_counter;
  __Pyx_Buffer __pyx_pybuffer_counter;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_seq;
  __Pyx_Buffer __pyx_pybuffer_seq;
  PyObject *__pyx_r = NULL;
  <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
  <span class='refnanny'>__Pyx_RefNannySetupContext</span>("aacounts_cython", 0);
  __pyx_pybuffer_counter.pybuffer.buf = NULL;
  __pyx_pybuffer_counter.refcount = 0;
  __pyx_pybuffernd_counter.data = NULL;
  __pyx_pybuffernd_counter.rcbuffer = &amp;__pyx_pybuffer_counter;
  __pyx_pybuffer_seq.pybuffer.buf = NULL;
  __pyx_pybuffer_seq.refcount = 0;
  __pyx_pybuffernd_seq.data = NULL;
  __pyx_pybuffernd_seq.rcbuffer = &amp;__pyx_pybuffer_seq;
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(<span class='pyx_c_api'>__Pyx_GetBufferAndValidate</span>(&amp;__pyx_pybuffernd_seq.rcbuffer-&gt;pybuffer, (PyObject*)__pyx_v_seq, &amp;__Pyx_TypeInfo_nn___pyx_t_5numpy_int64_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) <span class='error_goto'>__PYX_ERR(0, 8, __pyx_L1_error)</span>
  }
  __pyx_pybuffernd_seq.diminfo[0].strides = __pyx_pybuffernd_seq.rcbuffer-&gt;pybuffer.strides[0]; __pyx_pybuffernd_seq.diminfo[0].shape = __pyx_pybuffernd_seq.rcbuffer-&gt;pybuffer.shape[0];
/* … */
  /* function exit code */
  __pyx_L1_error:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_1);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_4);
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_5);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    <span class='pyx_c_api'>__Pyx_ErrFetch</span>(&amp;__pyx_type, &amp;__pyx_value, &amp;__pyx_tb);
    <span class='pyx_c_api'>__Pyx_SafeReleaseBuffer</span>(&amp;__pyx_pybuffernd_counter.rcbuffer-&gt;pybuffer);
    <span class='pyx_c_api'>__Pyx_SafeReleaseBuffer</span>(&amp;__pyx_pybuffernd_seq.rcbuffer-&gt;pybuffer);
  <span class='pyx_c_api'>__Pyx_ErrRestore</span>(__pyx_type, __pyx_value, __pyx_tb);}
  <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_d8cff81e055fca030227c03a77197d1e.aacounts_cython", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  <span class='pyx_c_api'>__Pyx_SafeReleaseBuffer</span>(&amp;__pyx_pybuffernd_counter.rcbuffer-&gt;pybuffer);
  <span class='pyx_c_api'>__Pyx_SafeReleaseBuffer</span>(&amp;__pyx_pybuffernd_seq.rcbuffer-&gt;pybuffer);
  __pyx_L2:;
  <span class='pyx_macro_api'>__Pyx_XDECREF</span>((PyObject *)__pyx_v_counter);
  <span class='refnanny'>__Pyx_XGIVEREF</span>(__pyx_r);
  <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
  return __pyx_r;
}
/* … */
  __pyx_tuple__8 = <span class='py_c_api'>PyTuple_Pack</span>(3, __pyx_n_s_seq, __pyx_n_s_counter, __pyx_n_s_i);<span class='error_goto'> if (unlikely(!__pyx_tuple__8)) __PYX_ERR(0, 8, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_tuple__8);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_tuple__8);
/* … */
  __pyx_t_1 = PyCFunction_NewEx(&amp;__pyx_mdef_46_cython_magic_d8cff81e055fca030227c03a77197d1e_1aacounts_cython, NULL, __pyx_n_s_cython_magic_d8cff81e055fca0302);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 8, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_aacounts_cython, __pyx_t_1) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 8, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
</pre><pre class="cython line score-36" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">09</span>:     <span class="k">cdef</span> <span class="kt">np</span>.<span class="kt">ndarray</span>[<span class="kt">np</span>.<span class="nf">int64_t</span><span class="p">,</span> <span class="nf">cast</span><span class="o">=</span><span class="bp">True</span><span class="p">]</span> <span class="n">counter</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">naminoacids</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">int64</span><span class="p">)</span></pre>
<pre class='cython code score-36 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_1, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 9, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_1, __pyx_n_s_zeros);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 9, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyInt_From_int</span>(__pyx_v_46_cython_magic_d8cff81e055fca030227c03a77197d1e_naminoacids);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 9, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  __pyx_t_3 = <span class='py_c_api'>PyTuple_New</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 9, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_1);
  <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_3, 0, __pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyDict_NewPresized</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 9, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_4, __pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 9, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_4);
  __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_4, __pyx_n_s_int64);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 9, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_1, __pyx_n_s_dtype, __pyx_t_5) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 9, __pyx_L1_error)</span>
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_t_2, __pyx_t_3, __pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 9, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_5) == Py_None) || likely(<span class='pyx_c_api'>__Pyx_TypeTest</span>(__pyx_t_5, __pyx_ptype_5numpy_ndarray))))) <span class='error_goto'>__PYX_ERR(0, 9, __pyx_L1_error)</span>
  __pyx_t_6 = ((PyArrayObject *)__pyx_t_5);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(<span class='pyx_c_api'>__Pyx_GetBufferAndValidate</span>(&amp;__pyx_pybuffernd_counter.rcbuffer-&gt;pybuffer, (PyObject*)__pyx_t_6, &amp;__Pyx_TypeInfo_nn___pyx_t_5numpy_int64_t, PyBUF_FORMAT| PyBUF_STRIDES| PyBUF_WRITABLE, 1, 1, __pyx_stack) == -1)) {
      __pyx_v_counter = ((PyArrayObject *)Py_None); <span class='pyx_macro_api'>__Pyx_INCREF</span>(Py_None); __pyx_pybuffernd_counter.rcbuffer-&gt;pybuffer.buf = NULL;
      <span class='error_goto'>__PYX_ERR(0, 9, __pyx_L1_error)</span>
    } else {__pyx_pybuffernd_counter.diminfo[0].strides = __pyx_pybuffernd_counter.rcbuffer-&gt;pybuffer.strides[0]; __pyx_pybuffernd_counter.diminfo[0].shape = __pyx_pybuffernd_counter.rcbuffer-&gt;pybuffer.shape[0];
    }
  }
  __pyx_t_6 = 0;
  __pyx_v_counter = ((PyArrayObject *)__pyx_t_5);
  __pyx_t_5 = 0;
</pre><pre class="cython line score-0">&#xA0;<span class="">10</span>:     <span class="k">cdef</span> <span class="kt">Py_ssize_t</span> <span class="nf">i</span></pre>
<pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">11</span>:     <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">seq</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mf">0</span><span class="p">]):</span></pre>
<pre class='cython code score-0 '>  __pyx_t_7 = (__pyx_v_seq-&gt;dimensions[0]);
  __pyx_t_8 = __pyx_t_7;
  for (__pyx_t_9 = 0; __pyx_t_9 &lt; __pyx_t_8; __pyx_t_9+=1) {
    __pyx_v_i = __pyx_t_9;
</pre><pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">12</span>:         <span class="n">counter</span><span class="p">[</span><span class="n">seq</span><span class="p">[</span><span class="n">i</span><span class="p">]]</span> <span class="o">+=</span> <span class="mf">1</span></pre>
<pre class='cython code score-0 '>    __pyx_t_10 = __pyx_v_i;
    __pyx_t_11 = (*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_int64_t *, __pyx_pybuffernd_seq.rcbuffer-&gt;pybuffer.buf, __pyx_t_10, __pyx_pybuffernd_seq.diminfo[0].strides));
    *__Pyx_BufPtrStrided1d(__pyx_t_5numpy_int64_t *, __pyx_pybuffernd_counter.rcbuffer-&gt;pybuffer.buf, __pyx_t_11, __pyx_pybuffernd_counter.diminfo[0].strides) += 1;
  }
</pre><pre class="cython line score-2" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">13</span>:     <span class="k">return</span> <span class="n">counter</span></pre>
<pre class='cython code score-2 '>  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_r);
  <span class='pyx_macro_api'>__Pyx_INCREF</span>(((PyObject *)__pyx_v_counter));
  __pyx_r = ((PyObject *)__pyx_v_counter);
  goto __pyx_L0;
</pre></div></body></html>




```python
%timeit aacounts_cython(seq)
```

    1.28 µs ± 31.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)



```python
aacounts_cython(seq)
```




    array([2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])




```python
aacounts(seq)
```




    array([2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])




```python

```
