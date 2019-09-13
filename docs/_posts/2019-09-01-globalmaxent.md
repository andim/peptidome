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
%matplotlib inline

plt.style.use('../peptidome.mplstyle')

from lib import *
from lib.maxent import *

%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



```python
humanseqs = [s for s in fasta_iter(human, returnheader=False)]

```


```python
train, test = train_test_split(humanseqs, test_size=0.5)

```


```python
k = 3
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
pseudocount = 1e-6
aafreqs = np.mean(np.asarray(aacountss), axis=0)/4
prob_aa_ks = prob_aa(aacountss, k, pseudocount=pseudocount)
for i in range(len(aminoacids)):
    aa = map_numbertoaa([i])[0]
    prob_aa_k = prob_aa_ks[i]
    print(aa, aafreqs[i], prob_aa_k)
    if i < 4:
        l, = plt.plot(prob_aa_k, label=aa)
        x = np.arange(0, k+1, 1)
        n = k
        p = aafreqs[i]
        plt.plot(x, binom.pmf(x, n, p), '--', c=l.get_color())
plt.legend()
plt.yscale('log')
fks = np.asarray(prob_aa_ks)
```

    A 0.052869612410112614 [0.80854678 0.17271444 0.01745235 0.00128644]
    C 0.01805536711239755 [9.30250906e-01 6.73169212e-02 2.39197182e-03 4.02012121e-05]
    D 0.0364863642530867 [8.61883728e-01 1.30417741e-01 7.56787723e-03 1.30653928e-04]
    E 0.05207312599560802 [0.811788   0.16936769 0.01760813 0.00123619]
    F 0.027593103482932074 [8.94341178e-01 1.01070860e-01 4.46233399e-03 1.25628777e-04]
    G 0.05020502615591033 [8.14727712e-01 1.70438042e-01 1.41206740e-02 7.13571430e-04]
    H 0.020278996376866216 [9.21652873e-01 7.56486214e-02 2.62815392e-03 7.03521173e-05]
    I 0.03280041608249288 [8.75848622e-01 1.17216669e-01 6.81912975e-03 1.15578475e-04]
    K 0.042550465077713956 [8.42833381e-01 1.44548465e-01 1.22010663e-02 4.17087528e-04]
    L 0.07325162438002202 [0.7350841  0.23793587 0.02586948 0.00111056]
    M 0.015817918683008456 [9.38331348e-01 6.00907542e-02 1.55277163e-03 2.51257594e-05]
    N 0.027300388444163035 [8.95562289e-01 9.98045216e-02 4.50253519e-03 1.30653928e-04]
    P 0.04768742556495259 [0.82693883 0.15630229 0.01582923 0.00092965]
    Q 0.035217513655847514 [8.67200338e-01 1.25045855e-01 7.43722331e-03 3.16584510e-04]
    R 0.04283438610244272 [8.41355987e-01 1.46442947e-01 1.17086016e-02 4.92464791e-04]
    S 0.061713877959185724 [0.77887326 0.19708139 0.02236192 0.00168343]
    T 0.038485118015668424 [8.54863592e-01 1.36553450e-01 8.36185107e-03 2.21106644e-04]
    V 0.04376780787843155 [8.35833346e-01 1.53448007e-01 1.05327162e-02 1.85930588e-04]
    W 0.01021990060251559 [9.59748541e-01 3.96333650e-02 6.08043262e-04 1.00503068e-05]
    Y 0.020791561766642045 [9.19723215e-01 7.74325499e-02 2.79900905e-03 4.52263629e-05]



![png](notebook_files/globalmaxent_4_1.png)



```python
df4 = count(train, k)

kmers = [''.join(s) for s in itertools.product(aminoacids, repeat=k)]
df4_test = pd.DataFrame.from_dict(dict(seq=kmers, count=np.ones(len(kmers))))
df4_test.set_index('seq', inplace=True)
df4_count = counter_to_df(count_kmers_iterable(test, k), norm=False)
df4_count.set_index('seq', inplace=True)
df4_test = df4_test.add(df4_count, fill_value=0.0)
df4_test['freq'] = df4_test['count'] / np.sum(df4_test['count'])

m, jsd_test = calc_logfold(df4, df4_test)
jsd_flat = calc_jsd(df4_test['freq'], np.ones_like(df4_test['freq']), base=2)

tripletparams = calc_tripletmodelparams(train)
kmers = df4_test.index
df4_test['freq_ind'] = np.array([10**(loglikelihood_independent(s, **tripletparams)) for s in kmers])
df4_test['freq_mc'] = np.array([10**(loglikelihood_mc(s, **tripletparams)) for s in kmers])
df4_test['freq_tri'] = np.array([10**(loglikelihood_triplet(s, **tripletparams)) for s in kmers])
jsd_ind = calc_jsd(df4_test['freq'], df4_test['freq_ind'], base=2)
jsd_mc = calc_jsd(df4_test['freq'], df4_test['freq_mc'], base=2)
jsd_tri = calc_jsd(df4_test['freq'], df4_test['freq_tri'], base=2)
```

    0.010897964608850834 0.004330712857986406 0.0006560602716436357



```python
# evaluate empirical observables for fitting
df0 = count(train, 1)
df1 = count(train, 2, gap=0)
dfgap1 = count(train, 2, gap=1)
#dfgap2 = count(train, 2, gap=2)
```


```python
h, Jk = fit_ising(df0, [df1, dfgap1], nmcmc=1e5, niter=10, epsilon=0.2, N=k, output=True)
```

    [ 0.45583391 -0.65966922  0.06195008  0.46711346 -0.20365802  0.38743241
     -0.53224446 -0.03012386  0.24849105  0.80557161 -0.73964675 -0.21818239
      0.3494629   0.06753122  0.24014704  0.62725912  0.18191248  0.29630799
     -1.28896439 -0.51652418]
    iteration 0
    f1 1.7980113655344387e-05
    f2, gap 0 0.0028484694647664304
    f2, gap 1 0.002543765259393053
    iteration 1
    f1 1.909891196019059e-05
    f2, gap 0 0.0021012734690487316
    f2, gap 1 0.00204135343936187
    iteration 2
    f1 2.581531551959246e-05
    f2, gap 0 0.0016414890433930139
    f2, gap 1 0.0017588411425468627
    iteration 3
    f1 3.149932535818803e-05
    f2, gap 0 0.0013222696028610996
    f2, gap 1 0.001682078457705895
    iteration 4
    f1 2.37098776610066e-05
    f2, gap 0 0.0009425092680246468
    f2, gap 1 0.0014162282847352292
    iteration 5
    f1 3.3826509577924464e-05
    f2, gap 0 0.0008713812455070927
    f2, gap 1 0.0014788084147246757
    iteration 6
    f1 1.82518618672324e-05
    f2, gap 0 0.0008719767237254431
    f2, gap 1 0.0012051425125093342
    iteration 7
    f1 4.088294320124979e-05
    f2, gap 0 0.0007653191447326585
    f2, gap 1 0.001378997653632085
    iteration 8
    f1 4.596228509951644e-05
    f2, gap 0 0.0006390897658745285
    f2, gap 1 0.0015096784112672212
    iteration 9
    f1 4.6797464050848576e-05
    f2, gap 0 0.0008103418806007312
    f2, gap 1 0.0015002282774818215



```python
Z = np.exp(scipy.special.logsumexp([-clib.energy(np.array(s), h, Jk) for s in itertools.product(range(naminoacids), repeat=k)]))
df4_test['freq_maxent'] = np.exp([-clib.energy(map_aatonumber(s), h, Jk) for s in kmers])/Z
jsd_maxent = calc_jsd(df4_test['freq'], df4_test['freq_maxent'], base=2)
```


```python
hks = fit_global(fks, niter=3, nmcmc=2e5, epsilon=0.1)
```


```python
niter=10
nmcmc=1e6
epsilon=10.0
prng=None
output=False

N = len(fks[0])-1
if prng is None:
    prng = np.random
q = len(aminoacids)
aas_arr = np.array(list(aminoacids))
hks = np.zeros((q, N+1))
for i in range(niter):
    if output:
        print('iteration %g'%i)
    def jump(x):
        xp = x.copy()
        i = np.random.randint(0, len(x))
        xp[i] = (xp[i]+np.random.randint(0, q-1))%q
        return xp
#    def jump(x):
#        return prng.randint(q, size=N)
    def energy(x):
        return energy_global(aacounts_int(x), hks)
    x0 = jump(prng.randint(q, size=N))
    samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng, nburnin=1e3)
    aacountss = [aacounts_int(s) for s in samples]
    prob_aa_ks = prob_aa(aacountss, N, pseudocount=pseudocount)
    hks += np.log(fks/prob_aa_ks)*epsilon
    #hks += (fks-prob_aa_ks)*epsilon
    jsd = calc_jsd(fks, prob_aa_ks, base=2)
    print(jsd)

```

    [0.00077246 0.02835408 0.10252522 0.29745049]



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-202-8153a3419192> in <module>
         24         return energy_global(aacounts_int(x), hks)
         25     x0 = jump(prng.randint(q, size=N))
    ---> 26     samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng, nburnin=1e3)
         27     aacountss = [aacounts_int(s) for s in samples]
         28     prob_aa_ks = prob_aa(aacountss, N, pseudocount=pseudocount)


    ~/repos/peptidome/code/lib/main.py in mcmcsampler(x0, energy, jump, nsteps, nburnin, nsample, prng)
        376     for i in range(nsteps):
        377         xp = jump(x)
    --> 378         Exp = energy(xp)
        379         if prng.rand() < np.exp(-Exp+Ex):
        380             x = xp


    <ipython-input-202-8153a3419192> in energy(x)
         22 #        return prng.randint(q, size=N)
         23     def energy(x):
    ---> 24         return energy_global(aacounts_int(x), hks)
         25     x0 = jump(prng.randint(q, size=N))
         26     samples = mcmcsampler(x0, energy, jump, nmcmc, prng=prng, nburnin=1e3)


    KeyboardInterrupt: 



```python
plt.plot(fks.flatten(), prob_aa_ks.flatten(), 'o')
x = [1e-5, 1e0]
plt.plot(x, x, 'k')
plt.xscale('log')
plt.yscale('log')
```


```python
Z = np.exp(scipy.special.logsumexp([-energy_global(aacounts_int(np.array(s)), hks) for s in itertools.product(range(naminoacids), repeat=k)]))
df4_test['freq_maxentglobal'] = np.exp([-energy_global(aacounts(s), hks) for s in kmers])/Z
jsd_maxentglobal = calc_jsd(df4_test['freq'], df4_test['freq_maxentglobal'], base=2)
```


```python
print('4mer', 'test', jsd_test, 'maxent', jsd_maxent, 'maxentglobal', jsd_maxentglobal,
              'flat', jsd_flat, 'ind', jsd_ind, 'mc', jsd_mc, 'tri', jsd_tri)
```

    4mer test 0.0004548957507683046 maxent 0.0026720994413888133 maxentglobal 0.010713706421513562 flat 0.11583779521335114 ind 0.010897964608850834 mc 0.004330712857986406 tri 0.0006560602716436357



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
from scipy.interpolate import interpn
def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs, logcolor=True)   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        ax = plt.gca()
    if logcolor:
        
    data , x_e, y_e = np.histogram2d( x, y, bins = bins)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    return ax
```


      File "<ipython-input-140-dbbdf9711b33>", line 2
        def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs, logcolor=True)   :
                                                                                        ^
    SyntaxError: invalid syntax




```python
# Generate fake data
x = np.random.normal(size=100000)
y = x * 3 + np.random.normal(size=100000)

fig, ax = plt.subplots()
scatter(x, y, ax=ax, s=1)
plt.show()
```


```python
logscatter = False
#scatter = lambda x, y, ax: density_scatter(np.log10(x), np.log10(y), ax=ax, s=1, bins=100)
scatter = lambda x, y, ax: ax.scatter(x, y, s=1, alpha=1, edgecolor=None)
#scatter = lambda x, y, ax: ax.plot(x, y, 'o', ms=0.1, alpha=1)


fig, axes = plt.subplots(figsize=(5.8, 2.0), ncols=4, sharex=True, sharey=True)
ax = axes[0]
scatter(df4_test['freq_ind'], df4_test['freq'], ax)
ax.set_xlabel('independent prediction')
ax.set_ylabel('test set')
ax.set_title('JSD = %g'%round(jsd_ind, 4))
ax = axes[1]
scatter(df4_test['freq_maxentglobal'], df4_test['freq'], ax)
ax.set_xlabel('global maxent prediction')
#ax.set_ylabel('test set')
ax.set_title('JSD = %g'%round(jsd_maxentglobal, 4))
ax = axes[2]
scatter(df4_test['freq_maxent'], df4_test['freq'], ax)
ax.set_xlabel('maxent prediction')
#ax.set_ylabel('test set')
ax.set_title('JSD = %g'%round(jsd_maxent, 4))
ax = axes[3]
scatter(df4['freq'], df4_test['freq'], ax)
ax.set_xlabel('training set')
#ax.set_ylabel('test set')
ax.set_title('JSD = %g'%round(jsd_test, 4))
if logscatter:
    x = np.linspace(-8, -3)
else:
    x = np.logspace(-6, -2)
for ax in axes:
    ax.plot(x, x, 'k')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(x), max(x))
    #plt.legend()
    if not logscatter:
        ax.set_xscale('log')
        ax.set_yscale('log')
fig.tight_layout()
fig.savefig('main.png', dpi=600)
#fig.savefig('main.svg')
#fig.savefig('main.pdf')
```


![png](notebook_files/globalmaxent_17_0.png)



```python
df4_test['freq_maxentglobal'].sum()
```




    0.9999999999999996




```python
print(4*np.log2(20),
      scipy.stats.entropy(df4_test['freq_ind'], base=2),
      scipy.stats.entropy(df4_test['freq_maxentglobal'], base=2),
      scipy.stats.entropy(df4_test['freq_maxent'], base=2),
      scipy.stats.entropy(df4_test['freq'], base=2),
     )
```

    17.28771237954945 12.531732292067517 12.584080695323632 12.497181813942442 12.483591187682844



```python
df4_test['freq_maxentglobal'].sort_values(ascending=False)
```




    seq
    SSS    0.001448
    EEE    0.001164
    AAA    0.001085
    LLL    0.001044
    PPP    0.000907
    GGG    0.000786
    LSS    0.000751
    SSL    0.000751
    SLS    0.000751
    LLS    0.000688
    LSL    0.000688
    SLL    0.000688
    ALL    0.000587
    LLA    0.000587
    LAL    0.000587
    ALA    0.000584
    AAL    0.000584
    LAA    0.000584
    GLL    0.000579
    LGL    0.000579
    LLG    0.000579
    LEL    0.000569
    LLE    0.000569
    ELL    0.000569
    LEE    0.000567
    ELE    0.000567
    EEL    0.000567
    RRR    0.000560
    PLL    0.000534
    LLP    0.000534
             ...   
    WHM    0.000009
    HWM    0.000009
    MWH    0.000009
    WMM    0.000008
    MWM    0.000008
    MMW    0.000008
    NWW    0.000008
    WWN    0.000008
    WNW    0.000008
    WFW    0.000008
    FWW    0.000008
    WWF    0.000008
    MCW    0.000008
    WMC    0.000008
    CMW    0.000008
    CWM    0.000008
    MWC    0.000008
    WCM    0.000008
    YWW    0.000006
    WWY    0.000006
    WYW    0.000006
    HWW    0.000006
    WHW    0.000006
    WWH    0.000006
    WWC    0.000005
    CWW    0.000005
    WCW    0.000005
    WMW    0.000005
    MWW    0.000005
    WWM    0.000005
    Name: freq_maxentglobal, Length: 8000, dtype: float64




```python
df4['freq'].sort_values(ascending=False)
```




    seq
    SSS    0.001469
    LLL    0.001425
    EEE    0.001338
    PPP    0.001204
    AAA    0.001154
    SSL    0.000905
    SLL    0.000898
    LLS    0.000879
    ALL    0.000856
    LLA    0.000845
    LSS    0.000826
    LEE    0.000823
    LSL    0.000804
    SLS    0.000790
    EEL    0.000787
    LLE    0.000778
    GGG    0.000763
    LLG    0.000753
    LAL    0.000741
    ELL    0.000733
    SSP    0.000721
    LGL    0.000700
    AAL    0.000699
    LAA    0.000697
    PGP    0.000682
    LPP    0.000672
    VLL    0.000671
    SPS    0.000669
    GSS    0.000667
    PSS    0.000666
             ...   
    HWM    0.000007
    HMW    0.000007
    YMW    0.000006
    WYC    0.000006
    WMC    0.000006
    MWH    0.000006
    WYW    0.000006
    WWY    0.000006
    MYW    0.000006
    HWW    0.000006
    WWW    0.000006
    YWW    0.000006
    NWW    0.000006
    WCW    0.000005
    MCW    0.000005
    MMW    0.000005
    WWH    0.000005
    WFM    0.000005
    CMW    0.000005
    WHM    0.000005
    WMY    0.000004
    MWC    0.000004
    WWC    0.000004
    MWM    0.000004
    WWM    0.000004
    WMW    0.000004
    WHW    0.000004
    CWW    0.000003
    MWW    0.000003
    WCM    0.000003
    Name: freq, Length: 8000, dtype: float64




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
