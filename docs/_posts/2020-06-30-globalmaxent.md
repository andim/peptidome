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
k = 4

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

    A 0.06972035254397053 [7.57393048e-01 2.09882702e-01 2.95552322e-02 2.78781255e-03
     3.81205276e-04]
    C 0.022843102684972524 [9.13829972e-01 8.11273932e-02 4.89188283e-03 1.41740141e-04
     9.01199830e-06]
    D 0.0476272678362107 [8.24808202e-01 1.60717668e-01 1.36800107e-02 7.45081385e-04
     4.90383703e-05]
    E 0.07110607075311093 [7.55377174e-01 2.08989992e-01 3.19126296e-02 3.27176778e-03
     4.48435719e-04]
    F 0.03690882044359964 [8.61708011e-01 1.29246197e-01 8.75226180e-03 2.89543151e-04
     3.98704251e-06]
    G 0.06556454079267812 [7.68533029e-01 2.03353898e-01 2.56842835e-02 2.17944635e-03
     2.49343505e-04]
    H 0.02624239867133237 [9.00113055e-01 9.50624620e-02 4.60459398e-03 1.81593239e-04
     3.82953614e-05]
    I 0.043771004098804495 [8.38082575e-01 1.49228540e-01 1.22182684e-02 4.63510586e-04
     7.10598059e-06]
    K 0.05726877189212979 [7.96646790e-01 1.79512735e-01 2.20628499e-02 1.67383183e-03
     1.03793061e-04]
    L 0.10041100673257444 [6.59162613e-01 2.85213027e-01 5.07565604e-02 4.55330478e-03
     3.14494656e-04]
    M 0.020150809321119803 [9.22319721e-01 7.48115436e-02 2.81536317e-03 5.25038571e-05
     8.68104432e-07]
    N 0.036126140253448374 [8.64420101e-01 1.26948579e-01 8.34454729e-03 2.80186337e-04
     6.58615758e-06]
    P 0.0628775756147557 [7.83292537e-01 1.85878141e-01 2.72882841e-02 3.10854335e-03
     4.32494480e-04]
    Q 0.04776489098011764 [8.25871240e-01 1.58591245e-01 1.43646176e-02 9.52490767e-04
     2.20406690e-04]
    R 0.05618619713941393 [7.99590201e-01 1.77776180e-01 2.10083023e-02 1.54924759e-03
     7.60691670e-05]
    S 0.08284982572933336 [7.17985785e-01 2.37636919e-01 3.97279953e-02 4.29079416e-03
     3.58506337e-04]
    T 0.053328470100213216 [8.05389173e-01 1.76923497e-01 1.67178564e-02 9.23207404e-04
     4.62659809e-05]
    V 0.0601274086216113 [7.82562705e-01 1.95496947e-01 2.08352012e-02 1.07828794e-03
     2.68592551e-05]
    W 0.012258119852127679 [9.52174716e-01 4.66341906e-02 1.17497502e-03 1.61162461e-05
     1.73274338e-09]
    Y 0.02686722593847548 [8.97973810e-01 9.67359455e-02 5.14139788e-03 1.45205628e-04
     3.64049383e-06]



![png](notebook_files/globalmaxent_6_1.png)



```python
for i, aa in enumerate(aminoacids):
    dist = scipy.stats.rv_discrete(values=(np.arange(k+1), prob_aa_ks[i]))
    print(aa, dist.var()/(dist.mean()*(1-dist.mean()/k)))
```

    A 1.085109700967974
    C 1.0501836842781322
    D 1.028653055257551
    E 1.1066018482339912
    F 1.0206946248295172
    G 1.0646887991946026
    H 1.0244027624024779
    I 1.0257558671704634
    K 1.0743560570745991
    L 1.032156007699662
    M 1.0137188731566193
    N 1.0200171104068345
    P 1.1314179608014783
    Q 1.0533779126031229
    R 1.0676147274766012
    S 1.0892723513943574
    T 1.0267558865012723
    V 1.0224676297413469
    W 1.0132874203383413
    Y 1.0242446147525897



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
h, Jk = fit_potts(df0, args, nmcmc=5e5, niter=10, epsilon=0.2, N=k, output=True)
```

    [ 0.44746516 -0.66927156  0.06488325  0.46528148 -0.19050289  0.38355623
     -0.53138133 -0.02079074  0.25017281  0.80998859 -0.73103507 -0.21180513
      0.34233483  0.06704333  0.22998163  0.6195353   0.17771732  0.29710163
     -1.29180768 -0.50846716]
    iteration 0
    f1 4.476127273362751e-06
    f2, gap 0 0.0024146260305523814
    f2, gap 1 0.0018117193322767565
    f2, gap 2 0.002019313740316058
    iteration 1
    f1 1.2584504711115676e-05
    f2, gap 0 0.0015747695930211163
    f2, gap 1 0.0011877504201717845
    f2, gap 2 0.0013958006676574818
    iteration 2
    f1 1.120406752720984e-05
    f2, gap 0 0.0010625495536692028
    f2, gap 1 0.000772223791703597
    f2, gap 2 0.0010109149612457348
    iteration 3
    f1 9.980313949829674e-06
    f2, gap 0 0.0006979550782313768
    f2, gap 1 0.0005821444345439923
    f2, gap 2 0.0006670138354170465
    iteration 4
    f1 1.2166275274410422e-05
    f2, gap 0 0.0005021279830033098
    f2, gap 1 0.0004514366511976827
    f2, gap 2 0.0005356060343491994
    iteration 5
    f1 8.28324569029826e-06
    f2, gap 0 0.0003860064772410925
    f2, gap 1 0.0003012936115492517
    f2, gap 2 0.000530944555331742
    iteration 6
    f1 9.35558974193211e-06
    f2, gap 0 0.00027289098970431907
    f2, gap 1 0.00026414300637994626
    f2, gap 2 0.0004218665234370844
    iteration 7
    f1 9.45077689208547e-06
    f2, gap 0 0.00020424714213987264
    f2, gap 1 0.00023570088865206007
    f2, gap 2 0.0003957130594563968
    iteration 8
    f1 1.652228835580059e-05
    f2, gap 0 0.0001939415970841791
    f2, gap 1 0.00021354522056234773
    f2, gap 2 0.00041326997061243555
    iteration 9
    f1 1.2673191241158162e-05
    f2, gap 0 0.0001688426219162046
    f2, gap 1 0.0001778468377037196
    f2, gap 2 0.0003703979189660182



```python
Z = np.exp(scipy.special.logsumexp([-clib.energy(np.array(s), h, Jk) for s in itertools.product(range(naminoacids), repeat=k)]))
df['freq_maxent'] = np.exp([-clib.energy(map_aatonumber(s), h, Jk) for s in kmers])/Z
jsd_maxent = calc_jsd(df['freq_test'], df['freq_maxent'], base=2)
```


```python
hks = fit_global(fks, niter=10, nmcmc=1e6, epsilon=0.1, output=True)
```

    iteration 0
    [2.90512334e-06 9.53816257e-05 6.47926451e-04 1.44376155e-02
     1.24333634e-01]
    iteration 1
    [2.75856055e-06 9.43912499e-05 7.23561702e-04 1.59741977e-02
     1.65529153e-01]
    iteration 2
    [2.60718043e-06 8.72245979e-05 6.53964603e-04 1.06421084e-02
     1.31269296e-01]
    iteration 3
    [2.54442038e-06 8.44896742e-05 8.72001408e-04 1.50421243e-02
     1.82890370e-01]
    iteration 4
    [2.17801555e-06 7.48453272e-05 7.04547413e-04 1.00884850e-02
     1.23185166e-01]
    iteration 5
    [1.81960943e-06 6.74847124e-05 6.53230829e-04 1.47243340e-02
     1.51058786e-01]
    iteration 6
    [1.76266499e-06 5.95955101e-05 6.93436030e-04 1.33729896e-02
     1.71697610e-01]
    iteration 7
    [1.78671962e-06 6.80359522e-05 8.79047080e-04 1.19240230e-02
     2.74568610e-01]
    iteration 8
    [1.45582555e-06 5.27958254e-05 6.72773626e-04 1.51166095e-02
     2.43349670e-01]
    iteration 9
    [1.56038815e-06 5.70676281e-05 7.51670642e-04 1.12625961e-02
     1.54369180e-01]



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
#### cov.ipynb

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
import numpy.testing as npt

plt.style.use('../peptidome.mplstyle')

from lib import *
from lib.maxent import *

from numba import njit

%load_ext autoreload
%autoreload 2
```


```python
L = 9
datasets = ['train', 'test', 'model']
sample_matrices = {}
for dataset in datasets:
    sample_matrices[dataset] =  load_matrix('../maxent/data/%s_matrix_L%i.csv.gz' % (dataset, L))
```


```python
q = naminoacids
```


```python
aacounts = to_aacounts(sample_matrices['train'])
```


```python
cov_kmer = np.cov(aacounts.T/L)
```


```python
arr = np.load('../aafreqpca/data/data.npz')
aa_human = arr['human']
cov_protein = np.cov(aa_human.T)
```


```python
fi = np.mean(aacounts/L, axis=0)
```


```python
cov_multinomial = np.zeros_like(cov_kmer)
for i in range(q):
    for j in range(q):
        if i == j:
            cov_multinomial[i, i] = fi[i]*(1-fi[i])/L
        else:
            cov_multinomial[i, j] = -fi[i]*fi[j]/L
```


```python
fig, axes = plt.subplots(figsize=(9, 3), ncols=3)
for i, matrix in enumerate([cov_protein, cov_kmer-cov_multinomial, cov_multinomial]):
    ax = axes[i]
    im = ax.imshow(matrix, vmin=-matrix.max(), vmax=matrix.max(), cmap='coolwarm')
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.tight_layout()
```


![png](notebook_files/cov_8_0.png)



```python
fig, ax = plt.subplots()
x = cov_protein.flatten()
y = (cov_kmer - cov_multinomial).flatten()
sns.regplot(x, y, ax=ax, scatter_kws=dict(s=1))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print(r_value**2, p_value, slope)
ax.text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
        va='top', ha='left', transform=ax.transAxes)
```

    0.8338968859181958 3.1342288456311137e-157 1.8816198601520313





    Text(0.05, 1.0, 'slope$=1.88$\n$r^2=0.83$')




![png](notebook_files/cov_9_2.png)



```python
fig, ax = plt.subplots(figsize=(3, 2.5))
ax.hist(np.diag(cov_kmer/cov_multinomial), bins=4)
fig.tight_layout()
```


![png](notebook_files/cov_10_0.png)


# Maxent fitting


```python
n1 = calc_n1(aacounts)
n1
```




    array([0.63160537, 0.20116395, 0.43189682, 0.63882889, 0.32990002,
           0.59293757, 0.22709944, 0.39747958, 0.50522771, 0.90904145,
           0.18091782, 0.32592778, 0.56904603, 0.42765345, 0.50409083,
           0.74833342, 0.48167295, 0.54522752, 0.11091272, 0.24103668])




```python
n2 = calc_n2(aacounts)
```


```python
npt.assert_allclose(cov_kmer, (n2 - np.outer(n1, n1))/L**2, atol=1e-8)
```


```python
x0 = np.random.randint(q, size=L)
energy_cov(x0, np.ones(q), np.zeros((q, q)))
```




    -9.0




```python
def fit_cov(n1, n2, L, sampler, q=naminoacids,
            niter=1, epsilon=0.1, pseudocount=1.0,
            prng=None, output=False):
    """ sampler(x0, energy, jump, prng=prng): function returning samples from the distribution """
    if prng is None:
        prng = np.random
    h = np.log(n1/L)
    h -= np.mean(h)
    J = np.zeros_like(n2)
    for iteration in range(niter):
        if output:
            print('iteration %g/%g'%(iteration+1,niter))

        x0 = global_jump(np.zeros(L), q, prng=prng)
        
        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_cov(x, h, J)

        samples = sampler(x0, energy, jump)
        aacounts = to_aacounts(samples)

        n1_model = calc_n1(aacounts)
        n2_model = calc_n2(aacounts)
 
        h -= np.log(n1_model/n1)*epsilon
        J -= np.log(n2_model/n2)*epsilon
        #h -= (n1_model-n1)*epsilon
        #J -= (n2_model-n2)*epsilon
    return h, J
```


```python
prng = np.random
niter = 50
stepsize = 0.05
nsample = L
nsteps = 2e5
nburnin = 1e3
output = True
```


```python
def sampler(*args, **kwargs):
    return mcmcsampler(*args, nsteps=nsteps, nsample=nsample, nburnin=nburnin, **kwargs)
h, J = fit_cov(n1, n2, L, sampler=sampler, niter=niter,
                         epsilon=stepsize, prng=prng, output=output)
```

    iteration 1/50
    iteration 2/50
    iteration 3/50
    iteration 4/50
    iteration 5/50
    iteration 6/50
    iteration 7/50
    iteration 8/50
    iteration 9/50
    iteration 10/50
    iteration 11/50
    iteration 12/50
    iteration 13/50
    iteration 14/50
    iteration 15/50
    iteration 16/50
    iteration 17/50
    iteration 18/50
    iteration 19/50
    iteration 20/50
    iteration 21/50
    iteration 22/50
    iteration 23/50
    iteration 24/50
    iteration 25/50
    iteration 26/50
    iteration 27/50
    iteration 28/50
    iteration 29/50
    iteration 30/50
    iteration 31/50
    iteration 32/50
    iteration 33/50
    iteration 34/50
    iteration 35/50
    iteration 36/50
    iteration 37/50
    iteration 38/50
    iteration 39/50
    iteration 40/50
    iteration 41/50
    iteration 42/50
    iteration 43/50
    iteration 44/50
    iteration 45/50
    iteration 46/50
    iteration 47/50
    iteration 48/50
    iteration 49/50
    iteration 50/50



```python
nsteps_generate = int(sample_matrices['train'].shape[0]*nsample)

@njit
def energy(x):
    return energy_cov(x, h, J)

@njit
def jump(x):
    return local_jump_jit(x, q)

x0 = prng.randint(q, size=L)
model_matrix = mcmcsampler(x0, energy, jump,
                           nsteps=nsteps_generate, nsample=nsample, nburnin=nburnin)
aacounts_model = to_aacounts(model_matrix)
cov_model = np.cov(aacounts_model.T/L)
n1_model = calc_n1(aacounts_model)
n2_model = calc_n2(aacounts_model)
```


```python
np.savetxt('data/model_matrix.csv.gz', model_matrix, fmt='%i')
np.savez('data/Human_L%g.npz'%L, h=h, J=J)
```


```python
@njit
def energy(x):
    return energy_cov(x, h, np.zeros_like(J))

model_ind = mcmcsampler(x0, energy, jump,
                        nsteps=nsteps_generate, nsample=nsample)
aacounts_ind = to_aacounts(model_ind)
cov_ind = np.cov(aacounts_ind.T/L)
```


```python
fig, axes = plt.subplots(figsize=(9, 3), ncols=3)
vmax = (cov_kmer-cov_multinomial).max()
for i, matrix in enumerate([cov_kmer-cov_multinomial,
                            cov_model-cov_multinomial,
                            cov_ind-cov_multinomial]):
    ax = axes[i]
    im = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap='coolwarm')
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.tight_layout()
```


![png](notebook_files/cov_22_0.png)



```python
fig, ax = plt.subplots()
x = (cov_kmer-cov_multinomial).flatten()
y = (cov_model-cov_multinomial).flatten()
sns.regplot(x, y, ax=ax, scatter_kws=dict(s=1))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print(r_value**2, p_value, slope)
ax.text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
        va='top', ha='left', transform=ax.transAxes)
```

    0.9919535133524677 0.0 0.9307956024449778





    Text(0.05, 1.0, 'slope$=0.93$\n$r^2=0.99$')




![png](notebook_files/cov_23_2.png)



```python
fig, ax = plt.subplots()
x = n1
y = n1_model
sns.regplot(x, y, ax=ax, scatter_kws=dict(s=1))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print(r_value**2, p_value, slope)
ax.text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
        va='top', ha='left', transform=ax.transAxes)
```

    0.9992588119640597 1.2524936646632855e-29 0.986280554535421





    Text(0.05, 1.0, 'slope$=0.99$\n$r^2=1.00$')




![png](notebook_files/cov_24_2.png)



```python
fig, ax = plt.subplots()
x = n2.flatten()
y = n2_model.flatten()
sns.regplot(x, y, ax=ax, scatter_kws=dict(s=1))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print(r_value**2, p_value, slope)
ax.text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
        va='top', ha='left', transform=ax.transAxes)
```

    0.9994860511089362 0.0 0.98518132440638





    Text(0.05, 1.0, 'slope$=0.99$\n$r^2=1.00$')




![png](notebook_files/cov_25_2.png)


# Compare with disordered maxent model


```python
aacounts_maxent_disordered = to_aacounts(sample_matrices['model'])
```


```python
cov_maxent_disordered = np.cov(aacounts_maxent_disordered.T/L)
```


```python
fig, axes = plt.subplots(figsize=(6, 3), ncols=2)
vmax = (cov_kmer-cov_multinomial).max()
for i, matrix in enumerate([cov_kmer-cov_multinomial,
                            cov_maxent_disordered-cov_multinomial]):
    ax = axes[i]
    im = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap='coolwarm')
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.tight_layout()
```


![png](notebook_files/cov_29_0.png)



```python
params = np.load('../maxent/data/Human_9.npz')
hi = params['hi']
Jij = params['Jij']
```


```python
fig, ax = plt.subplots()
x = h
y = hi.mean(axis=0)
sns.regplot(x, y, ax=ax, scatter_kws=dict(s=5))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print(r_value**2, p_value, slope)
ax.text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
        va='top', ha='left', transform=ax.transAxes)
```

    0.9988577406112923 6.143007015022327e-28 1.0143513489458438





    Text(0.05, 1.0, 'slope$=1.01$\n$r^2=1.00$')




![png](notebook_files/cov_31_2.png)



```python
for i in range(L):
    Jij[i, i, :, :] = 0.0
    for j in range(i+1, L):
        Jij[i, j, :, :] = 0.0
```


```python
Jmean = 2*np.mean(Jij, axis=(0, 1))*(L**2/(L-1)**2)
```


```python
fig, ax = plt.subplots()
mask = ~np.eye(Jmean.shape[0], dtype=bool)
x = J[mask].flatten()
y = Jmean[mask].flatten()
sns.regplot(x, y, ax=ax, scatter_kws=dict(s=5))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print(r_value**2, p_value, slope)
ax.text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
        va='top', ha='left', transform=ax.transAxes)
```

    0.8321432879285802 1.465528308024156e-148 1.0508870331746092





    Text(0.05, 1.0, 'slope$=1.05$\n$r^2=0.83$')




![png](notebook_files/cov_34_2.png)


# Higher order couplings


```python
aacounts = to_aacounts(sample_matrices['train'])
n3 = calc_n3(aacounts)
```


```python
aacounts = to_aacounts(sample_matrices['test'])
n3_test = calc_n3(aacounts)
```


```python
plt.hist(((n3 - n3_test)/n3).flatten())
```




    (array([  46.,  312.,  942., 1819., 2184., 1571.,  702.,  331.,   66.,
              27.]),
     array([-0.04584061, -0.03459433, -0.02334804, -0.01210175, -0.00085546,
             0.01039083,  0.02163711,  0.0328834 ,  0.04412969,  0.05537598,
             0.06662227]),
     <a list of 10 Patch objects>)




![png](notebook_files/cov_38_1.png)



```python
fig, ax = plt.subplots()
x = n3.flatten()
y = n3_test.flatten()
sns.regplot(x, y, ax=ax, scatter_kws=dict(s=5))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print(r_value**2, ap_value, slope)
ax.text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
        va='top', ha='left', transform=ax.transAxes)
```

    0.9997411188135638 0.0 1.0063049544963758





    Text(0.05, 1.0, 'slope$=1.01$\n$r^2=1.00$')




![png](notebook_files/cov_39_2.png)



```python
n3_model = calc_n3(aacounts_model)
```


```python
plt.hist(((n3_model - n3_test)/n3_test).flatten());
```


![png](notebook_files/cov_41_0.png)



```python
fig, ax = plt.subplots()
x = n3.flatten()
y = n3_model.flatten()
sns.regplot(x, y, ax=ax, scatter_kws=dict(s=5))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print(r_value**2, p_value, slope)
ax.text(0.05, 1.0, 'slope$={1:.2f}$\n$r^2={0:.2f}$'.format(r_value**2, slope),
        va='top', ha='left', transform=ax.transAxes)
```

    0.9987563441196157 0.0 0.9765281785895622





    Text(0.05, 1.0, 'slope$=0.98$\n$r^2=1.00$')




![png](notebook_files/cov_42_2.png)



```python
@njit
def energy_third(x, h, J, J2):
    counts = aacounts_int_jit(x)
    q = len(h)
    e = 0
    for alpha in range(q):
        e -= h[alpha]*counts[alpha]
        for beta in range(alpha, q):
            e -= J[alpha, beta]*counts[alpha]*counts[beta]
            for gamma in range(beta, q):
                e -= J2[alpha, beta, gamma]*counts[alpha]*counts[beta]*counts[gamma]
    return e
```


```python
def fit_third(n1, n2, n3, L, sampler, h=None, J=None, q=naminoacids,
            niter=1, epsilon=0.1, pseudocount=1.0,
            prng=None, output=False):
    """ sampler(x0, energy, jump, prng=prng): function returning samples from the distribution """
    if prng is None:
        prng = np.random
    if h is None:
        h = np.log(n1/L)
        h -= np.mean(h)
    else:
        h = h.copy()
    if J is None:
        J = np.zeros_like(n2)
    else:
        J = J.copy()
    J2 = np.zeros_like(n3)
    for iteration in range(niter):
        if output:
            print('iteration %g/%g'%(iteration+1,niter))

        x0 = global_jump(np.zeros(L), q, prng=prng)
        
        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_third(x, h, J, J2)

        samples = sampler(x0, energy, jump)
        aacounts = to_aacounts(samples)

        n1_model = calc_n1(aacounts)
        n2_model = calc_n2(aacounts)
        n3_model = calc_n3(aacounts)
 
        h -= np.log(n1_model/n1)*epsilon
        J -= np.log(n2_model/n2)*epsilon
        J2 -= np.log(n3_model/n3)*epsilon
    return h, J, J2
```


```python
prng = np.random
niter = 30
stepsize = 0.01
nsample = L
nsteps = 1e6
output = True
```


```python
def sampler(*args, **kwargs):
    return mcmcsampler(*args, nsteps=nsteps, nsample=nsample, **kwargs)
h_third, J_third, J2_third = fit_third(n1, n2, n3, L,
                                       h=h, J=J,
                                       sampler=sampler, niter=niter,
                                       epsilon=stepsize, prng=prng, output=output)
```

    iteration 1/30
    iteration 2/30
    iteration 3/30
    iteration 4/30
    iteration 5/30
    iteration 6/30
    iteration 7/30
    iteration 8/30
    iteration 9/30
    iteration 10/30
    iteration 11/30
    iteration 12/30
    iteration 13/30
    iteration 14/30
    iteration 15/30
    iteration 16/30
    iteration 17/30
    iteration 18/30
    iteration 19/30
    iteration 20/30
    iteration 21/30
    iteration 22/30
    iteration 23/30
    iteration 24/30
    iteration 25/30
    iteration 26/30
    iteration 27/30
    iteration 28/30
    iteration 29/30
    iteration 30/30



```python
plt.hist([J2_third[i, i, i] for i in range(q)])
```




    (array([2., 6., 3., 3., 2., 1., 0., 2., 0., 1.]),
     array([0.00013418, 0.00137673, 0.00261928, 0.00386183, 0.00510439,
            0.00634694, 0.00758949, 0.00883205, 0.0100746 , 0.01131715,
            0.0125597 ]),
     <a list of 10 Patch objects>)




![png](notebook_files/cov_47_1.png)



```python
plt.hist(J2_third.flatten())
```




    (array([  36.,   66.,  267.,  969., 2250., 2269., 1476.,  478.,  168.,
              21.]),
     array([-0.02482474, -0.01995712, -0.01508949, -0.01022186, -0.00535423,
            -0.00048661,  0.00438102,  0.00924865,  0.01411628,  0.0189839 ,
             0.02385153]),
     <a list of 10 Patch objects>)




![png](notebook_files/cov_48_1.png)



```python
nsteps_generate = int(sample_matrices['train'].shape[0]*nsample)

@njit
def energy(x):
    return energy_third(x, h_third, J_third, J2_third)

@njit
def jump(x):
    return local_jump_jit(x, q)

x0 = prng.randint(q, size=L)
model_matrix = mcmcsampler(x0, energy, jump,
                           nsteps=nsteps_generate, nsample=nsample)
```


```python
np.savetxt('data/model_third_matrix.csv.gz', model_matrix, fmt='%i')
np.savez('data/Human_third_L%g.npz'%L, h=h_third, J=J_third, J2=J2_third)
```


```python
aacounts = to_aacounts(sample_matrices['test'])
n3_model = calc_n3(aacounts, L)
```


```python

```
