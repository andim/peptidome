---
layout: post
title: Inference of maxent models
---

Infering and benchmarking of Maxent models.

{% include post-image-gallery.html filter="maxent_symmetric/" %}

### Code 
#### generic.ipynb

# Maximum entropy modeling

We consider a distribution $P(\boldsymbol \sigma)$, where $\boldsymbol \sigma$ is an N-dimensional state vector. We search for the distribution which maximizes the entropy subject to some constraints on the expectation value of a (smallish) number of observables:

$$\langle \sum_{\boldsymbol \sigma} P(\boldsymbol \sigma) f_\mu(\boldsymbol \sigma)\rangle = f_\mu^{emp}$$

Using the method of Lagrange multipliers we can show that the distributions take the form:

$$P(\boldsymbol \sigma) = \frac{1}{Z} \exp\left[ -\sum_\mu \lambda_\mu f_\mu(\boldsymbol \sigma) \right]$$



```python
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from lib import *
%matplotlib inline
```


```python
class MaxEntModel:
    def __init__(self, N, q, constraints, prng=None):
        """
        N: number of spins
        q: number of possible spin states
        constraints: list of constraints
        """
        self.N = N
        self.q = q
        self.constraints = constraints
        self.lambdas = np.zeros_like(constraints)
        if prng is None:
            self.prng = np.random
        else:
            self.prng = prng
    def energy(self, sigma):
        return np.sum(self.lambdas * np.array([c(sigma) for c in self.constraints]))
    def sample(self, n):
        'n: number of samples'
        def jump(x):
            return self.prng.randint(self.q, size=self.N)
        x0 = jump(np.zeros(self.N))
        return mcmcsampler(x0, self.energy, jump, n)
```


```python
def gen_field_constraint(index):
    def constraint(x):
        return x[index]
    return constraint
```


```python
m = MaxEntModel(5, 2, [gen_field_constraint(i) for i in range(5)])
m.lambdas = np.array([0.1, 0.2, 0.3, -0.1, -0.3])
m.energy([0, 1, 1, 0, 0])
```




    0.5




```python
m.sample(10)
```




    array([[0, 1, 0, 0, 0],
           [1, 0, 0, 1, 1],
           [1, 0, 0, 1, 1],
           [1, 0, 1, 1, 0],
           [1, 1, 1, 0, 1],
           [0, 1, 0, 1, 0],
           [1, 1, 1, 1, 0],
           [0, 0, 0, 1, 0],
           [1, 1, 0, 1, 1]])




```python

```
#### mfmaxent.ipynb

```python
import itertools, copy
import json
import numpy as np
import scipy.misc
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('../peptidome.mplstyle')

import evcouplings.align, evcouplings.couplings

import sys
sys.path.append('..')
from lib import *
from lib import maxent
```


```python
k = 4
q = len(aminoacids)
```


```python
humanseqs = [s for s in fasta_iter(human, returnheader=False)]
```


```python
train, test = train_test_split(humanseqs, test_size=0.5)
```


```python
empirical_kmers = [string[i:i+k] for string in train for i in range(len(string)-k+1) if isvalidaa(string[i:i+k])]
```


```python
seqmat = np.array([list(kmer) for kmer in empirical_kmers])
```


```python
map_ = map_ = {c: i for i, c in enumerate(aminoacids)}
mapped_seqmat = evcouplings.align.map_matrix(seqmat, map_)
```


```python
fi = evcouplings.align.frequencies(mapped_seqmat, np.ones(len(seqmat)), num_symbols=len(aminoacids))
```


```python
plt.plot(fi);
```


![png](notebook_files/mfmaxent_8_0.png)



```python
fij = evcouplings.align.pair_frequencies(mapped_seqmat, np.ones(len(seqmat)), num_symbols=len(aminoacids), fi=fi)
```


```python
cij = evcouplings.couplings.compute_covariance_matrix(fi, fij)
```


```python
cij.shape
```




    (76, 76)




```python
sns.heatmap(cij, vmin=-0.1, vmax=0.1, cmap='PuOr')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f60fd53d7b8>




![png](notebook_files/mfmaxent_12_1.png)



```python
invC = np.linalg.inv(cij)
```


```python
sns.heatmap(invC)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f60fc0b4710>




![png](notebook_files/mfmaxent_14_1.png)



```python
q = len(aminoacids)
N = k
cij.shape, N*(q-1)
```




    ((76, 76), 76)




```python
C = cij.copy()
D = np.diag(np.diagonal(cij))
A = C.copy()
A[np.diag_indices_from(A)] = 0.0
```


```python
np.diagonal(cij).min(), A.max()
```




    (0.01196938119559453, 0.002982074930161451)




```python
Dinv = np.diag(1/np.diagonal(cij))
A = cij.copy()
A[np.diag_indices_from(A)] = 0.0
invCapprox = Dinv-Dinv@A@Dinv
```


```python
mask = ~np.eye(invC.shape[0],dtype=bool)
plt.plot(invC[mask].flatten(), invCapprox[mask].flatten(), 'k.')
```




    [<matplotlib.lines.Line2D at 0x7f60fbf42ba8>]




![png](notebook_files/mfmaxent_19_1.png)



```python
sns.heatmap(invC-invCapprox)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f60fbefce48>




![png](notebook_files/mfmaxent_20_1.png)



```python
def JijfrominvC(invC):
    Jij = evcouplings.couplings.reshape_invC_to_4d(invC, seqmat.shape[1], len(aminoacids))
    Jij_zerogauge = evcouplings.couplings.model._zero_sum_gauge(Jij)
    return Jij_zerogauge
```


```python
Jij_zerogauge = JijfrominvC(invC)
Jij_zerogauge_approx = JijfrominvC(invCapprox)
plt.plot(Jij_zerogauge.flatten(), Jij_zerogauge_approx.flatten(), '.')
x = np.linspace(min(Jij_zerogauge.flatten()), max(Jij_zerogauge.flatten()))
plt.plot(x, x, 'k')
```




    [<matplotlib.lines.Line2D at 0x7f60f4cfdf98>]




![png](notebook_files/mfmaxent_22_1.png)



```python
rhoij = np.zeros(fij.shape)
for i in range(k):
    for j in range(k):
        rhoij[i, j] = fij[i, j]/np.outer(fi[i], fi[j]) - 1.0
mask = np.abs(Jij_zerogauge) > 1e-5
plt.plot(Jij_zerogauge[mask].flatten(), -rhoij[mask].flatten(), '.')
```




    [<matplotlib.lines.Line2D at 0x7f60f4cb4358>]




![png](notebook_files/mfmaxent_23_1.png)



```python
Jij_corr = -rhoij
Jij_zerogauge_corr = evcouplings.couplings.model._zero_sum_gauge(Jij_corr)
```


```python
hi = evcouplings.couplings.fields(Jij_zerogauge, fi)
print(hi)

#for i in range(hi.shape[0]):
#    for a in range(hi.shape[1]):
#        hi[i, a] += np.sum(Jij[i, :, a, :])
```

    [[ 0.89752269 -0.07281932  0.58232387  0.93548344  0.36732537  0.87021794
       0.01471589  0.53722642  0.76484283  1.29562027 -0.2191712   0.3288629
       0.77948882  0.55312893  0.73763077  1.08276793  0.67396597  0.81351711
      -0.72518618  0.08618009]
     [ 0.88676034 -0.11539266  0.57989394  0.92885253  0.36789733  0.87097104
       0.0157558   0.54564175  0.77961963  1.29668317 -0.28425886  0.3405876
       0.76150369  0.55258767  0.74558946  1.08921996  0.68329167  0.80825549
      -0.72641106  0.09296561]
     [ 0.87748452 -0.08881751  0.57791746  0.93020174  0.35761     0.85947842
       0.02370312  0.5371459   0.77849383  1.29444809 -0.28944578  0.34481987
       0.76633545  0.56138609  0.74716838  1.0899552   0.6877514   0.80782562
      -0.72887105  0.06493039]
     [ 0.87909548 -0.04198681  0.57624625  0.92331099  0.35083295  0.85984981
       0.00389778  0.52282629  0.7670217   1.3024009  -0.27931894  0.34848666
       0.78222357  0.55458928  0.73347018  1.08513932  0.68127697  0.80689981
      -0.73302811  0.05557514]]



```python
N = k
q = 20
cij_flat = cij.reshape(N, q-1, N, q-1)
```


```python
sns.heatmap(cij_flat[0, :, 1, :])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f60f4c52be0>




![png](notebook_files/mfmaxent_27_1.png)



```python
sns.heatmap(Jij_zerogauge_approx[0, 1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f60f4604b00>




![png](notebook_files/mfmaxent_28_1.png)



```python
sns.heatmap(Jij_zerogauge[0, 2])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f60f51f1630>




![png](notebook_files/mfmaxent_29_1.png)



```python
import numba
@numba.jit(nopython=True)
def energy_potts(x, hi, Jij):
    e = 0
    for i in range(len(x)):
        e += hi[i, x[i]]
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            e += Jij[i, j, x[i], x[j]]
    return -e
```


```python
jump = lambda x: np.random.randint(q, size=k)
x0 = jump(0)
samples = mcmcsampler(x0, lambda x: energy_potts(x, hi, -Jij_zerogauge), jump, 1e6)
```


```python
fi_model = evcouplings.align.frequencies(samples, np.ones(len(samples)), num_symbols=q)
fij_model = evcouplings.align.pair_frequencies(samples, np.ones(len(samples)), num_symbols=q, fi=fi_model)
cij_model = evcouplings.couplings.compute_covariance_matrix(fi_model, fij_model).flatten()
```


```python
plt.plot(fi.flatten(), fi_model.flatten(), 'o')
```




    [<matplotlib.lines.Line2D at 0x7f60f422c898>]




![png](notebook_files/mfmaxent_33_1.png)



```python
plt.plot(cij.flatten(), cij_model.flatten(), 'o')
x = [-0.01, 0.01]
plt.plot(x, x, 'k')
plt.xlim(-0.01, 0.01)
plt.ylim(-0.01, 0.01)
```




    (-0.01, 0.01)




![png](notebook_files/mfmaxent_34_1.png)



```python
kmers = [''.join(s) for s in itertools.product(aminoacids, repeat=k)]
df = pd.DataFrame.from_dict(dict(seq=kmers, freq=np.zeros(len(kmers))))
df.set_index('seq', inplace=True)
df['freq'] = df['freq'].add(maxent.count(train, k)['freq'], fill_value=0.0)
df['freq_test'] = np.zeros(len(kmers))
df['freq_test'] = df['freq_test'].add(maxent.count(test, k)['freq'], fill_value=0.0)
```


```python
df['freq_maxent'] = np.exp([-energy_potts(evcouplings.align.map_matrix(list(s), map_), hi, -Jij_zerogauge) for s in kmers])
df['freq_maxent'] /= np.sum(df['freq_maxent'])
jsd_maxent = calc_jsd(df['freq_test'], df['freq_maxent'], base=2)
jsd_maxent, scipy.stats.entropy(df['freq_maxent'], base=2)
```




    (0.01468193683722711, 16.6121511830779)




```python
Sind = sum([scipy.stats.entropy(fi[i], base=2) for i in range(fi.shape[0])])
Itot = 0.0
for i in range(fij.shape[0]):
    for j in range(i+1, fij.shape[0]):
        I = np.sum(fij[i, j]*np.log2(fij[i, j]/np.outer(fi[i], fi[j])))
        print(i, j, I)
        Itot += I
Sind, Itot, Sind-Itot
```

    0 1 0.013772404691858177
    0 2 0.009740576343510084
    0 3 0.010410108709941416
    1 2 0.01370965677736796
    1 3 0.009771762985039182
    2 3 0.013689177643913818





    (16.701907472487587, 0.07109368715163064, 16.630813785335956)




```python
scatter = lambda x, y, ax: plotting.density_scatter(x, y, ax=ax, s=0.5, bins=100,
                                           trans=lambda x: np.log(x+1e-8),
                                           norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=50 if k ==3 else 400),
                                           cmap='viridis')
#scatter = lambda x, y, ax: ax.scatter(x, y, s=1, alpha=1, edgecolor=None)

fig, axes = plt.subplots(figsize=(3.5, 2.0), ncols=2, sharex=True, sharey=True)
axes[0].set_ylabel('test set')

for ax, column, xlabel in [(axes[0], 'freq_maxent','maxent prediction'),
                           (axes[1], 'freq', 'training set')
                            ]:
    scatter(df[column], df['freq_test'], ax)
    ax.set_xlabel(xlabel)
    jsd = calc_jsd(df['freq_test'], df[column], base=2)
    entropy = scipy.stats.entropy(df[column], base=2)
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


![png](notebook_files/mfmaxent_38_0.png)



```python
plt.hist(np.diagonal(Jij_zerogauge, axis1=2, axis2=3).flatten(), bins=20)
plt.yscale('log')
```


![png](notebook_files/mfmaxent_39_0.png)



```python
plt.hist(Jij_zerogauge.flatten(), bins=20)
plt.yscale('log')
```


![png](notebook_files/mfmaxent_40_0.png)



```python
sns.heatmap(pd.DataFrame(Jij_zerogauge[0, 4], index=list(aminoacids), columns=list(aminoacids)))#.reshape(N*q, N*q))
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-43-0e300336a31f> in <module>
    ----> 1 sns.heatmap(pd.DataFrame(Jij_zerogauge[0, 4], index=list(aminoacids), columns=list(aminoacids)))#.reshape(N*q, N*q))
    

    IndexError: index 4 is out of bounds for axis 1 with size 4



```python
alignment = evcouplings.align.Alignment(seqmat,
                                        sequence_ids=[str(i)+'/1-4' if i == 0 else '' for i in range(len(seqmat))],
                                        alphabet=evcouplings.align.ALPHABET_PROTEIN_NOGAP)
```


```python
mfdca = evcouplings.couplings.MeanFieldDCA(alignment)
```


```python
fit = mfdca.fit(pseudo_count=0.5)
```


```python
fit.J_ij.shape
```


```python
plt.imshow(evcouplings.couplings.model._zero_sum_gauge(fit.J_ij)[0, 1])
plt.colorbar()
```


```python

```
#### analyze_fitall.ipynb

```python
import itertools, copy
import json
import numpy as np
import scipy.misc
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('../peptidome.mplstyle')

import evcouplings.align, evcouplings.couplings

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *
```


```python
h = pd.read_csv('data/Human_N9_h.csv', index_col=0)
Jks = pd.read_csv('data/Human_N9_Jk.csv', index_col=0)
Jks
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>J0</th>
      <th>J1</th>
      <th>J2</th>
      <th>J3</th>
      <th>J4</th>
      <th>J5</th>
      <th>J6</th>
      <th>J7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AA</th>
      <td>0.232767</td>
      <td>0.176694</td>
      <td>0.141982</td>
      <td>0.177703</td>
      <td>0.119090</td>
      <td>0.118538</td>
      <td>0.132371</td>
      <td>0.119879</td>
    </tr>
    <tr>
      <th>AC</th>
      <td>-0.063087</td>
      <td>-0.036587</td>
      <td>-0.055878</td>
      <td>-0.039001</td>
      <td>-0.053162</td>
      <td>-0.053999</td>
      <td>-0.081942</td>
      <td>-0.018395</td>
    </tr>
    <tr>
      <th>AD</th>
      <td>-0.102322</td>
      <td>-0.034965</td>
      <td>-0.059437</td>
      <td>-0.056501</td>
      <td>-0.027444</td>
      <td>-0.044379</td>
      <td>-0.041642</td>
      <td>-0.039879</td>
    </tr>
    <tr>
      <th>AE</th>
      <td>-0.025501</td>
      <td>-0.035273</td>
      <td>-0.031560</td>
      <td>-0.050066</td>
      <td>-0.040061</td>
      <td>-0.015409</td>
      <td>-0.033394</td>
      <td>-0.027260</td>
    </tr>
    <tr>
      <th>AF</th>
      <td>0.060716</td>
      <td>-0.059644</td>
      <td>-0.038790</td>
      <td>-0.035201</td>
      <td>-0.038278</td>
      <td>-0.028070</td>
      <td>-0.041062</td>
      <td>-0.032721</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>YS</th>
      <td>-0.003065</td>
      <td>-0.074109</td>
      <td>-0.015768</td>
      <td>-0.060618</td>
      <td>-0.034259</td>
      <td>-0.049508</td>
      <td>-0.053662</td>
      <td>-0.041771</td>
    </tr>
    <tr>
      <th>YT</th>
      <td>0.016021</td>
      <td>-0.022783</td>
      <td>0.012062</td>
      <td>-0.021468</td>
      <td>-0.016917</td>
      <td>-0.036559</td>
      <td>-0.033014</td>
      <td>0.007947</td>
    </tr>
    <tr>
      <th>YV</th>
      <td>-0.018982</td>
      <td>0.003307</td>
      <td>0.004703</td>
      <td>0.095129</td>
      <td>-0.046605</td>
      <td>-0.010358</td>
      <td>0.033803</td>
      <td>-0.016848</td>
    </tr>
    <tr>
      <th>YW</th>
      <td>0.068311</td>
      <td>0.024986</td>
      <td>0.065827</td>
      <td>0.015152</td>
      <td>-0.008751</td>
      <td>0.061589</td>
      <td>0.151537</td>
      <td>0.108424</td>
    </tr>
    <tr>
      <th>YY</th>
      <td>0.174683</td>
      <td>0.125903</td>
      <td>0.184713</td>
      <td>0.162090</td>
      <td>0.146841</td>
      <td>0.147405</td>
      <td>0.153054</td>
      <td>0.159920</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 8 columns</p>
</div>




```python
params = np.load('data/Human_N9.npz')
h = params['h']
Jk = params['Jk']
```


```python
N = 9
q = naminoacidsz
prng = np.random
nmcmc = 1e6
aas_arr = np.array(list(aminoacids))
```


```python
def energy(x):
    return clib.energy(x, h, Jk)
x0 = prng.randint(q, size=N)
def jump(x):
    return local_jump(x, q)
samples = mcmcsampler(x0, energy, jump, nmcmc*10, nsample=10)
samples = [''.join(aas_arr[s]) for s in samples]
```


```python
seed = 1234
prng = np.random.RandomState(seed)
proteome = proteome_path('Human')
seqs = [s for s in fasta_iter(proteome, returnheader=False)]
train, test = train_test_split(seqs, test_size=0.5, random_state=prng)
```


```python
df1s = []
df2s = []
for i, seqs in enumerate([train, test, samples]):
    df1 = pseudocount_f1(seqs)
    df1s.append(df1)
    df2s.append([list(pseudocount_f2(seqs, 2, gap, df1)['freq'])  for gap in range(0, N-1)])
```


```python
train_kmers = list(to_kmers(train, k=N))
test_kmers = list(to_kmers(test, k=N))
```


```python
mapped_seqs
```




    array([[10, 10,  9, ...,  5, 15,  6],
           [17,  0,  9, ...,  5, 15,  6],
           [ 0, 17, 14, ...,  5, 15,  6],
           ...,
           [ 0,  0, 13, ...,  1, 11, 15],
           [ 0,  9, 13, ...,  1,  0,  3],
           [ 0,  8, 10, ..., 18,  0, 13]])




```python
fis = []
cijs = []
for i, seqs in enumerate([train_kmers, test_kmers, samples]):
    seqs_arr = np.array([list(sample) for sample in seqs])
    map_ = map_ = {c: i for i, c in enumerate(aminoacids)}
    mapped_seqs = evcouplings.align.map_matrix(seqs_arr, map_)
    fi = evcouplings.align.frequencies(mapped_seqs, np.ones(len(seqs)), num_symbols=q)
    fij = evcouplings.align.pair_frequencies(mapped_seqs, np.ones(len(seqs)), num_symbols=q, fi=fi)
    cij = evcouplings.couplings.compute_covariance_matrix(fi, fij).flatten()
    fis.append(fi.copy())
    cijs.append(cij.copy())
```


```python
fig, axes = plt.subplots(figsize=(6, 3), ncols=2)
axes[0].plot(fis[2], fis[1], 'o')
axes[0].set_xlabel('maxent frequency')
axes[1].plot(fis[0], fis[1], 'o')
axes[1].set_xlabel('training frequency')
for ax in axes:
    ax.set_ylabel('test frequency')
fig.tight_layout()
```


![png](notebook_files/analyze_fitall_10_0.png)



```python
lim = 0.005
fig, axes = plt.subplots(figsize=(6, 3), ncols=2)
axes[0].plot(cijs[2].flatten(), cijs[1].flatten(), 'o', ms=1)
axes[0].set_xlabel('maxent cij')
axes[1].plot(cijs[0].flatten(), cijs[1].flatten(), 'o', ms=1)
axes[1].set_xlabel('training cij')
for ax in axes:
    ax.set_ylabel('test cij')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], 'k')
fig.tight_layout()
```


![png](notebook_files/analyze_fitall_11_0.png)



```python

```
#### energy-benchmarking.ipynb

```python
import copy 
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from lib import *
%matplotlib inline
%load_ext Cython
```


```python
q = 20
N = 10
h = np.zeros(q)
J = np.zeros((N-1, q, q))
```


```python
def energy_fast(s, h, J):
    energy = np.sum(h[s])
    for i in range(N):
        for j in range(i, N):
            energy += J[j-i-1, s[i], s[j]]
    return energy
```


```python
%%timeit
s = np.random.randint(q, size=N)
energy_fast(s, h, J)
```

    33.9 µs ± 799 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)



```cython
%%cython
cimport numpy as np
def energy_cython(np.ndarray[long, ndim=1] s, np.ndarray[double, ndim=1] h, np.ndarray[double, ndim=3] J):
    cdef int N = len(s)
    cdef double energy = 0.0
    cdef int i, j
    for i in range(N):
        energy += h[s[i]]
        for j in range(i, N):
            energy += J[j-i-1, s[i], s[j]]
    return energy
```


```python
%%timeit
s = np.random.randint(q, size=N)
energy_cython(s, h, J)
```

    2.91 µs ± 38.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)



```python
aas_arr = np.array(list(aminoacids))
h = dict(zip(aminoacids, np.zeros(q)))
J0 = np.zeros((len(aminoacids), len(aminoacids)))
J0 = pd.DataFrame(np.asarray(J0), index=list(aminoacids), columns=list(aminoacids)).to_dict()
Jk = [J0]
for gap in range(1, N):
    Jk.append(copy.deepcopy(J0))

```


```python
%%timeit
s = ''.join(np.random.choice(aas_arr, size=N))
energy_ising(s, h, Jk)
```

    33.7 µs ± 2.31 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)



```python
s = np.random.randint(q, size=N)
s
```




    array([ 0, 16, 11,  7, 17, 12, 16,  3, 11,  1])




```python
''.join(np.array(list(aminoacids))[s])
```




    'ATNIVPTENC'




```python
aas_arr = np.array(list(aminoacids))

```


```python
%%timeit
s = ''.join(np.random.choice(aas_arr, size=N))
```

    11.2 µs ± 67.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)



```python
%%timeit
s = aas_arr[np.random.randint(q, size=N)]
#s = ''.join(aas_arr[s])

```

    1.65 µs ± 52.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)



```python

```
#### maxent.ipynb

```python
import sys
sys.path.append('..')
import itertools, copy
import json
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('../peptidome.mplstyle')

from lib import *
from base import *
```


```python
output = True
aas_arr = np.array(list(aminoacids))
N = 4
seed = 1234
prng = np.random.RandomState(seed)

```


```python
humanseqs = [s for s in fasta_iter(human, returnheader=False)]
train, test = train_test_split(humanseqs, test_size=0.5)

```


```python
# evaluate empirical observables for fitting
df0 = count(train, 1)
df1 = count(train, 2, gap=0)
dfgap1 = count(train, 2, gap=1)
dfgap2 = count(train, 2, gap=2)
```


```python
h, Jk = fit_potts(df0, [df1, dfgap1, dfgap2], nmcmc=1e6, niter=30, epsilon=0.1, prng=prng, output=output, N=N)
```

    [ 0.45400779 -0.66050852  0.05611162  0.46870351 -0.20250439  0.39748563
     -0.52087632 -0.03259483  0.25030339  0.80545478 -0.74174209 -0.21726625
      0.35364766  0.07429697  0.23695342  0.62506977  0.17829996  0.29101695
     -1.30176657 -0.51409249]
    iteration 0
    f1 1.548136412473544e-06
    f2, gap 0 0.002401566853782732
    f2, gap 1 0.0018014411864347973
    f2, gap 2 0.0019662817228899058
    iteration 1
    f1 2.6460712837714895e-06
    f2, gap 0 0.0020081820945886814
    f2, gap 1 0.0015029653544621014
    f2, gap 2 0.0016196840151457676
    iteration 2
    f1 5.644796886231619e-06
    f2, gap 0 0.001593894769203269
    f2, gap 1 0.0011119418518341444
    f2, gap 2 0.0012559946365008322
    iteration 3
    f1 4.9896998263220985e-06
    f2, gap 0 0.0012959723035681327
    f2, gap 1 0.0009880889623188655
    f2, gap 2 0.001093922169590812
    iteration 4
    f1 5.6958500904870736e-06
    f2, gap 0 0.0010616233306976392
    f2, gap 1 0.0007901491994967216
    f2, gap 2 0.0009664777109721381
    iteration 5
    f1 4.852053641590939e-06
    f2, gap 0 0.000861826237087767
    f2, gap 1 0.0006929872125033048
    f2, gap 2 0.0007378712068947736
    iteration 6
    f1 4.829883367998935e-06
    f2, gap 0 0.0007616803315415308
    f2, gap 1 0.0005458098824819909
    f2, gap 2 0.0007108559727005716
    iteration 7
    f1 5.324866594919233e-06
    f2, gap 0 0.0005673491504112425
    f2, gap 1 0.0004144854592168397
    f2, gap 2 0.0006103340569476015
    iteration 8
    f1 5.01899476677927e-06
    f2, gap 0 0.00045998040404187014
    f2, gap 1 0.0004121310459417326
    f2, gap 2 0.0004893900545394005
    iteration 9
    f1 4.259055610558267e-06
    f2, gap 0 0.00038967032027807853
    f2, gap 1 0.00032737353665926444
    f2, gap 2 0.0004499636671384638
    iteration 10
    f1 4.860344025865403e-06
    f2, gap 0 0.0003280971089555705
    f2, gap 1 0.000260673601083313
    f2, gap 2 0.00037087924417709896
    iteration 11
    f1 8.438685588806649e-06
    f2, gap 0 0.00025221902648853937
    f2, gap 1 0.00022256694286771625
    f2, gap 2 0.00030657243754823237
    iteration 12
    f1 5.361268190875379e-06
    f2, gap 0 0.00021230833713349696
    f2, gap 1 0.00019579092058847323
    f2, gap 2 0.0003147085142236149
    iteration 13
    f1 5.434373125814059e-06
    f2, gap 0 0.00022735078572080658
    f2, gap 1 0.00018366150799912438
    f2, gap 2 0.00029854499699931794
    iteration 14
    f1 7.251259996734966e-06
    f2, gap 0 0.00015303056862415516
    f2, gap 1 0.0001611238759376954
    f2, gap 2 0.00026487768489788684
    iteration 15
    f1 5.124866541439963e-06
    f2, gap 0 0.00015096841246278216
    f2, gap 1 0.00013853527050005393
    f2, gap 2 0.0002419607613620416
    iteration 16
    f1 7.218949656037291e-06
    f2, gap 0 0.00012947885368629075
    f2, gap 1 0.00013627498281314654
    f2, gap 2 0.00023578829366847795
    iteration 17
    f1 5.4763269413449785e-06
    f2, gap 0 0.00013574437825096376
    f2, gap 1 0.0001342434755818682
    f2, gap 2 0.00021019752906954172
    iteration 18
    f1 5.260449023115211e-06
    f2, gap 0 0.00011090315172730098
    f2, gap 1 0.00010909468534347336
    f2, gap 2 0.00017923185944991794
    iteration 19
    f1 5.214901536860589e-06
    f2, gap 0 8.826459986553017e-05
    f2, gap 1 0.00010282227160806246
    f2, gap 2 0.00019322646096022638
    iteration 20
    f1 7.353466795579548e-06
    f2, gap 0 8.160542205897787e-05
    f2, gap 1 9.315056958994269e-05
    f2, gap 2 0.00016382104386739144
    iteration 21
    f1 4.502606604702198e-06
    f2, gap 0 8.248294146892695e-05
    f2, gap 1 9.630026905426908e-05
    f2, gap 2 0.0001983386035923123
    iteration 22
    f1 6.305391741953893e-06
    f2, gap 0 9.644836603169498e-05
    f2, gap 1 0.0001091622633290176
    f2, gap 2 0.00019255534614438409
    iteration 23
    f1 8.381059943051194e-06
    f2, gap 0 7.269200367290658e-05
    f2, gap 1 0.00010936554764831229
    f2, gap 2 0.00016591974925993477
    iteration 24
    f1 7.542094184622236e-06
    f2, gap 0 6.373121095777164e-05
    f2, gap 1 8.31558455987215e-05
    f2, gap 2 0.00019779426497731583
    iteration 25
    f1 6.226832254903351e-06
    f2, gap 0 7.525271058860124e-05
    f2, gap 1 0.00010203269011657942
    f2, gap 2 0.000187139756690101
    iteration 26
    f1 7.278105865153421e-06
    f2, gap 0 7.372330622026563e-05
    f2, gap 1 9.160004044013792e-05
    f2, gap 2 0.00019400048817127063
    iteration 27
    f1 6.481490296453591e-06
    f2, gap 0 6.424515008145165e-05
    f2, gap 1 8.795082144529974e-05
    f2, gap 2 0.00017525610856562192
    iteration 28
    f1 5.888328231075019e-06
    f2, gap 0 7.405111286319058e-05
    f2, gap 1 9.445416451924458e-05
    f2, gap 2 0.00017719893073482297
    iteration 29
    f1 7.104793354059407e-06
    f2, gap 0 6.50681336847835e-05
    f2, gap 1 9.727214956718809e-05
    f2, gap 2 0.0001575284475242545



```python
k = 4
df4 = count(train, k)
df4 = df4.merge(count(test, k), right_index=True, left_index=True, suffixes=['_train', '_test'])
jsd_test = calc_jsd(df4['freq_train'], df4['freq_test'])
jsd_flat = calc_jsd(df4['freq_test'], np.ones_like(df4['freq_test']))
```


```python
with open('../../data/triplet-human.json', 'r') as f:
    tripletparams = json.load(f)
kmers = df4.index
df4['freq_ind'] = np.array([10**(loglikelihood_independent(s, **tripletparams)) for s in kmers])
df4['freq_mc'] = np.array([10**(loglikelihood_mc(s, **tripletparams)) for s in kmers])
df4['freq_tri'] = np.array([10**(loglikelihood_triplet(s, **tripletparams)) for s in kmers])
jsd_ind = calc_jsd(df4['freq_test'], df4['freq_ind'])
jsd_mc = calc_jsd(df4['freq_test'], df4['freq_mc'])
jsd_tri = calc_jsd(df4['freq_test'], df4['freq_tri'])
```


```python
q = len(aminoacids)
Z = np.exp(scipy.special.logsumexp([-clib.energy(np.array(s), h, Jk) for s in itertools.product(range(q), repeat=k)]))
df4['freq_maxent'] = np.exp([-clib.energy(map_aatonumber(s), h, Jk) for s in kmers])/Z
jsd_maxent = calc_jsd(df4['freq_test'], df4['freq_maxent'])
```


```python
print('flat', jsd_flat, 'ind', jsd_ind, 'mc', jsd_mc, 'model', jsd_maxent, 'tri', jsd_tri, 'test', jsd_test, )
```

    flat 0.11000058339844129 ind 0.019343485765939982 mc 0.01307517982350593 model 0.009142299971584104 tri 0.00867386621822569 test 0.007527760831336014



```python
q = len(aminoacids)
N = 4
nmcmc = 1e6
prng = np.random
def jump(x):
    return prng.randint(q, size=N)
def energy(x):
    return clib.energy(x, h, Jk)
x0 = jump(None)
samples = mcmcsampler(x0, energy, jump, nmcmc)
samples = [''.join(aas_arr[s]) for s in samples]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-3b422524ff45> in <module>
    ----> 1 q = len(aminoacids)
          2 N = 4
          3 nmcmc = 1e6
          4 prng = np.random
          5 def jump(x):


    NameError: name 'aminoacids' is not defined



```python
dfm0 = df0.merge(count(samples, 1), left_index=True, right_index=True)
x = np.linspace(0.0, 0.1)
plt.plot(x, x, 'k')
plt.scatter(dfm0['freq_x'], dfm0['freq_y'])
dfm0['logfold'] = np.log(dfm0['freq_x']/dfm0['freq_y'])
np.abs(dfm0['logfold']).mean()
```




    0.0056646042369886815




![png](notebook_files/maxent_10_1.png)



```python
dfm1 = df1.merge(count(samples, 2), left_index=True, right_index=True)
x = np.linspace(0.0, 0.11**2)
plt.plot(x, x, 'k')
plt.scatter(dfm1['freq_x'], dfm1['freq_y'])
dfm1['logfold'] = np.log(dfm1['freq_x']/dfm1['freq_y'])
np.abs(dfm1['logfold']).mean()
```




    0.019856011693173588




![png](notebook_files/maxent_11_1.png)



```python
dfmgap1 = dfgap1.merge(count(samples, 2, gap=1), left_index=True, right_index=True)
x = np.linspace(0.0, 0.11**2)
plt.plot(x, x, 'k')
plt.scatter(dfmgap1['freq_x'], dfmgap1['freq_y'])
dfmgap1['logfold'] = np.log(dfmgap1['freq_x']/dfmgap1['freq_y'])
np.abs(dfmgap1['logfold']).mean()
```




    0.022609933480723373




![png](notebook_files/maxent_12_1.png)



```python
dfmgap2 = dfgap2.merge(count(samples, 2, gap=2), left_index=True, right_index=True)
x = np.linspace(0.0, 0.11**2)
plt.plot(x, x, 'k')
plt.scatter(dfmgap2['freq_x'], dfmgap2['freq_y'])
dfmgap2['logfold'] = np.log(dfmgap2['freq_x']/dfmgap2['freq_y'])
np.abs(dfmgap2['logfold']).mean()
```




    0.029568890847527802




![png](notebook_files/maxent_13_1.png)



```python
fig, axes = plt.subplots(figsize=(3.8, 2.0), ncols=2, sharex=True, sharey=True)
ax = axes[0]
ax.scatter(df4['freq_maxent'], df4['freq_test'], s=0.5, label='maxent', alpha=.1)
ax.set_xlabel('maxent prediction')
ax.set_ylabel('test set')
ax = axes[1]
ax.scatter(df4['freq_train'], df4['freq_test'], s=0.5, label='train', alpha=.1)
ax.set_xlabel('training set')
ax.set_ylabel('test set')
x = np.logspace(-7, -3)
for ax in axes:
    ax.plot(x, x, 'k')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(x), max(x))
    #plt.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
fig.tight_layout()
fig.savefig('4mer-comparison.png', dpi=600)
```


![png](notebook_files/maxent_14_0.png)



```python
dfJk = [pd.DataFrame.from_dict(J) for J in Jk]
```


```python
fig, axes = plt.subplots(figsize=(20, 6), ncols=len(dfJk), sharex=True, sharey=True)
for i, dfJ in enumerate(dfJk):
    sns.heatmap(dfJ, vmin=-0.5, vmax=0.5, cmap='RdBu_r', ax=axes[i])
fig.tight_layout()
```


![png](notebook_files/maxent_16_0.png)



```python
k = 5
#kmers = list(itertools.product(aminoacids, repeat=k))
df = counter_to_df(count_kmers_proteome(human, k))
df = df[~df['seq'].str.contains('U|B|X|Z')]
df = df.set_index('seq')
kmers = df.index
exp = np.array([float(df.loc[''.join(s)]) for s in kmers])
Z = np.exp(scipy.special.logsumexp([-energy_ising(s, h, Jk) for s in itertools.product(aminoacids, repeat=k)]))
ising = np.exp([-energy_ising(s, h, Jk) for s in kmers])/Z
tri = np.array([10**(loglikelihood_triplet(s, **tripletparams)) for s in kmers])
mc = np.array([10**(loglikelihood_mc(s, **tripletparams)) for s in kmers])
ind = np.array([10**(loglikelihood_independent(s, **tripletparams)) for s in kmers])
rising = np.corrcoef(ising, exp)[1, 0]
rind = np.corrcoef(ind, exp)[1, 0]
rtri = np.corrcoef(tri, exp)[1, 0]
rtri, rising, np.corrcoef(mc, exp)[1, 0], rind
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-18-9020411d0d1b> in <module>
          6 kmers = df.index
          7 exp = np.array([float(df.loc[''.join(s)]) for s in kmers])
    ----> 8 Z = np.exp(scipy.special.logsumexp([-energy_ising(s, h, Jk) for s in itertools.product(aminoacids, repeat=k)]))
          9 ising = np.exp([-energy_ising(s, h, Jk) for s in kmers])/Z
         10 tri = np.array([10**(loglikelihood_triplet(s, **tripletparams)) for s in kmers])


    <ipython-input-18-9020411d0d1b> in <listcomp>(.0)
          6 kmers = df.index
          7 exp = np.array([float(df.loc[''.join(s)]) for s in kmers])
    ----> 8 Z = np.exp(scipy.special.logsumexp([-energy_ising(s, h, Jk) for s in itertools.product(aminoacids, repeat=k)]))
          9 ising = np.exp([-energy_ising(s, h, Jk) for s in kmers])/Z
         10 tri = np.array([10**(loglikelihood_triplet(s, **tripletparams)) for s in kmers])


    ~/repos/peptidome/code/lib/main.py in energy_ising(s, h, Jk)
        382 def energy_ising(s, h, Jk):
        383     "energy of a translation invariant ising model"
    --> 384     energy = sum(h[c] for c in s)
        385     for k, J in enumerate(Jk):
        386         for i in range(len(s)-1-k):


    ~/repos/peptidome/code/lib/main.py in <genexpr>(.0)
        382 def energy_ising(s, h, Jk):
        383     "energy of a translation invariant ising model"
    --> 384     energy = sum(h[c] for c in s)
        385     for k, J in enumerate(Jk):
        386         for i in range(len(s)-1-k):


    IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices



```python
fig = plt.figure(figsize=(4, 4))
plt.scatter(ind, exp, s=1, label='independent, r=%1.2f'%rind)
#plt.scatter(mc, exp, s=1)
plt.scatter(tri, exp, s=1, label='tri, r=%1.2f'%rtri)
x = np.logspace(-7, -3)
plt.xlabel('predicted')
plt.ylabel('observed')
plt.plot(x, x, 'k')
plt.xlim(min(x), max(x))
plt.ylim(min(x), max(x))
plt.legend()
plt.xscale('log')
plt.yscale('log')
fig.tight_layout()
fig.savefig('plots/modelfits-4mer-tri.png', dpi=300)
```


```python
pd.DataFrame(index=[key for key in h], data=[h[key] for key in h], columns=['h'])
```


```python
plot_sorted(df['count'])
```


```python
doublets = [''.join(s) for s in itertools.product(list(aminoacids), repeat=2)]
```


```python
df = pd.DataFrame(index=doublets, data=[Jk[0][s[0]][s[1]] for s in doublets], columns=['J0'])
for i in range(1, len(Jk)):
    df['J%g'%i] = [Jk[i][s[0]][s[1]] for s in doublets]
```


```python
from functools import reduce
```


```python
df
```


```python
reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True), [pd.DataFrame.from_dict(
    {i+j: Jk[gap][i][j] for i in Jk[gap].keys() for j in Jk[gap][i].keys()},
    orient='index')
           for gap in range(len(Jk))])
```


```python
pd.merge?
```


```python

```
#### triplet_frequencies.ipynb

```python
import itertools, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *
```


```python
def triplet_frequencies(matrix, num_symbols=2, pseudocount=0):
    """
    Calculate triplet frequencies of symbols.

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    num_symbols : int
        Number of different symbols contained in alignment
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters.

    Returns
    -------
    np.array
        Matrix of size L x L x L x num_symbols x num_symbols x num_symbols containing
        relative triplet frequencies of all character combinations
    """
    N, L = matrix.shape
    fijk = pseudocount*np.ones((L, L, L, num_symbols, num_symbols, num_symbols))
    for s in range(N):
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    fijk[i, j, k, matrix[s, i], matrix[s, j], matrix[s, k]] += 1

    # normalize frequencies by the number
    # of sequences
    fijk /= (N+pseudocount)

    return fijk

```


```python
@jit(nopython=True)
def triplet_frequencies_fast(matrix, fij, fi, num_symbols=2, pseudocount=0):
    """
    Calculate triplet frequencies of symbols.

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    num_symbols : int
        Number of different symbols contained in alignment
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters.

    Returns
    -------
    np.array
        Matrix of size L x L x L x num_symbols x num_symbols x num_symbols containing
        relative triplet frequencies of all character combinations
    """
    N, L = matrix.shape
    fijk = pseudocount*np.ones((L, L, L, num_symbols, num_symbols, num_symbols))
    for s in range(N):
        for i in range(L):
            for j in range(i+1, L):
                for k in range(j+1, L):
                    fijk[i, j, k, matrix[s, i], matrix[s, j], matrix[s, k]] += 1
    # set permuted indices to same value
    for i in range(L):
        for j in range(i+1, L):
            for k in range(j+1, L):
                for alpha in range(num_symbols):
                    for beta in range(num_symbols):
                        for gamma in range(num_symbols):
                            value = fijk[i, j, k, alpha, beta, gamma]
                            fijk[i, k, j, alpha, gamma, beta] = value
                            fijk[j, i, k, beta, alpha, gamma] = value
                            fijk[j, k, i, beta, gamma, alpha] = value
                            fijk[k, j, i, gamma, beta, alpha] = value
                            fijk[k, i, j, gamma, alpha, beta] = value

    # normalize frequencies by the number
    # of sequences
    fijk /= (N+pseudocount)
    
    # set the frequency of a doublet
    # in position i to the respective single-site
    # frequency of alpha in position i
    for i in range(L):
        for j in range(i+1, L):
            for alpha in range(num_symbols):
                for beta in range(num_symbols):
                    fijk[i, j, i, alpha, beta, alpha] = fij[i, j, alpha, beta]
                    fijk[i, i, j, alpha, alpha, beta] = fij[i, j, alpha, beta]
                    fijk[j, i, j, beta, alpha, beta] = fij[i, j, alpha, beta]
                    fijk[j, j, i, beta, beta, alpha] = fij[i, j, alpha, beta]
    
    # set the frequency of a triplet
    # in position i to the respective single-site
    # frequency of alpha in position i
    for i in range(L):
        for alpha in range(num_symbols):
            fijk[i, i, i, alpha, alpha, alpha] = fi[i, alpha]

    return fijk

```


```python
matrix = np.array([[0, 1, 0],
                   [1, 1, 1]])
fi = frequencies(matrix, num_symbols=2)
fij = pair_frequencies(matrix, num_symbols=2, fi=fi)
fijk = triplet_frequencies_fast(matrix, fij=fij, fi=fi, num_symbols=2)
fijk_ref = triplet_frequencies(matrix, num_symbols=2)
```


```python
plt.imshow(fijk_ref[0,:,:,0,0,0])
```




    <matplotlib.image.AxesImage at 0x7ff3ad521a58>




![png](notebook_files/triplet_frequencies_4_1.png)



```python
fijk[0,:,:,0,0,0]
```




    array([[0.5, 0. , 0.5],
           [0. , 0. , 0. ],
           [0.5, 0. , 0. ]])




```python
np.testing.assert_array_equal(fijk, fijk_ref)
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-50-7531535a1889> in <module>
    ----> 1 np.testing.assert_array_equal(fijk, fijk_ref)
    

    ~/.conda/envs/py3/lib/python3.6/site-packages/numpy/testing/_private/utils.py in assert_array_equal(x, y, err_msg, verbose)
        934     __tracebackhide__ = True  # Hide traceback for py.test
        935     assert_array_compare(operator.__eq__, x, y, err_msg=err_msg,
    --> 936                          verbose=verbose, header='Arrays are not equal')
        937 
        938 


    ~/.conda/envs/py3/lib/python3.6/site-packages/numpy/testing/_private/utils.py in assert_array_compare(comparison, x, y, err_msg, verbose, header, precision, equal_nan, equal_inf)
        844                                 verbose=verbose, header=header,
        845                                 names=('x', 'y'), precision=precision)
    --> 846             raise AssertionError(msg)
        847     except ValueError:
        848         import traceback


    AssertionError: 
    Arrays are not equal
    
    Mismatched elements: 12 / 216 (5.56%)
    Max absolute difference: 0.5
    Max relative difference: 1.
     x: array([[[[[[0.5, 0. ],
               [0. , 0. ]],
    ...
     y: array([[[[[[0.5, 0. ],
               [0. , 0. ]],
    ...



```python

```
#### jsd.ipynb

```python
import sys, copy, itertools
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
```


```python
names = ['Human', 'Mouse', 'Vaccinia', 'InfluenzaB', 'InfluenzaA', 'CMV', 'HCV', 'HSV1',
       'DENV', 'HIV', 'EBV', 'Ebola', 'Ecoli', 'Tuberculosis', 'Listeria',
       'Burkholderia', 'Meningococcus', 'StrepA', 'Hpylori',
       'Lyme', 'Tetanus', 'Leprosy', 'Malaria', 'Chagas',
       'OnchocercaVolvulus']
```


```python
dfs = {}
for name in names:
    path = 'data/%s.csv'%name
    if os.path.exists(path):
        df = pd.read_csv(path)
        #df.set_index('seq', inplace=True)
        dfs[name] = df
```


```python
N = len(dfs)
distances_uniform = np.zeros(N)
distances = np.zeros((N, N))
for i, namei in enumerate(dfs):
    df1 = dfs[namei]
    f1 = df1['freq_maxent']
    f2 = np.ones_like(f1)
    distances_uniform[i] = calc_jsd(f1, f2)
    for j, namej in enumerate(dfs):
        df2 = dfs[namej]
        dfm = pd.merge(df1, df2, on='seq', suffixes=['_1', '_2'])
        f1, f2 = np.asarray(dfm['freq_maxent_1']), np.asarray(dfm['freq_maxent_2'])
        distances[i, j] = calc_jsd(f1, f2, base=2)
```


```python
names = list(dfs.keys())
fullnames = list(proteomes.loc[names]['fullname'])
```


```python
df = pd.DataFrame(distances, index=names, columns=names, copy=True)
```


```python
fig = plt.figure(figsize=(8, 6))
sns.heatmap(df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9d2314f160>




![png](notebook_files/jsd_6_1.png)



```python
type_to_color = {'virus' : colors[0],
                 'bacterium' : colors[1],
                 'parasite' : colors[2],
                 'vertebrate' : colors[3]
                }
```


```python
cond_distances = scipy.spatial.distance.squareform(0.5*(distances+distances.T))
Z = scipy.cluster.hierarchy.linkage(cond_distances, method='average', optimal_ordering=True)
typecolors = np.array([type_to_color[proteomes.loc[name]['type']] for  name in names])
cg = sns.clustermap(df*(9.0/4.0), row_linkage=Z, col_linkage=Z, cbar_kws=dict(label='JSD in bits'), figsize=(8, 8))
for label, color in zip(cg.ax_heatmap.get_yticklabels(), typecolors[cg.dendrogram_col.reordered_ind]):
    label.set_color(color)
for label, color in zip(cg.ax_heatmap.get_xticklabels(), typecolors[cg.dendrogram_row.reordered_ind]):
    label.set_color(color)
ax = cg.ax_col_dendrogram
for i, (type_, color) in enumerate(type_to_color.items()):
    ax.text(1.07, 0.7-i*0.12, type_, color=color, transform=cg.ax_col_dendrogram.transAxes)
```


![png](notebook_files/jsd_8_0.png)



```python
df = pd.DataFrame(distances, index=names, columns=names, copy=True)
df['Uniform'] = distances_uniform
df = df.append(pd.Series(distances_uniform, name='Uniform', index=names))
df.iloc[-1, -1] = 0.0
rand = np.random.RandomState(seed=5)
mds = sklearn.manifold.MDS(n_components=2, dissimilarity='precomputed',
                           n_init=20, max_iter=500,
                           random_state=rand)
transformed = mds.fit_transform(df*4.5)
```


```python
type_to_color = {'virus' : colors[0],
                 'bacterium' : colors[1],
                 'parasite' : colors[2],
                 'vertebrate' : colors[3],
                 'uniform' : colors[4]
                }

fig, ax = plt.subplots(figsize=(8, 8))

typecolors = [type_to_color[proteomes.loc[name]['type']] for name in names]
typecolors.append(type_to_color['uniform'])
ax.scatter(transformed[:, 0], transformed[:, 1], color=typecolors)
offsets = 0.01*np.ones(len(df.index))
for index in [1, 4, 13, 14, 15, 23]:
    offsets[index] = -0.02
for i, name in enumerate(df.index):
    ax.text(transformed[i, 0], transformed[i, 1]+offsets[i], name, ha='center', color=typecolors[i])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
sns.despine(fig, left=True, bottom=True)

for i, (type_, color) in enumerate(type_to_color.items()):
    ax.text(0.8, 0.9-i*0.025, type_, color=color, transform=ax.transAxes)

fontprops = fm.FontProperties(size=12)
scalebar = AnchoredSizeBar(ax.transData,
                           0.5, '0.5 bit', 'lower left', 
                           pad=0.5,
                           color='k',
                           frameon=False,
                           size_vertical=0.005,
                           label_top=True,
                           fontproperties=fontprops)

ax.add_artist(scalebar)
fig.tight_layout()
```


![png](notebook_files/jsd_10_0.png)



```python
uniform = df['Uniform']
human = df['Human']
ys = uniform[names]
xs = human[names]
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(xs, ys, color=typecolors[:-1])
#offset=0.00
#for i, name in enumerate(names):
#    ax.text(xs[i], ys[i]+offset, name, ha='center', color=typecolors[i])
ax.plot([0, 0.3], [0, 0.3], 'k-')
ax.set_xlabel('JSD(proteome, human)')
ax.set_ylabel('JSD(proteome, uniform)')
ax.set_aspect('equal')
ax.set_xlim(-0.01, 0.3)
ax.set_ylim(-0.01, 0.3)
for i, (type_, color) in enumerate(type_to_color.items()):
    ax.text(0.8, 0.2-i*0.03, type_, color=color, transform=ax.transAxes)
sns.despine(fig)
fig.tight_layout()
fig.savefig('dist_human_uniform.svg')
```


![png](notebook_files/jsd_11_0.png)



```python

```
#### fitmaxent_all.py

```python
import itertools, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
import clib
from base import *

output = True
aas_arr = np.array(list(aminoacids))
N = 9

proteome = proteome_path('Human')

seed = 1234
prng = np.random.RandomState(seed)

seqs = [s for s in fasta_iter(proteome, returnheader=False)]
train, test = train_test_split(seqs, test_size=0.5, random_state=prng)
#train, test = seqs, seqs

# evaluate empirical observables for fitting
#df1 = pseudocount_f1(train)
#df2s = [pseudocount_f2(train, 2, gap, df1)  for gap in range(0, N-1)]
df1 = count(train, 1)
df2s = [count(train, 2, gap=gap)  for gap in range(0, N-1)]
print('fit')
h, Jk = fit_potts(df1, df2s, nmcmc=1e6, niter=10, epsilon=0.1, prng=prng, output=output)
print(Jk)

save('Human_N%g'%N, h, Jk)
np.savez('data/Human_N%g.npz'%N, h=h, Jk=Jk)

```
#### fitmaxent.py

```python
import itertools, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
import clib
from base import *

output = True
aas_arr = np.array(list(aminoacids))
N = 4
for name, row in proteomes.iterrows():
    if name == 'Human':
    #if not os.path.exists('data/%s.csv' % name):
        print(name)
        seed = 1234
        prng = np.random.RandomState(seed)

        proteome = datadir + row['path']
        seqs = [s for s in fasta_iter(proteome, returnheader=False)]
        #train, test = train_test_split(seqs, test_size=0.5)
        train, test = seqs, seqs

        # evaluate empirical observables for fitting
        #df0 = count(train, 1)
        df0 = pseudocount_f1(train)
        #df1 = count(train, 2, gap=0)
        df1 = pseudocount_f2(train, 2, 0, df0) 
        #dfgap1 = count(train, 2, gap=1)
        dfgap1 = pseudocount_f2(train, 2, 1, df0) 
        #dfgap2 = count(train, 2, gap=2)
        dfgap2 = pseudocount_f2(train, 2, 2, df0) 

        print('fit')
        h, Jk = fit_potts(df0, [df1, dfgap1, dfgap2], nmcmc=1e5, niter=20, epsilon=0.1, prng=prng, output=output, N=N)

        print('compare on 4mers')
        k = 4
        df4 = count(train, k)

        #df4_count = count(test, k)
        kmers = [''.join(s) for s in itertools.product(aminoacids, repeat=k)]
        df4_test = pd.DataFrame.from_dict(dict(seq=kmers, count=np.ones(len(kmers))))
        df4_test.set_index('seq', inplace=True)
        df4_count = counter_to_df(count_kmers_iterable(test, k), norm=False)
        df4_count.set_index('seq', inplace=True)
        df4_test = df4_test.add(df4_count, fill_value=0.0)
        df4_test['freq'] = df4_test['count'] / np.sum(df4_test['count'])

        m, jsd_test = calc_logfold(df4, df4_test)
        jsd_flat = calc_jsd(df4_test['freq'], np.ones_like(df4_test['freq']))

        tripletparams = calc_tripletmodelparams(proteome)
        kmers = df4_test.index
        df4_test['freq_ind'] = np.array([10**(loglikelihood_independent(s, **tripletparams)) for s in kmers])
        df4_test['freq_mc'] = np.array([10**(loglikelihood_mc(s, **tripletparams)) for s in kmers])
        df4_test['freq_tri'] = np.array([10**(loglikelihood_triplet(s, **tripletparams)) for s in kmers])
        jsd_ind = calc_jsd(df4_test['freq'], df4_test['freq_ind'])
        jsd_mc = calc_jsd(df4_test['freq'], df4_test['freq_mc'])
        jsd_tri = calc_jsd(df4_test['freq'], df4_test['freq_tri'])

        q = len(aminoacids)
        Z = np.exp(scipy.special.logsumexp([-clib.energy(np.array(s), h, Jk) for s in itertools.product(range(q), repeat=k)]))
        df4_test['freq_maxent'] = np.exp([-clib.energy(map_aatonumber(s), h, Jk) for s in kmers])/Z
        jsd_maxent = calc_jsd(df4_test['freq'], df4_test['freq_maxent'])
        #nmcmc = 1e7
        #prng = np.random
        #def jump(x):
        #    return prng.randint(q, size=N)
        #def energy(x):
        #    return clib.energy(x, h, Jk)
        #x0 = jump(None)
        #samples = mcmcsampler(x0, energy, jump, nmcmc)
        #samples = [''.join(aas_arr[s]) for s in samples]
        #df4_model = count(samples, 4)
        #m, jsd_model = calc_logfold(df4_test, df4_model)
        print('4mer', 'test', jsd_test, 'maxent', jsd_maxent,
              'flat', jsd_flat, 'ind', jsd_ind, 'mc', jsd_mc, 'tri', jsd_tri)

        df4_test.to_csv('data/%s.csv' % name)

        print(h, Jk)
        save(name, h, Jk)

```
#### base.py

```python
import itertools

import sys
sys.path.append('..')
from lib import *
from lib import clib
from lib.maxent import *

import numpy as np

def save(name, h, Jk):
    aas_arr = np.array(list(aminoacids))
    dfh = pd.DataFrame(index=aas_arr, data=h, columns=['h'])
    #dfJk = pd.DataFrame(data=Jk, columns=range(len(Jk)))
    doublets = [''.join(s) for s in itertools.product(list(aminoacids), repeat=2)]
    dfJk = pd.DataFrame(index=doublets,
                        data=[Jk[0,aatonumber(s[0]),aatonumber(s[1])] for s in doublets],
                        columns=['J0'])
    for i in range(1, len(Jk)):
        dfJk['J%g'%i] = [Jk[i,aatonumber(s[0]),aatonumber(s[1])] for s in doublets]

    dfh.to_csv('data/%s_h.csv' % name)
    dfJk.to_csv('data/%s_Jk.csv' % name)

```
