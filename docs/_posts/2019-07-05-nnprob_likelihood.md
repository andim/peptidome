---
layout: post
title: How does neighbor density relate to likelihoods ?
---

How does the likelihood of a sequence relate to the probability that any of its neighboring sequences (here defined as differing by a single amino acid) relate to each other?

It turns out that for an independent site model (to which the amino acid distribution is close to empirically) the total probability of all neighboring sites is completely determined by the likelihood of the sequence itself as long as the variance is small. We can even get an analytical formula for the scaling, which turns out to be somewhat sublinear (with an exponent of approximately ~ 1-1/k). In the figure below this is shown for an independent site model for 9mers based on the amino acid frequencies of the human proteome.

Practically this finding helps us relate our findings in terms of likelihoods to more traditional bioinformatic approaches using nearest neighbor densities.

{% include post-image-gallery.html filter="nnprob_likelihood/" %}

### Code 
#### math.ipynb


# How does neighbor density relate to likelihoods ?


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('../peptidome.mplstyle')
#plt.style.use('talk')

import sys
sys.path.append('..')

from lib import *
```

## check summation formula


```python
def neighbors(sigma, S):
    for i in range(len(sigma)):
        for s in range(S):
            if not sigma[i] == s:
                yield np.asarray(list(sigma[:i]) + [s] + list(sigma[i+1:]))

S = 2
k = 2
sigma_lognormal = 1.0
pi = np.random.lognormal(sigma=sigma_lognormal, size=S)
pi /= pi.sum()
sigma = np.random.randint(0, S, k)
nsigma = np.prod(pi[sigma])*(np.sum(1/pi[sigma]) - k)
nsigma_sum = np.sum(np.fromiter((np.prod(pi[sigmap]) for sigmap in neighbors(sigma, S)), np.float))
print(nsigma, nsigma_sum)
```

    0.43379441914720135 0.4337944191472014


## correlation between probabilities


```python
S = 20
k = 10
sigma = 0.3
pi = np.random.lognormal(sigma=sigma, size=S)
pi /= pi.sum()
psigmas = []
psigmaps = []
Nsample = 10000
for i in range(Nsample):
    sigma = np.random.randint(0, S, k)
    psigma = np.prod(pi[sigma])
    i = np.random.randint(0, k)
    sigmai = np.random.choice([s for s in range(0, S) if s != sigma[i]])
    sigmap = np.asarray(list(sigma[:i]) + [sigmai] + list(sigma[i+1:]))
    psigmap = np.prod(pi[sigmap])
    psigmas.append(psigma)
    psigmaps.append(psigmap)
psigmaps = np.asarray(psigmaps)
psigmas = np.asarray(psigmas)
```


```python
fig, ax = plt.subplots(figsize=(3.42, 3.42))
trans = np.log10
ax.plot(trans(psigmas), trans(psigmaps), '.', label='data', ms=1)
ax.plot(trans(psigmas), trans(psigmas), '-', label='linear')
#psigmas_theory = np.linspace(0.5*S**(-k), 2*S**(-k))
#ax.plot(np.log10(psigmas_theory), np.log10(psigmas_theory)+np.log10(k*(S-1)-S*(psigmas_theory*S**k-1)),
#        '-', label='theory')
#ax.plot([-k*np.log10(S)], [-k*np.log10(S) + np.log10(k*(S-1))], 'o')
ax.set_xlabel('$\log_{10} p(\sigma)$')
ax.set_ylabel("$\log_{10} p(\sigma')$")
```




    Text(0, 0.5, "$\\log_{10} p(\\sigma')$")




![png](notebook_files/math_6_1.png)



```python
rhop = np.corrcoef(psigmaps, psigmas)[0, 1]
rhop, np.corrcoef(np.log(psigmaps), np.log(psigmas))[0, 1]
```




    (0.8491838235254919, 0.8955826296953464)



## generate samples from P_uniform(psigma, nsigma)


```python
df = Counter(human, 1).to_df(norm=True, clean=True)
paa = np.asarray(df['freq'])
paa
```




    array([0.02132759, 0.06577845, 0.07012693, 0.06314819, 0.09970845,
           0.08332527, 0.01217275, 0.05643573, 0.05963846, 0.07101057,
           0.02662735, 0.02305051, 0.05347764, 0.0473157 , 0.05724314,
           0.04330068, 0.04767035, 0.02626757, 0.0365297 , 0.03584496])




```python
S = 20
k = 9

Nn = k*(S-1)
N = float(S**k)

#sigma_lognormal = 0.4
#pi = np.random.lognormal(sigma=sigma_lognormal, size=S)
pi = paa
#pi = np.random.uniform(size=S)
pi /= pi.sum()


psigmas = []
nsigmas = []
Nsample = 100000
for i in range(Nsample):
    sigma = np.random.randint(0, S, k)
    psigma = np.prod(pi[sigma])
    nsigma = np.prod(pi[sigma])*np.sum((1-pi[sigma])/pi[sigma])
    psigmas.append(psigma)
    nsigmas.append(nsigma)
nsigmas = np.asarray(nsigmas)
psigmas = np.asarray(psigmas)
rho = np.corrcoef(psigmas, nsigmas)[1, 0]
print(r'$\rho_{p(\sigma), n(\sigma)}$:', rho)
```

    $\rho_{p(\sigma), n(\sigma)}$: 0.9906201713914548



```python
np.var(np.log(psigmas))/k, np.var(np.log(pi))
```




    (0.2650286172753722, 0.2637393500109682)




```python
sigmasq = np.var(np.log(pi))*k
np.var(psigmas), (np.exp(sigmasq)-1)/N**2
```




    (1.5312729094932873e-23, 3.714200465543339e-23)




```python
fig, ax = plt.subplots(figsize=(3.42, 3.42))
ax.plot(np.log10(psigmas), np.log10(nsigmas), '.', label='data', ms=1)
ax.plot(np.log10(psigmas), np.log10(psigmas)+np.log10(k*(S-1)), '-', label='linear')
#factor = 20
#psigmas_theory = np.linspace(S**(-k)/factor, factor*S**(-k))
psigmas_theory = np.linspace(min(psigmas), max(psigmas))
ax.plot(np.log10(psigmas_theory),
        #np.log10(psigmas_theory) + np.log10(Nn - S*(psigmas_theory*float(S**k) - 1)),
        np.log10(psigmas_theory) + np.log10(Nn - S*np.log(psigmas_theory*N)),
        'k-', label='theory', lw=2)
#ax.axvline(-k*np.log10(S), c='k')
ax.set_xlabel('$\log_{10} p(\sigma)$\n Log-Likelihood')
ax.set_ylabel("Log-Neighborlikelihood\n $\log_{10} n(\sigma) = \log_{10} \sum_{\sigma' \sim \sigma} p(\sigma')$")
ax.legend()
fig.tight_layout()
fig.savefig('main.png')
fig.savefig('../../paper/images/nnproblikelihood.pdf')

slope, intercept = np.polyfit(np.log10(psigmas), np.log10(nsigmas), 1)
rho1 = 1 - S/(k*(S-1))
print('slope, prediction', slope, rho1)
nsigmavar_pred = np.var(psigmas)* Nn**2 * rho1**2
nsigmavar_pred2 = Nn*np.var(psigmas)*(1+(Nn-1)*rhop)
print('nsigmavar (sampled, upper, pred):',
      np.var(nsigmas),
      np.var(psigmas)*Nn**2,
      nsigmavar_pred,
      nsigmavar_pred2)
nsigmabar = np.sum(nsigmas*psigmas/np.sum(psigmas))
nsigmabar_upper = Nn*np.mean(psigmas**2)*N
nsigmabar_pred = Nn/N + N*Nn*rho1*np.var(psigmas)
nsigmabar_pred2 = Nn/N + N*(Nn-1.5*S)*np.var(psigmas)
nsigmabar_pred3 = Nn/N + N*(np.var(psigmas) * nsigmavar_pred2)**.5
print('nsigmabar (lower, sampled, upper):', Nn/N, nsigmabar, nsigmabar_upper)
print('prediction (-S, -3/2S, rhop)', nsigmabar_pred, nsigmabar_pred2, nsigmabar_pred3)
```

    slope, prediction 0.8653734425950762 0.8830409356725146
    nsigmavar (sampled, upper, pred): 2.7861268219362906e-19 4.477595114649322e-19 3.4914553609356446e-19 3.806250426060438e-19
    nsigmabar (lower, sampled, upper): 3.33984375e-10 1.376673489089933e-09 1.6786749617905207e-09
    prediction (-S, -3/2S, rhop) 1.5178420867874503e-09 1.4394409138213942e-09 1.5700597083794556e-09



![png](notebook_files/math_13_1.png)



```python
bins = np.linspace(0.0, np.percentile(nsigmas, 99), 100)
fig, ax = plt.subplots()
histkwargs = dict(bins=bins, histtype='step')
ax.hist(psigmas*k*(S-1), label='$P(\sigma)$ rescaled', **histkwargs)
ax.hist(nsigmas, label='$n(\sigma)$', **histkwargs)
ax.legend()
ax.axvline(1/S**k * k*(S-1), c='k')
ax.set_xlim(min(bins), max(bins))
ax.set_yticks([])
ax.set_ylabel('Density')
ax.set_xlabel('Probability');
```


![png](notebook_files/math_14_0.png)


## TODO

- What about longer distances? (Second, third neighbours etc.?) Do things generalize?
- Test with rough Mount Fuji?


```python

```
