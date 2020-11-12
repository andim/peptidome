---
layout: post
title: Benchmark of free energy calculation via thermodynamic integration
---

How much sampling is needed for accurate results?

{% include post-image-gallery.html filter="free_energy_benchmark/" %}

### Code 
#### free_energy_benchmark.ipynb

```python
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

from numba import njit
```


```python
mcmc_kwargs = dict(nsteps=1e6, nsample=10, nburnin=1e3)
```


```python
L = 9
params = np.load('../maxent/data/Human_reference_%i.npz'%L)
hi = params['hi']
Jij = params['Jij']
```


```python
Fs = [Fpotts_thermodynamic_integration(hi, Jij, integration_intervals=3, mcmc_kwargs=mcmc_kwargs) for i in range(10)]
```


```python
Fexact = Fpotts_thermodynamic_integration(hi, Jij, integration_intervals=6,
                                     mcmc_kwargs=dict(nsteps=2e6, nsample=10, nburnin=1e3))
```


```python
F0 = -np.sum(np.log(np.sum(np.exp(hi), axis=1)))
F0
```




    -28.03156529665821




```python
fig, ax = plt.subplots()
ax.hist(Fs)
ax.axvline(F0, c='k')
ax.axvline(Fexact, c='g')
fig.savefig('main.png')
```


![png](notebook_files/free_energy_benchmark_6_0.png)



```python
np.std(Fs, ddof=1)/(np.mean(Fs)-F0)
```




    0.04384399496514539




```python

```
