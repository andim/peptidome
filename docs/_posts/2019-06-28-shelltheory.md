---
layout: post
title: Immunogenicity only within a shell of likelihoods?
---

An exploration of a shell model of immunogenicity.

{% include image-gallery.html filter="shelltheory/" %}

### Code 
#### plot.py

```python
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

plt.style.use('../peptidome.mplstyle')

fig, ax = plt.subplots()
xmin, xmax = -13.5, -9.5
p = np.logspace(xmin, xmax)
N = 2e7
k = 9
NT = 1e3#k*19
ax.plot(p, 1-np.exp(- p * NT * N), label='tolerated')
ND = 1e4#k*(k-1)*19**2
ax.plot(p, 1-np.exp(- p * ND * N), label='detectable')
#ax.plot(p, (1-np.exp(- p * ND * N))*np.exp(- p * NT * N), label='immunogenic')
mu = -1.26
sigma = 0.18
ax.plot(p, scipy.stats.norm.pdf(np.log10(p), k*mu, k**.5*sigma), label='pathogen')
ax.plot(p, (1-np.exp(- p * ND*N))*np.exp(- p * NT*N)*scipy.stats.norm.pdf(np.log10(p), k*mu, k**.5*sigma), label='epitopes')
ax.legend(loc='upper left')
ax.set_xscale('log')
ax.set_xlabel('peptide frequency p')
ax.set_ylabel('Probability')
ax.set_xlim(10**xmin, 10**xmax)
fig.tight_layout()
plt.show()
fig.savefig('main.png')

```
