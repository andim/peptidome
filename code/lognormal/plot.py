import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from lib import *

df = pd.read_csv('data/freq1.csv')

xk = np.log10(df['freq'])
pk = np.asarray(df['freq'])
mu = np.sum(pk * xk)
var = np.sum(pk * (xk-mu)**2)
cm3 = np.sum(pk * (xk-mu)**3)
print(mu, var, cm3)

fig, ax = plt.subplots(figsize=(5, 3))
dx = 0.075
ax.hist(np.log10(df['freq']), bins=np.arange(-2.0, -0.8, dx), weights=df['freq'], label='data\n$\gamma_1=%g$'%round(cm3/var**1.5, 2))
xmin = mu-4*var**.5
xmax = mu+4*var**.5
x = np.linspace(xmin, xmax)
ax.plot(x, scipy.stats.norm.pdf(x, mu, var**.5)*dx, label='lognormal\n$\mu=%g$,\n$\sigma=%g$'%(round(mu, 2), round(var**.5, 2)))
ax.set_xlim(xmin, xmax)
ax.legend(loc='upper right')
ax.set_xlabel('$log_{10}$ likelihood');
ax.set_ylabel('frequency');

fig.tight_layout()
fig.savefig('../../paper/images/lognormalaa.pdf')
plt.show()
