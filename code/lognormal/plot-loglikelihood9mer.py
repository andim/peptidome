import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats

df = pd.read_csv('data/freq1.csv')
xk = np.log10(df['freq'])
pk = np.asarray(df['freq'])
mu = np.sum(pk * xk)
var = np.sum(pk * (xk-mu)**2)

phuman = np.load('data/loglikelihood-k9-human.npy')

k = 9
fig, ax = plt.subplots(figsize=(5, 3))
phuman = phuman[~np.isnan(phuman)]
counts, bins = np.histogram(phuman, bins=70)
axinset =  inset_axes(ax, width='30%', height='40%', loc='upper left')
axinset.tick_params(labelleft=False, labelright=True, left=False, right=True)
x = 0.5*(bins[:-1]+bins[1:])
y = counts/(np.sum(counts)*np.diff(bins))
ax.plot(x, y, label='data')
axinset.plot(x, y)
xmin, xmax = round(k*mu-5*(k*var)**.5), round(k*mu+5*(k*var)**.5)
x = np.linspace(xmin, xmax)
y = scipy.stats.norm.pdf(x, k*mu, (k*var)**.5)
ax.plot(x, y, label='model')
axinset.plot(x, y)
axinset.set_yscale('log')
for a in [ax, axinset]:
    a.set_xlim(xmin, xmax)
ax.set_xlabel('$log_{10}$ likelihood');
ax.set_ylabel('frequency');
#ax.set_yscale('log')
ax.legend()
fig.tight_layout()
plt.show()
