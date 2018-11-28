import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.style.use('../peptidome.mplstyle')

import plotting

df = pd.read_csv('data/freq1.csv')

xk = np.log10(df['freq'])
pk = np.asarray(df['freq'])
mu = np.sum(pk * xk)
var = np.sum(pk * (xk-mu)**2)
cm3 = np.sum(pk * (xk-mu)**3)
print(mu, var, cm3)

fig, axes = plt.subplots(figsize=(3.4, 6), nrows=3)

base = 20
xfactor = 1.0/np.log10(base)

ax = axes[0]
dx = 0.075
#counts, bins = np.histogram(np.log10(df['freq']), bins=np.arange(-2.0, -0.8, dx), weights=df['freq'])
#print(np.diff(bins))
#x = 0.5*(bins[:-1]+bins[1:])
#y = counts/(np.sum(counts)*np.diff(bins))
#ax.bar(x, y)
ax.hist(xfactor*np.log10(df['freq']), bins=xfactor*np.arange(-2.0, -0.8, dx),
        weights=df['freq'], label='data\n$\gamma_1=%g$'%round(cm3/var**1.5, 2),
        density=True)
xmin = mu-4*var**.5
xmax = mu+4*var**.5
x = np.linspace(xmin, xmax)
ax.plot(xfactor*x, scipy.stats.norm.pdf(x, mu, var**.5),
        label='lognormal\n$\mu=%g$,\n$\sigma=%g$'%(round(xfactor*mu, 2), round(xfactor*var**.5, 2)))
ax.set_xlim(xfactor*xmin, xfactor*xmax)
ax.legend(loc='upper right')
ax.set_xlabel('$log_{20}$ likelihood');
ax.set_ylabel('probability density');

ax = axes[1]
axinset =  inset_axes(ax, width='28%', height='42%', loc='upper left')
axinset.tick_params(labelleft=False, labelright=True, left=False, right=True)
k = 4
df = pd.read_csv('data/freq4.csv')
counts, bins = np.histogram(np.log10(df['freq']), bins=int(2*np.log(len(df['freq']))), weights=df['freq'])
x = 0.5*(bins[:-1]+bins[1:])
y = counts/(np.sum(counts)*np.diff(bins))
ax.plot(xfactor*x, y, label='data')
axinset.plot(xfactor*x, y)
xmin, xmax = round(k*mu-5*(k*var)**.5, 1), round(k*mu+4.5*(k*var)**.5, 1)
x = np.linspace(xmin, xmax)
y = scipy.stats.norm.pdf(x, k*mu, (k*var)**.5)
ax.plot(xfactor*x, y, label='lognormal')
axinset.plot(xfactor*x, y)
p4mers = np.load('data/loglikelihood-k4-all.npy')
counts, bins = np.histogram(p4mers, bins=bins, weights=10**p4mers)
x = 0.5*(bins[:-1]+bins[1:])
y = counts/(np.sum(counts)*np.diff(bins))
ax.plot(xfactor*x, y, label='independent')
axinset.plot(xfactor*x, y)
axinset.set_yscale('log')
for a in [ax, axinset]:
    a.set_xlim(xfactor*xmin, xfactor*xmax)
ax.set_xlabel('$log_{20}$ likelihood');
ax.set_ylabel('probability density');
ax.legend(loc='upper right')

ax = axes[2]
axinset =  inset_axes(ax, width='28%', height='42%', loc='upper left')
axinset.tick_params(labelleft=False, labelright=True, left=False, right=True)
k = 9
phuman = np.load('data/loglikelihood-k9-human.npy')
phuman = phuman[~np.isnan(phuman)]
counts, bins = np.histogram(phuman, bins=70)
x = 0.5*(bins[:-1]+bins[1:])
y = counts/(np.sum(counts)*np.diff(bins))
ax.plot(xfactor*x, y, label='data')
axinset.plot(xfactor*x, y)
xmin, xmax = round(k*mu-5*(k*var)**.5, 1), round(k*mu+4.5*(k*var)**.5, 1)
x = np.linspace(xmin, xmax)
y = scipy.stats.norm.pdf(x, k*mu, (k*var)**.5)
ax.plot(xfactor*x, y, label='lognormal')
axinset.plot(xfactor*x, y)
prandom = np.load('data/loglikelihood-k9-random.npy')
counts, bins = np.histogram(prandom, bins=bins)
x = 0.5*(bins[:-1]+bins[1:])
y = counts/(np.sum(counts)*np.diff(bins))
ax.plot(xfactor*x, y, label='independent')
axinset.plot(xfactor*x, y)
axinset.set_yscale('log')
for a in [ax, axinset]:
    a.set_xlim(xfactor*xmin, xfactor*xmax)
ax.set_xlabel('$log_{20}$ likelihood');
ax.set_ylabel('probability density');
ax.legend(loc='upper right')

plotting.label_axes(axes, labelstyle=r'%s', weight='bold')
fig.tight_layout()
fig.savefig('../../paper/images/lognormalaa.pdf')
plt.show()
