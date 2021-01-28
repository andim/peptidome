import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import colors

import sys
sys.path.append('..')
from lib import *
plt.style.use('../peptidome.mplstyle')

k = int(snakemake.wildcards.k)
proteome = snakemake.wildcards.proteome

fig, axes = plt.subplots(figsize=(4.5, 1.75), ncols=2)

energies = {}
models = ['test', 'independent', 'ncov', 'nskew', 'nskewfcov']
for model in models:
    print(model)
    energies[model] = np.load('data/{proteome}_{model}_k{k}_energies.npz'.format(proteome=proteome, model=model, k=k))['energies']

xmax = max([max(energies[model]) for model in models])+0.05
xmin = min([min(energies[model]) for model in models])-0.05
nbins = 100
scaley = nbins/(xmax-xmin)
bins = np.logspace(-xmax, -xmin, num=nbins+1, base=np.exp(1))
for ax in axes:
    kwargs = dict(lw=0.5)
    for model in models:
        values = np.exp(-energies[model])
        counts, bins = np.histogram(values, bins=bins)
        counts = counts/np.sum(counts)
        ax.step(bins[:-1], counts*scaley, label=model, where='mid', **kwargs)
    ax.set_xscale('log')
    ax.set_xlabel(r'$P(\sigma)$')
    ax.set_xlim(min(bins), max(bins))
axes[0].set_ylabel('Probability Density')
axes[0].set_ylim(0.0)
axes[0].legend(loc='upper right')
axes[1].set_yscale('log')
fig.tight_layout()
fig.savefig(snakemake.output[0])
