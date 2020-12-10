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

fig, axes = plt.subplots(figsize=(4.5, 1.75), ncols=2)

energies = {}
models = ['test', 'independent', 'ncov', 'nskew', 'nskewfcov']
for model in models:
    print(model)
    energies[model] = np.load('data/Human_{model}_k{k}_energies.npz'.format(model=model, k=k))['energies']

xmax = max([max(energies[model]) for model in models])+0.1
xmin = min([min(energies[model]) for model in models])-0.1
nbins = 100
for ax in axes:
    plot_histograms([energies[model] for model in models],
                    models,
                    step=True, nbins=nbins, xmin=xmin, xmax=xmax, lw=0.5, ax=ax, scaley=nbins/(xmax-xmin))
    ax.set_xlabel('Energy')
    ax.set_ylabel('Density')
axes[0].set_ylim(0.0)
axes[0].legend(loc='upper left')
axes[1].get_legend().remove()
axes[1].set_yscale('log')
fig.tight_layout()
fig.savefig(snakemake.output[0])
