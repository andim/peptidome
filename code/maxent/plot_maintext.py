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

fig, axes = plt.subplots(figsize=(5.5, 1.75), ncols=3, nrows=3)

## observables
observables = ['fi', 'cij', 'cijk']
observables_dict = {key: dict() for key in observables}
datasets = ['test', 'model']
for i, dataset in enumerate(datasets):
    params = np.load(snakemake.input[i])
    for observable in observables:
        observables_dict[observable][dataset] = params[observable]

for j, (observable, label, lims, flattener) in enumerate([('fi', '$f_i$', (0, 0.12), np.ravel),
                                               ('cij', '$C_{ij}$', (-0.0025, 0.0035), flatten_ij),
                                               ('cijk', '$C_{ijk}$', (-4e-4, 7e-4), flatten_ijk)]):
    ax = axes[0, j]
    if observable in ['cij', 'cijk']:
        plotting.density_scatter(flattener(observables_dict[observable][dataset]),
                                 flattener(observables_dict[observable]['test']),
                                 norm=colors.LogNorm(vmin=1),
                                 s=0.5,
                                 bins=50, ax=ax)
    else:
        ax.plot(flattener(observables_dict[observable][dataset]),
                flattener(observables_dict[observable]['test']),
                'o', ms=2 if observable == 'fi' else 1)

    ax.set_xlabel('model %s'%(label))
    ax.set_ylabel('test %s'%label)
    ax.plot(lims, lims, 'k')
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)

for ax in axes[0, 1:]:
    ax.ticklabel_format(style='sci', scilimits=(0,0))

## dos
labels = {
          'test': 'test set',
          'independent': '1st moment',
          'ncov' : '2nd moment',
          'nskew' : '3rd moment',
          'nskewfcov' : '3rd moment, 2-point'
          }
energies = {}
models = ['test', 'independent', 'ncov', 'nskew', 'nskewfcov']
for model in models:
    energies[model] = np.load('data/Human_{model}_k{k}_energies.npz'.format(model=model, k=k))['energies']

xmax = max([max(energies[model]) for model in models])+0.1
xmin = min([min(energies[model]) for model in models])-0.1
nbins = 100
for ax in axes:
    plot_histograms([energies[model] for model in models],
                    [labels[model] for model in models],
                    step=True, nbins=nbins, xmin=xmin, xmax=xmax, lw=0.5, ax=ax, scaley=nbins/(xmax-xmin))
    ax.set_xlabel('Energy')
    ax.set_ylabel('Density')
axes[1, 0].set_ylim(0.0)
axes[1, 0].legend(loc='upper left')
axes[1, 1].get_legend().remove()
axes[1, 1].set_yscale('log')


label_axes(fig, labelstyle='%s', xy=(-0.25, 1.0))
fig.tight_layout()

fig.savefig(snakemake.output[0])
