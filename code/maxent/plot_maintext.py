import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import colors

import sys
sys.path.append('..')
from lib import *
plt.style.use('../peptidome.mplstyle')
from common import labels

k = 9
fig, axes = plt.subplots(figsize=(7.2, 5.0), ncols=3, nrows=2)

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
energies = {}
models = ['test', 'independent', 'ncov', 'nskew', 'nskewfcov']
for model in models:
    energies[model] = np.load('data/Human_{model}_k{k}_energies.npz'.format(model=model, k=k))['energies']

xmax = max([max(energies[model]) for model in models])+0.1
xmin = min([min(energies[model]) for model in models])-0.1
nbins = 100
for ax in axes[1,:2]:
    plot_histograms([energies[model] for model in models],
                    [labels[model] for model in models],
                    step=True, nbins=nbins, xmin=xmin, xmax=xmax, lw=0.5, ax=ax, scaley=nbins/(xmax-xmin))
    ax.set_xlabel('Energy')
    ax.set_ylabel('Density')
axes[1, 0].set_ylim(0.0)
axes[1, 0].legend(loc='upper left')
axes[1, 1].get_legend().remove()
axes[1, 1].set_yscale('log')


ax = axes[1, 2]
models = ['uniform', 'independent', 'ncov', 'nskew', 'nskewfcov']
entropies = np.zeros(len(models))
for i, model in enumerate(models):
    if model == 'uniform':
        S = k*np.log(20)
    else:
        S = np.float(pd.read_csv('data/Human_{model}_k{k}_entropy.csv'.format(model=model, k=k),
                        header=None, index_col=0).loc['S'])
    entropies[i] = S
data = np.exp(entropies)

labels = [labels[model] for model in models[1:]]
width = 0.5  # the width of the bars
x = np.arange(len(labels))  # the label locations
rects = ax.bar(x, data[1:]/data[0], width)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
#ax.axhline(data[1]/data[0], ls='--')

reduction = np.abs((data[2:]-data[1])/data[1])
for rect, toplabel in zip(rects[1:], reduction):
    ax.annotate('-{0:.0f}%'.format(100*toplabel),
                xy=(rect.get_x()+width/2, rect.get_height()),
                xytext=(0, 1),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize='xx-small')

ax.set_ylabel('Effective Diversity\n$\exp(S)/20^k$')

label_axes(fig, labelstyle='%s', xy=(-0.25, 1.0))
fig.tight_layout()

fig.savefig(snakemake.output[0])
