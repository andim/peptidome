import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import colors

import sys
sys.path.append('..')
from lib import *
from pathlib import Path
plt.style.use(Path('../peptidome.mplstyle'))

k = int(snakemake.wildcards.k)

observables = ['fi', 'cij', 'cijk']
observables_dict = {key: dict() for key in observables}
datasets = ['test', 'model']
for i, dataset in enumerate(datasets):
    params = np.load(snakemake.input[i])
    for observable in observables:
        observables_dict[observable][dataset] = params[observable]

fig, axes = plt.subplots(figsize=(5.5, 1.75), ncols=3, nrows=1)

for j, (observable, label, lims, flattener) in enumerate([('fi', '$f_i$', (0, 0.12), np.ravel),
                                               ('cij', '$C_{ij}$', (-0.0025, 0.0035), flatten_ij),
                                               ('cijk', '$C_{ijk}$', (-4e-4, 7e-4), flatten_ijk)]):
    ax = axes[j]
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

    ax.set_xlabel('%s %s'%(dataset, label))
    ax.set_ylabel('test %s'%label)
    ax.plot(lims, lims, 'k')
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)

for ax in axes[1:]:
    ax.ticklabel_format(style='sci', scilimits=(0,0))

label_axes(fig, labelstyle='%s', xy=(-0.25, 1.0))
fig.tight_layout()

fig.savefig(snakemake.output[0])
