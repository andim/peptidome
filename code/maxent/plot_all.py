import numpy as np
import pandas as pd
from matplotlib import colors

import sys
sys.path.append('..')
from lib import *
plt.style.use('../peptidome.mplstyle')

from common import labels


observables = ['fi', 'cij', 'cijk']
observables_dict = {key: dict() for key in observables}
datasets = ['train', 'test', 'model', 'model_ncov', 'model_nskewdiag', 'model_nskew', 'model_nskewfcov']
for dataset in datasets:
    params = np.load('data/%s_observables.npz'%dataset)
    for observable in observables:
        observables_dict[observable][dataset] = params[observable]

nrows = len(datasets[1:])
fig, axes = plt.subplots(figsize=(6.0, 1.75*nrows), ncols=3, nrows=nrows)
for j, (observable, label, lims, flattener) in enumerate([('fi', '$f_i$', (0, 0.12), np.ravel),
                                               ('cij', '$C_{ij}$', (-0.0025, 0.0035), flatten_ij),
                                               ('cijk', '$C_{ijk}$', (-4e-4, 7e-4), flatten_ijk)]):
    for i, model_type in enumerate(datasets[1:]):
        ax = axes[i, j]
        if observable in ['cij', 'cijk']:
            plotting.density_scatter(flattener(observables_dict[observable]['train']),
                                     flattener(observables_dict[observable][model_type]),
                                     norm=colors.LogNorm(vmin=1),
                                     s=0.5,
                                     bins=50, ax=ax)
        else:
            ax.plot(flattener(observables_dict[observable]['train']),
                    flattener(observables_dict[observable][model_type]),
                    'o', ms=2 if observable == 'fi' else 1)

        ax.set_xlabel('train %s'%label)
        ax.set_ylabel('%s %s'%(labels[model_type], label))
        ax.plot(lims, lims, 'k')
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)

for ax in axes[:, 1:].flatten():
    ax.ticklabel_format(style='sci', scilimits=(0,0))

label_axes(fig, labelstyle='%s')
fig.tight_layout()
fig.savefig('connected_correlations_allmodels.png')
