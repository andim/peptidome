import numpy as np
import pandas as pd
from matplotlib import colors

import sys
sys.path.append('..')
from lib import *
plt.style.use('../peptidome.mplstyle')


observables = ['fi', 'cij', 'cijk']
observables_dict = {key: dict() for key in observables}
datasets = ['train', 'test', 'model', 'model_ncov', 'model_nskew', 'model_nskewdiag']
for dataset in datasets:
    params = np.load('data/%s_observables.npz'%dataset)
    for observable in observables:
        observables_dict[observable][dataset] = params[observable]

for model_type in datasets[2:]:
    fig, axes = plt.subplots(figsize=(6, 3.5), ncols=3, nrows=2)

    for j, (observable, label, lims, flattener) in enumerate([('fi', '$f_i$', (0, 0.12), np.ravel),
                                                   ('cij', '$C_{ij}$', (-0.0025, 0.0035), flatten_ij),
                                                   ('cijk', '$C_{ijk}$', (-4e-4, 7e-4), flatten_ijk)]):
        for i, dataset in enumerate([model_type, 'test']):
            ax = axes[i, j]
            if observable in ['cij', 'cijk']:
                plotting.density_scatter(flattener(observables_dict[observable]['train']),
                                         flattener(observables_dict[observable][dataset]),
                                         norm=colors.LogNorm(vmin=1),
                                         s=0.5,
                                         bins=50, ax=ax)
            else:
                ax.plot(flattener(observables_dict[observable]['train']),
                        flattener(observables_dict[observable][dataset]),
                        'o', ms=2 if observable == 'fi' else 1)

            ax.set_xlabel('train %s'%label)
            ax.set_ylabel('%s %s'%(dataset, label))
            ax.plot(lims, lims, 'k')
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)

    for ax in axes[:, 1:].flatten():
        ax.ticklabel_format(style='sci', scilimits=(0,0))

    label_axes(fig, labelstyle='%s')
    fig.tight_layout()

    fig.savefig('main.png' if model_type == 'model' else model_type+'.png')
    if model_type == 'model':
        fig.savefig('../../paper/images/maxent_freqs.pdf')
    elif model_type == 'model_global':
        fig.savefig('../../paper/images/maxent_freqs_global.pdf')
    plt.show()
