import numpy as np
import pandas as pd
from matplotlib import colors

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

observables = ['fi', 'cij', 'cijk']
observables_dict = {key: dict() for key in observables}
for dataset in ['train', 'test', 'model', 'model_global']:
    params = np.load('data/%s_observables.npz'%dataset)
    for observable in observables:
        observables_dict[observable][dataset] = params[observable]

for model_type in ['model', 'model_global']:
    fig, axes = plt.subplots(figsize=(8, 5), ncols=3, nrows=2)

    for j, (observable, label, lims, flattener) in enumerate([('fi', '$f_i$', (0, 0.12), np.ravel),
                                                   ('cij', '$C_{ij}$', (-0.0035, 0.0035), flatten_ij),
                                                   ('cijk', '$C_{ijk}$', (-7e-4, 7e-4), flatten_ijk)]):
        for i, dataset in enumerate([model_type, 'train']):
            ax = axes[i, j]
            if observable in ['cij', 'cijk']:
                plotting.density_scatter(flattener(observables_dict[observable]['test']),
                                         flattener(observables_dict[observable][dataset]),
                                         norm=colors.LogNorm(vmin=1),
                                         s=0.5,
                                         bins=50, ax=ax)
            else:
                ax.plot(flattener(observables_dict[observable]['test']),
                        flattener(observables_dict[observable][dataset]),
                        'o', ms=2 if observable == 'fi' else 1)

            ax.set_xlabel('test %s'%label)
            ax.set_ylabel('%s %s'%(dataset, label))
            ax.plot(lims, lims, 'k')
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)

    for ax in axes[:, 1:].flatten():
        ax.ticklabel_format(style='sci', scilimits=(0,0))

    fig.tight_layout()

    fig.savefig('main.png' if model_type == 'model' else 'model_global.png')
    plt.show()
