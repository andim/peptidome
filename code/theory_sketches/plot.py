import sys
sys.path.append('..')
from lib import *
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

plt.style.use('../peptidome.mplstyle')

fig, axes = plt.subplots(figsize=(2*2.1, 1.6), ncols=2)
x = np.linspace(0, 1)


def hole(distance):
    if distance < 0.45:
        return 0.0
    else:
        return 1.0

def outlier(distance):
    return 1/(1+np.exp(-(distance-0.25)/0.15))**3

axes[0].plot(x, np.vectorize(hole)(x), label='Strong Whitelisting')
axes[1].plot(x, np.vectorize(outlier)(x), label='Outlier Detection')
axes[0].set_title('Whitelisting')
axes[1].set_title('Probabilistic classification')
for ax in axes:
    ax.set_xlabel('Distance to Self')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['low', 'high'])
    ax.xaxis.labelpad = -5
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['low', 'high'])
    ax.set_ylabel('Immunogenicity')
    ax.yaxis.labelpad = -10
fig.tight_layout()
plt.show()
fig.savefig('main.png')
fig.savefig(figuredir+'theory_sketches.svg')
