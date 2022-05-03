import sys
sys.path.append('..')
from lib import *
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

plt.style.use('../peptidome.mplstyle')

fig, ax = plt.subplots(figsize=(2.25, 1.75))
x = np.linspace(0, 1)


def hole(distance):
    if distance < 0.35:
        return 0.0
    else:
        return 1.0

def outlier(distance):
    return 1/(1+np.exp(-(distance-0.25)/0.15))**3

ax.plot(x, np.vectorize(hole)(x), label='Strong Whitelisting')
ax.plot(x, np.vectorize(outlier)(x), label='Outlier Detection')
ax.legend(title='Theory Expectation', loc='lower right')
ax.set_xlabel('Distance to Self')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_ylabel('Immunogenicity')
fig.tight_layout()
plt.show()
fig.savefig('main.png')
fig.savefig(figuredir+'theory_sketches.svg')
