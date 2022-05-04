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


def shell(distance):
    return 1.0/(1+np.exp(-(distance-0.05)/0.05))**5 * 1/(1+np.exp((distance-0.2)/0.5))

ax.plot(x, shell(x)/max(shell(x)), label='Shell Model')
ax.legend(title='Theory Expectation', loc='lower right')
ax.set_xlabel('Distance to Self')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_ylabel('Immunogenicity')
fig.tight_layout()
plt.show()
fig.savefig('shell.png')
fig.savefig(figuredir+'shell.svg')
