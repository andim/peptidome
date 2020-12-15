import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import colors

from common import labels

import sys
sys.path.append('..')
from lib import *
plt.style.use('../peptidome.mplstyle')

k = int(snakemake.wildcards.k)
proteome = snakemake.wildcards.proteome

models = ['uniform', 'independent', 'ncov', 'nskew', 'nskewfcov']
entropies = np.zeros(len(models))
for i, model in enumerate(models):
    if model == 'uniform':
        S = k*np.log(20)
    else:
        S = np.float(pd.read_csv('data/{proteome}_{model}_k{k}_entropy.csv'.format(proteome=proteome, model=model, k=k),
                        header=None, index_col=0).loc['S'])
    entropies[i] = S
data = np.exp(entropies)

fig, ax = plt.subplots(figsize=(1.5, 2.0))
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
fig.tight_layout()

fig.savefig(snakemake.output[0])
