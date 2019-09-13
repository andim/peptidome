import sys, copy
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import sklearn.manifold
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
plt.style.use('../peptidome.mplstyle')

from lib import *

df = pd.read_csv('data/jsds.csv', index_col=0)
print(df)
names = ['Human', 'Mouse', 'Vaccinia', 'InfluenzaB', 'InfluenzaA', 'CMV', 'HCV', 'HSV1',
       'DENV', 'HIV', 'EBV', 'Ebola', 'Ecoli', 'Tuberculosis', 'Listeria',
       'Burkholderia', 'Meningococcus', 'StrepA', 'Hpylori',
       'Lyme', 'Tetanus', 'Leprosy', 'Malaria', 'Chagas',
       'OnchocercaVolvulus']


colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
type_to_color = {'virus' : colors[0],
                 'bacterium' : colors[1],
                 'parasite' : colors[2],
                 'vertebrate' : colors[3]
                }
typecolors = np.array([type_to_color[proteomes.loc[name]['type']] for  name in names])


ys = df['Uniform'][names]
xs = df['Human'][names]
fig, ax = plt.subplots(figsize=(3.42, 3.42))
ax.scatter(xs, ys, color=typecolors)
#offset=0.00
#for i, name in enumerate(names):
#    ax.text(xs[i], ys[i]+offset, name, ha='center', color=typecolors[i])
ax.plot([0, 0.15], [0, 0.15], 'k-')
ax.set_xlabel('JSD(proteome, human)')
ax.set_ylabel('JSD(proteome, uniform)')
ax.set_aspect('equal')
ax.set_xlim(-0.0, 0.15)
ax.set_ylim(-0.0, 0.15)
for i, (type_, color) in enumerate(type_to_color.items()):
    ax.text(0.8, 0.2-i*0.04, type_, color=color, transform=ax.transAxes)
#sns.despine(fig)
fig.tight_layout()
fig.savefig('main.png')
plt.show()
