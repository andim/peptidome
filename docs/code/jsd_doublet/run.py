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

from lib import *
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

proteomes = load_proteomes()
human = datadir + proteomes.loc['Human']['path']
names = ['Human', 'Mouse', 'Vaccinia', 'InfluenzaB', 'InfluenzaA', 'CMV', 'HCV', 'HSV1',
       'DENV', 'HIV', 'EBV', 'Ebola', 'Ecoli', 'Tuberculosis', 'Listeria',
       'Burkholderia', 'Meningococcus', 'StrepA', 'Hpylori',
       'Lyme', 'Tetanus', 'Leprosy', 'Malaria', 'Chagas',
       'OnchocercaVolvulus']

dfs = {name: counter_to_df(count_kmers_proteome(datadir + proteomes.loc[name]['path'], 2), norm=False, clean=True) for name in proteomes.index}

N = len(names)
distances_uniform = np.zeros(N)
distances = np.zeros((N, N))
for i, namei in enumerate(names):
    df1 = dfs[namei]
    f1 = np.asarray(list(df1['count']))
    f1 += np.ones_like(f1)
    f2 = np.ones_like(f1)
    distances_uniform[i] = calc_jsd(f1, f2)
    for j, namej in enumerate(names):
#        if i == j:
#            f1 = np.asarray(df1['count'])
#            f1 += np.ones_like(f1)
#            f2 = np.ones_like(f1)
        df2 = dfs[namej]
        dfm = pd.merge(df1, df2, on='seq', suffixes=['_1', '_2'])
        f1, f2 = np.asarray(dfm['count_1']), np.asarray(dfm['count_2'])
        f1 += np.ones_like(f1)
        f2 += np.ones_like(f2)
        distances[i, j] = calc_jsd(f1, f2, base=2)

fullnames = list(proteomes.loc[names]['fullname'])

df = pd.DataFrame(distances, index=names, columns=names, copy=True)
df['Uniform'] = distances_uniform
df = df.append(pd.Series(distances_uniform, name='Uniform', index=names))
df.iloc[-1, -1] = 0.0
df.to_csv('data/jsds.csv', index=True)
