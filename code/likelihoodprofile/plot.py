import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from lib import *

df = pd.read_csv('data/proteome-refhuman-k9-Human.zip')
print(df)

phuman = df['likelihoods']

fig, ax = plt.subplots()
ps = [phuman]
labels = ['human']
plot_histograms(ps, labels, xmin=-14.1, xmax=-9, ax=ax)
ax.set_ylabel('frequency')
ax.set_xlabel('probability given human proteome statistics')
fig.tight_layout()
plt.show()
