import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import brokenaxes
plt.style.use('../peptidome.mplstyle')

name = sys.argv[1]

fig = plt.figure(figsize=(3.5, 2.5))
if name == 'Human':
    ax = brokenaxes.brokenaxes(ylims=((0.0, 0.2), (3.7, 4.22)), bottom=0.2, left=0.2)
else:
    ax = fig.add_subplot(111)
df = pd.read_csv('data/entropyaa-%s.csv'%name)
#ax = brokenaxes.brokenaxes(ylims=((2.2, 2.4), (3.7, 4.22)), bottom=0.2, left=0.2)
#df = pd.read_csv('data/entropyaa-viruses.csv')
ax.plot(df['position'], df['entropy'], 'o', label=name)
print(df['position'])
if name != 'Human':
    df = pd.read_csv('data/entropyaa-%s.csv'%'human')
    ax.plot(df['position'], df['entropy'], 'o', label='Human', alpha=.5)
    ax.legend()
ax.set_xlim(0, 51)
ax.set_ylabel('entropy in bits', labelpad=6)
ax.set_xlabel('position', labelpad=5)
fig.tight_layout()
if name == 'Human':
    fig.savefig('../../paper/images/entropyaa.pdf')
    fig.savefig('main.png')
else:
    fig.savefig('entropyvsposition_%s.png'%name)
plt.show()
