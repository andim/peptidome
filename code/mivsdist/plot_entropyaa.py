import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import brokenaxes

fig = plt.figure(figsize=(3.5, 2.5))
ax = brokenaxes.brokenaxes(ylims=((0.0, 0.2), (3.7, 4.22)), bottom=0.2, left=0.2)
df = pd.read_csv('data/entropyaa-human.csv')
#ax = brokenaxes.brokenaxes(ylims=((2.2, 2.4), (3.7, 4.22)), bottom=0.2, left=0.2)
#df = pd.read_csv('data/entropyaa-viruses.csv')
ax.plot(df['position'], df['entropy'], 'o')
print(df['position'])
ax.set_xlim(0, 51)
ax.set_ylabel('entropy in bits', labelpad=6)
ax.set_xlabel('position', labelpad=5)
fig.savefig('../../paper/images/entropyaa.pdf')
plt.show()
