import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import brokenaxes

fig = plt.figure(figsize=(3.5, 2.5))
ax = brokenaxes.brokenaxes(ylims=((0.65, 0.75), (3.85, 4.22)), bottom=0.2, left=0.2)
df = pd.read_csv('data/entropyaa-human.csv')
ax.plot(df['position'], df['entropy'], 'o')
print(df['position'])
ax.set_xlim(0, 51)
ax.set_ylabel('entropy in bits', labelpad=6)
ax.set_xlabel('position', labelpad=5)
fig.savefig('../../paper/images/entropyaa.pdf')
plt.show()
