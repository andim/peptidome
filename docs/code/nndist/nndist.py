import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.stats
from scipy.stats import poisson
import seaborn as sns
import sklearn.neighbors
import matplotlib.pyplot as plt

import Levenshtein

import sys
sys.path.append('..')
from lib import *

plt.style.use('../peptidome.mplstyle')

counter9 = count_kmers_proteome(human, k, clean=True)
human9 = set(counter9)
Nhuman = len(human9)

selfset = human9

def distance_distribution(sample, selfset):
    d0 = [x in selfset for x in sample]
    count0 = np.sum(d0)
    d1 = np.array([dist1(x, selfset) for x in sample])
    count1 = np.sum(d1)
    d2 = np.array([dist2(x, selfset) for x in sample]) & (~d1)
    count2 = np.sum(d2)
    ns = np.array([count0, count1, count2, len(sample)-count0-count1-count2])
    return ns

 = count_kmers_proteome(datadir + 'cancer/pb1ufo.fasta.gz')
