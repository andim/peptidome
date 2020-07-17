import itertools

import sys
sys.path.append('..')
from lib import *
from lib import clib
from lib.maxent import *

import numpy as np

def save(name, h, Jk):
    aas_arr = np.array(list(aminoacids))
    dfh = pd.DataFrame(index=aas_arr, data=h, columns=['h'])
    #dfJk = pd.DataFrame(data=Jk, columns=range(len(Jk)))
    doublets = [''.join(s) for s in itertools.product(list(aminoacids), repeat=2)]
    dfJk = pd.DataFrame(index=doublets,
                        data=[Jk[0,aatonumber(s[0]),aatonumber(s[1])] for s in doublets],
                        columns=['J0'])
    for i in range(1, len(Jk)):
        dfJk['J%g'%i] = [Jk[i,aatonumber(s[0]),aatonumber(s[1])] for s in doublets]

    dfh.to_csv('data/%s_h.csv' % name)
    dfJk.to_csv('data/%s_Jk.csv' % name)
