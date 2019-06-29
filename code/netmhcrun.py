# runs netMHC on all pathogen proteomes
# for all HLA types

import subprocess
import os
from multiprocessing import Pool
import pandas as pd

from lib import *
from netmhcrunutils import *

#number of parallel python processes
n_proc = 1

dfproteomes = load_proteomes(only_pathogens=True)
dfhla = pd.read_csv(datadir+'hlas.csv', sep='\t', skiprows=1)
pool = Pool(processes=n_proc)

for name, row in dfproteomes.iterrows():
    print(name)
    fastapath = datadir + row['path']
    outname = datadir + 'netmhc/%s' % name.replace(' ', '')
    pool.starmap(run_netMHC, [(fastapath, outname, hla) for hla in dfhla['name']])
