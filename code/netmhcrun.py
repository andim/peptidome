import subprocess
import os
from multiprocessing import Pool
import pandas as pd

from lib import *
from netmhcrunutils import *


#n_proc = 20 #number of parallel python processes

dfproteomes = load_proteomes()
dfhla = pd.read_csv(datadir+'hlas.csv', sep='\t', skiprows=1)

        
workpool = Pool()#(processes=n_proc)

for idx, row in dfproteomes.iterrows():
    #if idx < 2:
    if row['type'] not in ['fungus','virus','bacterium','parasite']:
        # skip species if not a human pathogen?
        continue
    fastapath = datadir + row['path']
    name = idx #row['shortname']

    outname = datadir + 'netmhc/%s' % name.replace(' ', '')
    
    workpool.starmap( run_netMHC, [(fastapath,outname,hla) for hla in dfhla['name']])

