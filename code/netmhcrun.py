import subprocess
import os

import pandas as pd

from lib import *

FNULL = open(os.devnull, 'w') # Mute the subprocess (prevents shell cluttering)

dfproteomes = pd.read_csv(datadir+'proteomes.csv', sep=',')
dfhla = pd.read_csv(datadir+'hlas.csv', sep='\t', skiprows=1)

for idx, row in dfproteomes.iterrows():
    if idx < 2:
        continue
    fastapath = datadir + row['proteomeid'] + row['shortname']
    name = row['shortname']

    outname = datadir + 'netmhc/%s' % name.replace(' ', '')

    for hla in dfhla['name']:
        fullout = '%s-%s.csv' % (outname, hla)
        if not os.path.exists(fullout):
            #netMHC fasta -xls -xlsfile out.csv -a HLA-A0101
            subprocess.run(['netMHC', '-f', fastapath+".fasta",
                            '-xls',
                            '-xlsfile %s' % fullout,
                            '-a %s'%hla],
                            stdout=FNULL, stderr=subprocess.STDOUT); # Mute the 
                            #subprocess (prevents shell cluttering) 

            df = pd.read_csv(fullout, sep='\t', skiprows=1)
            dfbinders = df[df['nM']<500]
            dfbinders.to_csv(fullout)
