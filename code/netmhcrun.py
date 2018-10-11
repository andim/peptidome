import subprocess
import os

import pandas as pd

from lib import *

dfproteomes = pd.read_csv('../data/proteomes.csv', sep=',')
dfhla = pd.read_csv('../data/hlas.csv', sep='\t', skiprows=1)

for idx, row in dfproteomes.iterrows():
    fastapath = datadir + row['path']
    name = row['shortname']

    outname = datadir + 'netmhc/%s' % name

    for hla in dfhla['name']:
        fullout = '%s-%s.csv' % (outname, hla)
        if not os.path.exists(fullout):
            #netMHC fasta -xls -xlsfile out.csv -a HLA-A0101
            subprocess.run(['netMHC', fastapath, '-xls', '-xlsfile %s' %fullout, '-a %s'%hla])
            df = pd.read_csv(fullout, sep='\t', skiprows=1)
            dfbinders = df[df['nM']<500]
            dfbinders.to_csv(fullout)
