import subprocess
import os
import pandas as pd
FNULL = open(os.devnull, 'w') # Mute the subprocess (prevents shell cluttering)

def run_netMHC(fastapath,outname,hla):
    '''Make a wrapper function to parallelize netMHC call.'''
    fullout = '%s-%s.csv' % (outname, hla)
    if not os.path.exists(fullout):
        #netMHC fasta -xls -xlsfile out.csv -a HLA-A0101
        subprocess.run(['netMHC', '-f', fastapath,
                        '-xls',
                        '-xlsfile %s' % fullout,
                        '-a %s'%hla],
                        stdout=FNULL, stderr=subprocess.STDOUT); # Mute the
                        #subprocess (prevents shell cluttering)

        df = pd.read_csv(fullout, sep='\t', skiprows=1)
        dfbinders = df[df['nM']<500]
        dfbinders.to_csv(fullout)
