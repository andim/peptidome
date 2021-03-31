import subprocess
import os
import pandas as pd

FNULL = open(os.devnull, 'w') # Mute the subprocess (prevents shell cluttering)

repopath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
netmhcpath = os.path.join(repopath, 'dependencies/netMHC-4.0/netMHC')

def run_netMHC(fastapath, outname, hla, binder_only=True):
    """run netMHC on a fasta file

    binder_only: filter to only keep high binders < 500nM

    """
    fastapath = os.path.abspath(fastapath)
    fullout = os.path.abspath('%s-%s.csv' % (outname, hla))
    if not os.path.exists(fullout):
        #netMHC -f fasta -xls -xlsfile out.csv -a HLA-A0101
        print("netMHC -f {fastapath} -xls -xlsfile {fullout} -a {hla}".format(fastapath=fastapath, fullout=fullout, hla=hla))
        subprocess.run([netmhcpath, '-f', fastapath,
                        '-xls',
                        '-xlsfile %s' % fullout,
                        '-a %s'%hla],
                        # Mute the subprocess (prevents shell cluttering)
                        stdout=FNULL, stderr=subprocess.STDOUT)
        if binder_only:
            df = pd.read_csv(fullout, sep='\t', skiprows=1)
            dfbinders = df[df['nM']<500]
            dfbinders.to_csv(fullout)
