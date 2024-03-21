import subprocess
import os
import pandas as pd

FNULL = open(os.devnull, 'w') # Mute the subprocess (prevents shell cluttering)

repopath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
netmhcpath = os.path.join(repopath, 'dependencies/netMHC-4.0/netMHC')

def run_netMHC(inputpath, outname, hla, binder_only=True):
    """run netMHC on a fasta file

    binder_only: filter to only keep strong binders < %rank<0.5

    """
    inputpath = os.path.abspath(inputpath)
    peptides = inputpath.endswith('csv') or inputpath.endswith('txt')
    fullout = os.path.abspath('%s-%s.csv' % (outname, hla))
    if not os.path.exists(fullout):
        #netMHC -f fasta -xls -xlsfile out.csv -a HLA-A0101
        print(f"{netmhcpath} -f {inputpath} -xls -xlsfile {fullout} -a {hla}")
        if peptides:
            subprocess.run([netmhcpath, '-f', inputpath,
                            '-xls',
                            '-xlsfile %s' % fullout,
                            '-a %s'%hla,
                            '-p'],
                            # Mute the subprocess (prevents shell cluttering)
                            stdout=FNULL, stderr=subprocess.STDOUT)
        else:
            subprocess.run([netmhcpath, '-f', inputpath,
                            '-xls',
                            '-xlsfile %s' % fullout,
                            '-a %s'%hla],
                            # Mute the subprocess (prevents shell cluttering)
                            stdout=FNULL, stderr=subprocess.STDOUT)
        if binder_only:
            df = pd.read_csv(fullout, sep='\t', skiprows=1)
            dfbinders = df[df['Rank']<0.5]
            dfbinders.to_csv(fullout)
