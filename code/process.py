##
#
# Process UFO csv files to fasta for more uniform processing
#
##

import pandas as pd
from lib import *
import glob

files = glob.glob(datadir + 'ufos/*.csv')

for f in files:
    df = pd.read_csv(f, sep='\t')
    write_fasta(df, f.split('.')[0]+'.fasta', seqcolumn='AA_seq', idcolumn='Acc', descriptioncolumn='Description')
