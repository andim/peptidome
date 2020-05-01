##
#
# Process UFO csv files to fasta for more uniform processing
#
##

import pandas as pd
from lib import *
import glob

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import Bio.Alphabet

files = glob.glob(datadir + 'ufos/*.csv')

for f in files:
    df = pd.read_csv(f, sep='\t')
    records = []
    for i, row in df.iterrows():
        record = SeqRecord(seq=Seq(row['AA_seq']), id=row['Acc'], description=row['Description'])
        records.append(record)
    SeqIO.write(records, f.split('.')[0]+'.fasta', format='fasta')
