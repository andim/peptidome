import os.path
import re
import urllib.request
import pandas as pd
from Bio import SeqIO

from lib import *

proteomes = load_proteomes()
# download proteomes from uniprot
for ind, row in proteomes.iterrows():
    if not os.path.exists(datadir+row['path']):
        if row['speciesid']:
            url = r"ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/%s_%s.fasta.gz"%(row['proteomeid'], row['speciesid'])
        else:
            url = r"http://www.uniprot.org/uniprot/?query=proteome%3A"+row['proteomeid']+r"&format=fasta&include=no"
        urllib.request.urlretrieve(url, datadir+row['path'])

# Download proteome of all viruses with human host
path = datadir+'human-viruses-uniref90.fasta'
if not os.path.exists(path):
    url = r"https://www.uniprot.org/uniref/?query=uniprot%3A(host%3A%22Homo+sapiens+(Human)+[9606]%22)+AND+identity%3A0.9&sort=score&format=fasta"
    print(url)
    urllib.request.urlretrieve(url, path)

# filter HIV1 proteins from the combined proteome
path_nohiv = datadir + 'human-viruses-uniref90_nohiv.fasta'
if not os.path.exists(path_nohiv):
    def load(fasta_name):
        with open(fasta_name) as fastain:
            for seq in SeqIO.parse(fastain, 'fasta'):
                if not ('Human immunodeficiency virus 1' in seq.description):
                    yield seq
    with open(path_nohiv, "w") as fastaout:
        SeqIO.write(list(load(path)), fastaout, "fasta")

# download immune epitope data from IEDB
path_urls = [('iedb-tcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip'),
             ('iedb-bcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3.zip') 
            ]
for path, url in path_urls:
    if not os.path.exists(datadir+path):
        print(path)
        urllib.request.urlretrieve(url, datadir+path)
