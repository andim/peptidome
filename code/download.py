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

# download protein domain data from Pfam
for ind, row in proteomes.iterrows():
    if row['speciesid']:
        path = datadir + row['speciesid'] +  '_pfam.tsv.gz'
        if not os.path.exists(path):
            print(path)
            url = r"ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/proteomes/%s.tsv.gz"%(row['speciesid'])
            try:
                urllib.request.urlretrieve(url, path)
            except urllib.error.URLError:
                print('%s not found in Pfam'%row['speciesid'])


# Download proteome of all viruses with human host
path = datadir+'human-viruses-uniref90.fasta'
if not os.path.exists(path):
    url = r"https://www.uniprot.org/uniref/?query=uniprot%3A(host%3A%22Homo+sapiens+(Human)+[9606]%22)+AND+identity%3A0.9&sort=score&format=fasta"
    print(url)
    try:
        urllib.request.urlretrieve(url, path)
    except:
        print('could not download human viruses')

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


path_urls = [
             # immune epitope data from IEDB
             ('iedb-tcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip'),
             ('iedb-bcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3.zip'),
             # Human protein atlas (tissue restriction, subcellular location)
             ('proteinatlas.tsv.zip', r'https://www.proteinatlas.org/download/proteinatlas.tsv.zip'),
             ('proteinatlas.xml.gz', r'https://www.proteinatlas.org/download/proteinatlas.xml.gz'),
             # DNA sequence data
             ('dna_chr21.fa.gz', r'ftp://ftp.ensembl.org/pub/release-96/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz'),
             ('dna_human_all.fa.gz', r'ftp://ftp.ensembl.org/pub/release-96/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz'),
             ('dna_pfalciparum.fasta', r'https://plasmodb.org/common/downloads/release-45/Pfalciparum3D7/fasta/data/PlasmoDB-45_Pfalciparum3D7_Genome.fasta'),
             ('disprot.tsv', r'https://www.disprot.org/api/search?release=current&show_ambiguous=false&show_obsolete=false&format=tsv')
            ]
for path, url in path_urls:
    if not os.path.exists(datadir+path):
        print(path)
        urllib.request.urlretrieve(url, datadir+path)
