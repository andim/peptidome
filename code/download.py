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
            # excluding isoforms (include=no)
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
# - all in Swiss-prot (manually annotated)
# - filtered at 90% or 50% sequence identity level
path_urls = [
        ('human-viruses-swissprot.fasta', r'https://www.uniprot.org/uniprot/?query=host:9606&fil=reviewed%3Ayes&format=fasta&include=no'),
        ('human-viruses-uniref90.fasta', 
         r"https://www.uniprot.org/uniref/?query=uniprot%3A(host%3A%22Homo+sapiens+(Human)+[9606]%22)+AND+identity%3A0.9&sort=score&format=fasta"),
        ('chicken-viruses-uniref90.fasta', 
        r"https://www.uniprot.org/uniref/?query=uniprot:(host%3A%22Gallus+gallus+%28Chicken%29+%5B9031%5D%22)+AND+identity:0.9&sort=score&format=fasta"),
        ('human-viruses-uniref50.fasta', 
         r"https://www.uniprot.org/uniref/?query=uniprot%3A(host%3A%22Homo+sapiens+(Human)+[9606]%22)+AND+identity%3A0.5&sort=score&format=fasta")
        ]
for path, url in path_urls:
    if not os.path.exists(datadir+path):
        try:
            urllib.request.urlretrieve(url, datadir+path)
        except:
            print('could not download %s'%path)

# filter HIV1 proteins from the human viruse
paths = [('human-viruses-uniref90.fasta', 'human-viruses-uniref90_nohiv.fasta'),
         ('human-viruses-uniref50.fasta', 'human-viruses-uniref50_nohiv.fasta')]
for pathin, pathout in paths:
    if not os.path.exists(datadir+pathout):
        def load(fasta_name):
            with open(fasta_name) as fastain:
                for seq in SeqIO.parse(fastain, 'fasta'):
                    if not ('Human immunodeficiency virus 1' in seq.description):
                        yield seq
        with open(datadir+pathout, "w") as fastaout:
            SeqIO.write(list(load(datadir+pathin)), fastaout, "fasta")


path_urls = [
             # immune epitope data from IEDB
             ('iedb-tcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip'),
             ('iedb-bcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3.zip'),
             # Human protein atlas (tissue restriction, subcellular location)
             ('proteinatlas.tsv.zip', r'https://www.proteinatlas.org/download/proteinatlas.tsv.zip'),
             ('proteinatlas.xml.gz', r'https://www.proteinatlas.org/download/proteinatlas.xml.gz'),
             # Viral Metadata repository
             ('vmr.xlsx', r'https://talk.ictvonline.org/taxonomy/vmr/m/vmr-file-repository/9603/download'),
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
