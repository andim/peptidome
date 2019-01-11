import os.path
import re
import urllib.request
import pandas as pd

from lib import *

proteomes = load_proteomes()
# download proteomes from uniprot
for ind, row in proteomes.iterrows():
    if not os.path.exists(datadir+row['path']):
        if row['speciesid']:
            url = r"ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/%s_%s.fasta.gz"%(row['proteomeid'], row['speciesid'])
        else:
         
            url = r"http://www.uniprot.org/uniprot/?query=proteome%3A"+row['proteomeid']+r"&format=fasta&include=no"
        print(url)
        urllib.request.urlretrieve(url, datadir+row['path'])

# download immune epitopes from iedb
path_urls = [('iedb-tcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip'),
             ('iedb-bcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3.zip') 
            ]
for path, url in path_urls:
    if not os.path.exists(datadir+path):
        print(path)
        urllib.request.urlretrieve(url, datadir+path)
