import os.path
import re
import urllib.request
import pandas as pd

from lib import *

# download proteomes from uniprot
proteomes = pd.read_csv(datadir + 'proteomes.csv')
for path in proteomes['path']:
    if not os.path.exists(datadir+path):
        print(path)
        proteomeid = re.search('(?<=UP)[0-9]+', path).group(0)
        urllib.request.urlretrieve(r"http://www.uniprot.org/uniprot/?query=proteome%3AUP"+proteomeid+r"&format=fasta", datadir+path)

# download immune epitopes from iedb
path_urls = [('iedb-tcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip'),
             ('iedb-bcell.zip', r'http://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3.zip') 
            ]
for path, url in path_urls:
    if not os.path.exists(datadir+path):
        print(path)
        urllib.request.urlretrieve(url, datadir+path)
