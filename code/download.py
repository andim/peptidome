import os.path
import re
import urllib.request
import pandas as pd

from lib import *

proteomes = pd.read_csv(datadir + 'proteomes.csv')

for path in proteomes['path']:
    if not os.path.exists(datadir+path):
        print(path)
        proteomeid = re.search('(?<=UP)[0-9]+', path).group(0)
        urllib.request.urlretrieve(r"http://www.uniprot.org/uniprot/?query=proteome%3AUP"+proteomeid+r"&format=fasta", datadir+path)

urliedb = r'http://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip'
path = 'iedb-tcell.zip'
if not os.path.exists(datadir+path):
    print(path)
    urllib.request.urlretrieve(urliedb, datadir+path)
