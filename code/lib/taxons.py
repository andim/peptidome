from subprocess import check_output
from io import StringIO
import json


class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        # if args is not Hashable we can't cache
        # easier to ask for forgiveness than permission
        try:
            if args in self.cache:
                return self.cache[args]
            else:
                value = self.func(*args)
                self.cache[args] = value
                return value
        except TypeError:
            return self.func(*args)

from joblib import Memory
cachedir = 'cache'
memory = Memory(cachedir, verbose=0)

@memory.cache
def normalize_taxid(taxid, rank='species'):
    """return species level taxon id"""
    if not taxid:
        return ''
    out = check_output('efetch -db taxonomy -id "{taxid}"  -format native -mode xml -json'.format(taxid=taxid), shell=True, text=True)
    buffer = StringIO(out)
    results = json.load(buffer)
    if results['TaxaSet']['Taxon']['Rank'] == 'species':
        print(taxid, 'is species')
        return taxid
    try:
        print(taxid, len(results['TaxaSet']['Taxon']['LineageEx']['Taxon']))
        ids = [level['TaxId'] for level in results['TaxaSet']['Taxon']['LineageEx']['Taxon'] if level['Rank'] == rank]
    except (KeyError, TypeError):
        return ''
    if len(ids) == 0:
        return ''
    return ids[0] 
