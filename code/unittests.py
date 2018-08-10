import numpy.testing as npt
from lib import *

def test_count_kmers():
    d = count_kmers('ABCD', 2)
    assert d['AB'] == 1
    assert d['BC'] == 1
    assert d['CD'] == 1

if __name__ == '__main__':
    npt.run_module_suite()
