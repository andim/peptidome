import numpy.testing as npt
from lib import *

def test_count_kmers():
    d = count_kmers('ABCD', 2)
    assert d['AB'] == 1
    assert d['BC'] == 1
    assert d['CD'] == 1
    d = count_kmers('ABCD', 2, gap=1)
    assert d['AC'] == 1
    assert d['BD'] == 1
    assert d['AB'] == 0
    assert d['CD'] == 0
    d = count_kmers('ABCD', 2, gap=2)
    assert d['AD'] == 1

if __name__ == '__main__':
    npt.run_module_suite()
