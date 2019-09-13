import numpy.testing as npt
from lib import *
import clib

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

def test_energy():
    for energy in [energy_ising, clib.energy]:
        N, q = 2, 2
        h = np.zeros(q)
        Jk = np.zeros((N-1, q, q))
        s = np.array([0, 1])
        assert energy(s, h, Jk) == 0.0
        h = np.array([0.1, 0.0])
        assert energy(s, h, Jk) == -0.1
        s = np.array([1, 0])
        assert energy(s, h, Jk) == -0.1
        s = np.array([0, 0])
        assert energy(s, h, Jk) == -0.2
        s = np.array([1, 1])
        assert energy(s, h, Jk) == 0.0
        h = np.zeros(q)
        Jk[0, 0, 0] = 0.5
        s = np.array([0, 0])
        assert energy(s, h, Jk) == -0.5
        s = np.array([0, 1])
        assert energy(s, h, Jk) == 0.0

def test_mapping():
    npt.assert_array_equal(map_aatonumber('AAAA'), np.zeros(4))


if __name__ == '__main__':
    npt.run_module_suite()
