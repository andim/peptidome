from collections import defaultdict
cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def energy(np.ndarray[long, ndim=1] s, np.ndarray[double, ndim=1] h, np.ndarray[double, ndim=3] J):
    cdef int N = len(s)
    cdef double energy = 0.0
    cdef int i, j
    for i in range(N):
        energy += h[s[i]]
        for j in range(i+1, N):
            energy += J[j-i-1, s[i], s[j]]
    return -energy


cdef int naminoacids = 20
@cython.boundscheck(False)
@cython.wraparound(False)
def aacounts_int(np.ndarray[np.int64_t, ndim=1] seq):
    cdef np.ndarray[np.int64_t, cast=True] counter = np.zeros(naminoacids, dtype=np.int64)
    cdef Py_ssize_t i
    for i in range(seq.shape[0]):
        counter[seq[i]] += 1
    return counter

def count_kmers(str string, int k, counter=None, int gap=0):
    """
    Count occurrence of kmers in a given string.
    """
    if counter is None:
        counter = defaultdict(int)
    cdef int i
    cdef int N = len(string)
    for i in range(N-k-gap+1):
        if gap:
            counter[string[i]+string[i+gap+1:i+k+gap]] += 1
        else:
            counter[string[i:i+k]] += 1
    return counter
