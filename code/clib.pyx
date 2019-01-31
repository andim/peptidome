from collections import defaultdict
cimport cython
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

def count_kmers(string, int k, counter=None, int gap=0):
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
