cimport numpy as np

def energy_cython(np.ndarray[long, ndim=1] s, np.ndarray[double, ndim=1] h, np.ndarray[double, ndim=3] J):
    cdef int N = len(s)
    cdef double energy = 0.0
    cdef int i, j
    for i in range(N):
        energy += h[s[i]]
        for j in range(i, N):
            energy += J[j-i-1, s[i], s[j]]
    return energy
