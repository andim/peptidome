import numpy as np
from sklearn.neighbors import BallTree

from .main import *

class BallTreeDist:
    """
    Speed up nearest neighbor distance calculation using a BallTree data structure
    """
    def __init__(self, sequences, nchunks=1):
        sequences_number = np.asarray([map_aatonumber(seq) for seq in sequences])
        if nchunks == 1:
            self.bts = [BallTree(sequences_number, metric='hamming')]
        else:
            seqss = np.array_split(sequences_number, nchunks)
            self.bts = [BallTree(seqs, metric='hamming') for seqs in seqss]
           
    def mindist(self, sequence):
        sequence_number= map_aatonumber(sequence).reshape(1, -1)
        d = min(bt.query(sequence_number)[0] for bt in self.bts)
        return int(d*len(sequence))

def dist1(x, reference):
    """ Is the kmer x a Hamming distance 1 away from any of the kmers in the reference set"""
    for i in range(len(x)):
        for aa in aminoacids:
            if aa == x[i]:
                continue
            if x[:i]+aa+x[i+1:] in reference:
                return True
    return False

def dist2(x, reference):
    """ Is the string x a Hamming distance 2 away from any of the kmers in the reference set"""
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            for aai in aminoacids:
                if aai == x[i]:
                    continue
                si = x[:i]+aai+x[i+1:]
                for aaj in aminoacids:
                    if aaj == x[j]:
                        continue
                    if si[:j]+aaj+si[j+1:] in reference:
                        return True
    return False

def number_hamming_neighbors(distance, length, q):
    return (q-1)**distance * falling_factorial(length, distance)

