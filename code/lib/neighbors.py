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

def dist3(x, reference):
    """ Is the string x a Hamming distance 3 away from any of the kmers in the reference set"""
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            for k in range(j+1, len(x)):
                for aai in aminoacids:
                    if aai == x[i]:
                        continue
                    si = x[:i]+aai+x[i+1:]
                    for aaj in aminoacids:
                        if aaj == x[j]:
                            continue
                        sij = si[:j]+aaj+si[j+1:] 
                        for aak in aminoacids:
                            if aak == x[k]:
                                continue
                            if sij[:k]+aak+sij[k+1:] in reference:
                                return True
    return False

def distance_distribution(sample, reference):
    """Return the distribution of Hamming distances for all sequences in a sample relative to a reference set."""
    reference = set(reference)
    d0 = np.array([x in reference for x in sample])
    count0 = np.sum(d0)
    d1 = np.array([dist1(x, reference) for x in sample]) & (~d0)
    count1 = np.sum(d1)
    d2 = np.array([dist2(x, reference) for x in sample]) & (~d1) & (~d0)
    count2 = np.sum(d2)
    ns = np.array([count0, count1, count2, len(sample)-count0-count1-count2])
    return ns

def number_hamming_neighbors(distance, length, q):
    return (q-1)**distance * falling_factorial(length, distance)

