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


