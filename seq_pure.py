import time
import re
import pdb

import numpy as np
from utils import STARTCODONS, STOPCODONS

# set regex for start and stop codons
STARTS = dict([(s,i+1) for i,s in enumerate(STARTCODONS)])
STOPS = dict([(s,i+1) for i,s in enumerate(STOPCODONS)])

# load pre-computed Kozak model
kozak_model = np.load("data/kozak_model.npz")
FREQ = dict([(c,np.log2(row)) for c,row in zip(['A','U','G','C'], kozak_model['freq'])])
ALTFREQ = dict([(c,np.log2(row)) for c,row in zip(['A','U','G','C'], kozak_model['altfreq'])])
for c in ['A','U','G','C']:
    FREQ[c][9:12] = ALTFREQ[c][9:12]

class RnaSequence(object):
    def __init__(self, sequence):

        # cdef str c
        # cdef dict kozak_model

        self.sequence = sequence
        self.S = len(self.sequence)

    def mark_codons(self):
        # cdef dict codon_flags
        codon_flags = dict()

        codon_flags['kozak'] = self._compute_kozak_scores()
        codon_flags['start'] = self._mark_start_codons()
        codon_flags['stop'] = self._mark_stop_codons()

        return codon_flags

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    def _mark_start_codons(self):

        # cdef int offset, s, f, S, M
        # cdef str start
        # cdef np.ndarray start_index

        offset = 3
        M = int(self.S/3) - 1
        start_index = np.zeros((M,3), dtype=np.uint8)
        for f in range(3):  # TODO Python 2/3 compatibility
            for s in range(f,3*M+f,3):  # TODO Python 2/3
                try:
                    start_index[int(s/3),f] = STARTS[self.sequence[s+offset:s+offset+3]]
                except KeyError:
                    pass
                for k in [3,6,9,12]:
                    try:
                        STOPS[self.sequence[s+offset+k:s+offset+3+k]]
                        start_index[int(s/3),f] = 0
                    except KeyError:
                        pass

        return start_index

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    def _mark_stop_codons(self):

        # cdef int offset, s, f, M
        # cdef str stop
        # cdef np.ndarray stop_index

        offset = 6
        M = int(self.S/3) - 1
        stop_index = np.zeros((M, 3), dtype=np.uint8)
        for f in range(3):
            for s in range(f, 3*M+f, 3):
                try:
                    stop_index[int(s/3),f] = STOPS[self.sequence[s+offset:s+offset+3]]
                except KeyError:
                    pass

        return stop_index

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    def _compute_kozak_scores(self):

        # cdef int offset, s, f
        # cdef np.ndarray score

        offset = 3
        try:
            score = np.zeros((int(self.S/3) - 1, 3), dtype=float)
        except:
            print('Trying to create dimensions ({}, 3)'.format(int(self.S/3) - 1))
            print('self.__dict__ = {}'.format(self.__dict__))
            raise
        for f in range(3):
            for s in range(2, int((self.S-f-4-offset) / 3)):
                score[s,f] = pwm_score(self.sequence[3*s+offset+f-9:3*s+offset+f+4])

        return score  # np.array[*, *]

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
def pwm_score(seq):

    # cdef long i
    # cdef str s
    # cdef double score

    score = 0
    for i, s in enumerate(seq):
        try:
            score = score + FREQ[s][i] - ALTFREQ[s][i]
        except KeyError:
            pass

    return score
