import numpy as np
from pkg_resources import Requirement, resource_listdir, resource_filename
from ribohmm.utils import STARTCODONS, STOPCODONS

# set regex for start and stop codons
STARTS = dict([(s,i+1) for i,s in enumerate(STARTCODONS)])
STOPS = dict([(s,i+1) for i,s in enumerate(STOPCODONS)])

# load pre-computed Kozak model
# kozak_model = np.load("data/kozak_model.npz")
# FREQ = dict([(c,np.log2(row)) for c,row in zip(['A','U','G','C'], kozak_model['freq'])])
# ALTFREQ = dict([(c,np.log2(row)) for c,row in zip(['A','U','G','C'], kozak_model['altfreq'])])
# for c in ['A','U','G','C']:
#     FREQ[c][9:12] = ALTFREQ[c][9:12]


# def get_util_scripts():
#     util_scripts = dict()
#     for util_script_filename in resource_listdir(Requirement.parse('swiftseq'), 'swiftseq/util_scripts'):
#         util_name = util_script_filename.rsplit('.', 1)[FIRST]
#         util_full_filepath = resource_filename(Requirement.parse('swiftseq'), 'swiftseq/util_scripts/{}'.format(
#             util_script_filename
#         ))
#         util_scripts['util_{}'.format(util_name)] = util_full_filepath
#     return util_scripts


def get_resource_kozak_path():
    return resource_filename(Requirement.parse('ribohmm'), 'ribohmm/include/kozak_model.npz')


def inflate_kozak_model(model_path=None):
    """
    Inflates and stores the Kozak model as class attributes of ``RnaSequence``
    """
    if model_path is None:
        model_path = get_resource_kozak_path()
    kozak_model = np.load(model_path)
    FREQ = dict([(c, np.log2(row)) for c, row in zip(['A', 'U', 'G', 'C'], kozak_model['freq'])])
    ALTFREQ = dict([(c, np.log2(row)) for c, row in zip(['A', 'U', 'G', 'C'], kozak_model['altfreq'])])
    for c in ['A', 'U', 'G', 'C']:
        FREQ[c][9:12] = ALTFREQ[c][9:12]

    RnaSequence._kozak_model_freq = FREQ
    RnaSequence._kozak_model_altfreq = ALTFREQ


class RnaSequence(object):
    _kozak_model_freq = None
    _kozak_model_altfreq = None

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
        M = int(self.S / 3) - 1
        start_index = np.zeros((M, 3), dtype=np.uint8)
        for f in range(3):  # TODO Python 2/3 compatibility
            for s in range(f, 3 * M + f, 3):  # TODO Python 2/3
                try:
                    start_index[int(s/3), f] = STARTS[self.sequence[s + offset:s + offset + 3]]
                except KeyError:
                    pass
                for k in [3, 6, 9, 12]:
                    try:
                        STOPS[self.sequence[s + offset + k:s + offset + 3 + k]]
                        start_index[int(s/3), f] = 0
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
        M = int(self.S / 3) - 1
        stop_index = np.zeros((M, 3), dtype=np.uint8)
        for f in range(3):
            for s in range(f, 3 * M + f, 3):
                try:
                    stop_index[int(s/3), f] = STOPS[self.sequence[s + offset:s + offset + 3]]
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
                score[s, f] = RnaSequence.pwm_score(self.sequence[3*s+offset+f-9:3*s+offset+f+4])

        return score  # np.array[*, *]

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    @classmethod
    def pwm_score(cls, seq):

        # cdef long i
        # cdef str s
        # cdef double score

        if not (cls._kozak_model_freq and cls._kozak_model_altfreq):
            raise ValueError('Kozak models have no been loaded')

        score = 0
        for i, s in enumerate(seq):
            try:
                score = score + cls._kozak_model_freq[s][i] - cls._kozak_model_altfreq[s][i]
            except KeyError:
                pass

        return score
