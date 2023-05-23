import numpy as np
from pkg_resources import Requirement, resource_listdir, resource_filename
from ribohmm.utils import STARTCODONS, STOPCODONS

import logging
logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d|%(levelname)s] %(message)s',
    datefmt='%d%b%Y %H:%M:%S',
    level=logging.DEBUG
)
logger = logging.getLogger('viterbi_log')

# set regex for start and stop codons
STARTS = {codon_name: i for i, codon_name in enumerate(STARTCODONS, start=1)}
STOPS = {codon_name: i for i, codon_name in enumerate(STOPCODONS, start=1)}

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
    logger.debug('In function get_resource_kozak_path()')
    logger.debug('Loading kozak model from path: {}'.format(resource_filename(Requirement.parse('ribohmm'), 'ribohmm/include/kozak_model.npz')))
    return resource_filename(Requirement.parse('ribohmm'), 'ribohmm/include/kozak_model.npz')


def inflate_kozak_model(model_path=None):
    """
    Inflates and stores the Kozak model as class attributes of ``RnaSequence``
    """
    logger.debug('In function inflate_kozak_model()')
    if model_path is None:
        logger.debug('model_path is None')
        model_path = get_resource_kozak_path()
        logger.debug('Got model path as {}'.format(model_path))
    kozak_model = np.load(model_path)
    logger.debug('Loaded kozak model')
    FREQ = dict([(c, np.log2(row)) for c, row in zip(['A', 'U', 'G', 'C'], kozak_model['freq'])])
    logger.debug('Calculated FREQ')
    ALTFREQ = dict([(c, np.log2(row)) for c, row in zip(['A', 'U', 'G', 'C'], kozak_model['altfreq'])])
    logger.debug('Calculated ALTFREQ')
    for c in ['A', 'U', 'G', 'C']:
        FREQ[c][9:12] = ALTFREQ[c][9:12]
    logger.debug('Assigned some ALTFREQ to FREQ')

    RnaSequence._kozak_model_freq = FREQ
    logger.debug('Assigned FREQ to RnaSequence._kozak_model_freq')
    RnaSequence._kozak_model_altfreq = ALTFREQ
    logger.debug('Assigned ALTFREQ to RnaSequence._kozak_model_altfreq')


class RnaSequence(object):
    _kozak_model_freq = None
    _kozak_model_altfreq = None

    def __init__(self, sequence):
        self.sequence = sequence
        self.S = len(self.sequence)
        self.sequence_length = len(self.sequence)

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
        offset = 3
        n_triplets = int(self.sequence_length / 3) - 1
        start_codon_map = np.zeros(shape=(n_triplets, 3), dtype=np.uint8)
        for frame_i in range(3):  # For each open reading frame
            for codon_start_pos in range(frame_i, 3 * n_triplets + frame_i, 3):  # TODO Python 2/3
                triplet_i = int(codon_start_pos / 3)
                try:
                    start_codon_map[triplet_i, frame_i] = STARTS[self.sequence[codon_start_pos + offset:codon_start_pos + offset + 3]]
                except KeyError:
                    pass
                for k in [3, 6, 9, 12]:
                    try:
                        STOPS[self.sequence[codon_start_pos + offset + k:codon_start_pos + offset + 3 + k]]
                        start_codon_map[triplet_i, frame_i] = 0
                    except KeyError:
                        pass

        return start_codon_map

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    def _mark_stop_codons(self):
        offset = 6
        n_triplets = int(self.sequence_length / 3) - 1
        stop_codon_map = np.zeros(shape=(n_triplets, 3), dtype=np.uint8)
        for frame_i in range(3):
            for codon_start_pos in range(frame_i, 3 * n_triplets + frame_i, 3):
                triplet_i = int(codon_start_pos / 3)
                codon_spelling = self.sequence[codon_start_pos + offset:codon_start_pos + offset + 3]
                if codon_spelling in STOPS:
                    stop_codon_map[triplet_i, frame_i] = STOPS[codon_spelling]

        return stop_codon_map

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    def _compute_kozak_scores(self):

        # cdef int offset, s, f
        # cdef np.ndarray score

        offset = 3
        try:
            score = np.zeros((int(self.sequence_length/3) - 1, 3), dtype=float)
        except:
            print('Trying to create dimensions ({}, 3)'.format(int(self.S/3) - 1))
            print('self.__dict__ = {}'.format(self.__dict__))
            raise
        for f in range(3):
            for s in range(2, int((self.sequence_length-f-4-offset) / 3)):
                score[s, f] = RnaSequence.pwm_score(self.sequence[3*s+offset+f-9:3*s+offset+f+4])

        return score  # np.array[*, *]

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    @classmethod
    def pwm_score(cls, seq):
        logger.debug('In method pwm_score()')

        # cdef long i
        # cdef str s
        # cdef double score

        if not (cls._kozak_model_freq and cls._kozak_model_altfreq):
            logger.debug('not (cls._kozak_model_freq and cls._kozak_model_altfreq)')
            raise ValueError('Kozak models have not been loaded')

        score = 0
        for i, s in enumerate(seq):
            try:
                score = score + cls._kozak_model_freq[s][i] - cls._kozak_model_altfreq[s][i]
            except KeyError:
                pass

        logger.debug('Completed function pwm_score()')
        return score
