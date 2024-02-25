import os
import numpy as np
import re
from functools import reduce
import time

# READ_LENGTHS = [28, 29, 30, 31]
STARTCODONS = [
  'AUG', 'CUG', 'GUG', 'UUG', 'AAG',
  'ACG', 'AGG', 'AUA', 'AUC', 'AUU'
]
STOPCODONS = ['UAA', 'UAG', 'UGA']


class Mappability:
  UUU = 0
  UUM = 1
  UMU = 2
  UMM = 3
  MUU = 4
  MUM = 5
  MMU = 6
  MMM = 7


class States:
  ST_5PRIME_UTS = 0
  ST_5PRIME_UTS_PLUS = 1
  ST_TIS = 2
  ST_TIS_PLUS = 3
  ST_TES = 4
  ST_TTS_MINUS = 5
  ST_TTS = 6
  ST_3PRIME_UTS_MINUS = 7
  ST_3PRIME_UTS = 8


binarize = dict([
  (0, np.array([False, False, False])),
  (1, np.array([False, False, True])),
  (2, np.array([False, True, False])),
  (3, np.array([False, True, True])),
  (4, np.array([True, False, False])),
  (5, np.array([True, False, True])),
  (6, np.array([True, True, False])),
  (7, np.array([True, True, True]))
])

# debinarize = dict([(val.tostring(),key) for key,val in binarize.iteritems()])
debinarize = dict([(val.tostring(), key) for key, val in binarize.items()])

# some essential functions
insum = lambda x, axes: np.apply_over_axes(np.sum, x, axes)
nplog = lambda x: np.nan_to_num(np.log(x))
andop = lambda x: reduce(lambda y, z: np.logical_and(y, z), x)
EPS = np.finfo(np.double).resolution
MAX = np.finfo(np.double).max
MIN = np.finfo(np.double).min

# nucleotide operations
DNA_COMPLEMENT = dict([('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G'), ('N', 'N')])

make_complement = lambda seq: [DNA_COMPLEMENT[s] for s in seq]
make_reverse_complement = lambda seq: [DNA_COMPLEMENT[s] for s in seq][::-1]
makestr = lambda seq: ''.join(map(chr, seq))

CODON_AA_MAP = dict([('GCU', 'A'), ('GCC', 'A'), ('GCA', 'A'), ('GCG', 'A'), \
                     ('UGU', 'C'), ('UGC', 'C'), \
                     ('GAU', 'D'), ('GAC', 'D'), \
                     ('GAA', 'E'), ('GAG', 'E'), \
                     ('CGU', 'R'), ('CGC', 'R'), ('CGA', 'R'), ('CGG', 'R'), ('AGA', 'R'), ('AGG', 'R'), \
                     ('UUU', 'F'), ('UUC', 'F'), \
                     ('GGU', 'G'), ('GGC', 'G'), ('GGA', 'G'), ('GGG', 'G'), \
                     ('CAU', 'H'), ('CAC', 'H'), \
                     ('AAA', 'K'), ('AAG', 'K'), \
                     ('UUA', 'L'), ('UUG', 'L'), ('CUU', 'L'), ('CUC', 'L'), ('CUA', 'L'), ('CUG', 'L'), \
                     ('AUU', 'I'), ('AUC', 'I'), ('AUA', 'I'), \
                     ('AUG', 'M'), \
                     ('AAU', 'N'), ('AAC', 'N'), \
                     ('CCU', 'P'), ('CCC', 'P'), ('CCA', 'P'), ('CCG', 'P'), \
                     ('CAA', 'Q'), ('CAG', 'Q'), \
                     ('UCU', 'S'), ('UCC', 'S'), ('UCA', 'S'), ('UCG', 'S'), ('AGU', 'S'), ('AGC', 'S'), \
                     ('ACU', 'T'), ('ACC', 'T'), ('ACA', 'T'), ('ACG', 'T'), \
                     ('GUU', 'V'), ('GUC', 'V'), ('GUA', 'V'), ('GUG', 'V'), \
                     ('UGG', 'W'), \
                     ('UAU', 'Y'), ('UAC', 'Y'), \
                     ('UAA', 'X'), ('UGA', 'X'), ('UAG', 'X')])

translate = lambda seq: ''.join([
  CODON_AA_MAP[seq[s:s + 3]] if seq[s:s + 3] in CODON_AA_MAP else 'X'
  for s in range(0, len(seq), 3)
])


def make_cigar(mask):
  char = ['M', 'N']
  if np.all(mask):
    cigar = ['%dM' % mask.sum()]
  else:
    switches = list(np.where(np.logical_xor(mask[:-1], mask[1:]))[0] + 1)
    switches.insert(0, 0)
    cigar = ['%d%s' % (switches[i + 1] - switches[i], char[i % 2]) for i in range(len(switches) - 1)]
    cigar.append('%d%s' % (mask.size - switches[-1], char[(i + 1) % 2]))
  return ''.join(cigar)


def make_mask(cigar):
  intervals = map(int, re.split('[MN]', cigar)[:-1])
  mask = np.zeros((np.sum(intervals),), dtype='bool')
  for i, inter in enumerate(intervals[::2]):
    mask[np.sum(intervals[:2 * i]):np.sum(intervals[:2 * i]) + inter] = True
  return mask


def get_exons(mask):
  if np.all(mask):
    exons = ['1', '%d,' % mask.sum(), '0,']
  else:
    exons = [0, [], []]
    switches = list(np.where(np.logical_xor(mask[:-1], mask[1:]))[0] + 1)
    switches.insert(0, 0)
    exons[0] = '%d' % (len(switches[::2]))
    exons[2] = ','.join(map(str, switches[::2])) + ','
    exons[1] = ','.join(map(str, [switches[i + 1] - switches[i] for i in range(0, len(switches) - 1, 2)])) + ','
    exons[1] = exons[1] + '%d,' % (mask.size - switches[-1])
  return exons


def outsum(arr):
  """Summation over the first axis, without changing length of shape.

  Arguments
      arr : array

  Returns
      thesum : array

  .. note::
      This implementation is much faster than `numpy.sum`.

  """

  thesum = sum([a for a in arr])
  shape = [1]
  shape.extend(list(thesum.shape))
  thesum = thesum.reshape(tuple(shape))
  return thesum


# def get_read_lengths():
#     return READ_LENGTHS


def which(program):
  """
  Get the path of an executable program in the $PATH
  environment variable
  :param program: str Name of the executable
  :return: str Full path to the executable in $PATH or
  None if not found
  """

  def _is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

  fpath, fname = os.path.split(program)
  if fpath and _is_exe(program):
    return program

  for path in os.environ['PATH'].split(os.pathsep):
    path = path.strip('"')
    exe_file = os.path.join(path, program)
    if _is_exe(exe_file):
      return exe_file

  return None


class Timer:
  def __init__(self, start_on_init=True):
    self._start_time = None
    if start_on_init:
      self._start_time = time.time()

  def start(self):
    if self._start_time is None:
      self._start_time = time.time()
    return self

  def restart(self):
    self._start_time = time.time()
    return self

  def elapsed(self):
    return time.time() - self._start_time


def find_start_codon(data, annotated_CDS_dict):
  for strand in ('pos', 'neg'):
    for trns_i, trns in enumerate(data[strand]):
      has_start_codon = list()
      within_five = list()

      # Get the transcript start and stop
      transcript_start = trns['transcript_info']['start']
      transcript_stop = trns['transcript_info']['stop']
      transcript_id = trns['transcript_info']['ref_id']
      transcript_seq = trns['sequence']

      # Get the annotated start
      annotated_start_base_pos = annotated_CDS_dict['start'][transcript_id]
      annotated_start_triplet_i, annotated_start_frame = divmod(annotated_start_base_pos - transcript_start, 3)

      # start_codon_index = 0 if orf['strand'] == '+' else 2
      for i, orf in enumerate(trns['results']['candidate_orf']):
        start_codon_index = 0 if orf['strand'] == '+' else 2
        if orf['start_codon_genomic_position'][start_codon_index] == orf['annotated_start']:
          has_start_codon.append(orf['definition'])
        if abs(orf['start_codon_genomic_position'][start_codon_index] - orf['annotated_start']) < 5:
          within_five.append(orf['definition'])

      # Determine whether this transcript passes
      if len(has_start_codon) > 0:
        print('{} transcript {} has detectable annotated start codon'.format(strand, trns['transcript_info'].get('id')))
      else:
        print('=====================================')
        if len(within_five) > 0:
          print('{} transcript {} has detectable annotated start codon within 5bp (index {})'.format(strand, trns[
            'transcript_info'].get('id'), trns_i))
        else:
          print('!!!!!! {} transcript {} is missing the annotated start codon (index {})'.format(strand, trns[
            'transcript_info'].get('id'), trns_i))

        # Show details
        print(f'transcript_info: {trns["transcript_info"]}')
        print(f'annotated start base pos: {annotated_start_base_pos}')
        print(f'annotated start triplet i: {annotated_start_triplet_i}')
        print(f'annotated start frame: {annotated_start_frame}')
        print('annotated start triplet i: {}'.format(trns['results']['candidate_orf'][0]['annotated_triplet_i']))
        print(f'exon mask: {trns["exons"]["mask"][:20]}')
        get_codon_from_seq(
          seq=transcript_seq,
          frame=annotated_start_frame,
          triplet_i=annotated_start_triplet_i - 10,
          stop_i=annotated_start_triplet_i + 10,
          show_index=True,
          transcript_start=transcript_start
        )
        print('**************************')
        get_codon_from_seq(
          seq=transcript_seq,
          frame=annotated_start_frame,
          triplet_i=trns['results']['candidate_orf'][0]['annotated_triplet_i'][0][0] - 10,
          stop_i=trns['results']['candidate_orf'][0]['annotated_triplet_i'][0][0] + 10,
          show_index=True,
          transcript_start=transcript_start
        )
        print('=====================================')



def get_codon_from_seq(seq, frame, triplet_i, stop_i=None, show_index=False, transcript_start=None):
  import math
  STARTCODONS = [
    'AUG', 'CUG', 'GUG', 'UUG', 'AAG',
    'ACG', 'AGG', 'AUA', 'AUC', 'AUU'
  ]
  STOPCODONS = ['UAA', 'UAG', 'UGA']
  stop_i = stop_i or triplet_i

  zfill_padding = math.ceil(math.log10(len(seq) / 3))
  for i in range(triplet_i, stop_i + 1):
    codon = seq[(3 * i) + frame:(3 * i) + frame + 3]
    codon_idx = '[{}][{}] '.format(str(i).zfill(zfill_padding), transcript_start + (3 * i) + frame) if show_index else ''
    if codon in STARTCODONS:
      print(f'{codon_idx}{codon} [Start]')
    elif codon in STOPCODONS:
      print(f'{codon_idx}{codon} [Stop]')
    else:
      print(f'{codon_idx}{codon}')


def write_seq(seq, numbers=False):
  out = ''
  for i, s in enumerate(seq):
    if i % 3 == 0:
      out += ' '
      if numbers:
        out += '[{}]'.format((int(i / 3)))
    out += s
  return out.strip()
  if divmod(len(seq), 3)[1] != 0:
    offset = 1
  else:
    offset = 0
  return ' '.join([seq[s:s + 3] for s in range(int(len(seq) / 3) + offset)])



pos_transcripts_without_start_codon = [
  'STRG.6369.3',
  'STRG.6877.1',
  'STRG.7104.1',
  'STRG.7464.6',
  'STRG.7464.30',
  'STRG.7464.33'
]
