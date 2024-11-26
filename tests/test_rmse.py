import pytest
import numpy as np
import sys
sys.path.append('/home/dfitzgerald/workspace/PycharmProjects/RiboHMM')
from ribohmm.core.ribohmm import Data

"""
What do I need to write a test:
  - An emissions probability matrix
  - Mappability information
  - Some number of ORFs as CandidateCDS namedtuples
    - ORFs need to have valid start and stop
  - All of the above for each footprint length
  - Can hijack Data() object info as a part of the test
  - Can we use chrM for this?
"""

# These are proportions
EMISSIONS_BY_STATE = {
  'st_5prime_uts': [1/3, 1/3, 1/3],
  'st_5prime_uts_plus': [0.5, 0.25, 0.25],
  'st_tis': [3/6, 2/6, 1/6],
  'st_tis_plus': [3/6, 2/6, 1/6],
  'st_tes': [3/6, 2/6, 1/6],
  'st_tts_minus': [3/6, 2/6, 1/6],
  'st_tts': [3/6, 2/6, 1/6],
  'st_3prime_uts_minus': [0.5, 0.25, 0.25],
  'st_3prime_uts': [1/3, 1/3, 1/3]
}

EMISSIONS = [
  EMISSIONS_BY_STATE['st_5prime_uts'],
  EMISSIONS_BY_STATE['st_5prime_uts_plus'],
  EMISSIONS_BY_STATE['st_tis'],
  EMISSIONS_BY_STATE['st_tis_plus'],
  EMISSIONS_BY_STATE['st_tes'],
  EMISSIONS_BY_STATE['st_tts_minus'],
  EMISSIONS_BY_STATE['st_tts'],
  EMISSIONS_BY_STATE['st_3prime_uts_minus'],
  EMISSIONS_BY_STATE['st_3prime_uts'],
]










# @pytest.fixture
def get_data_object(last_tes=None):
  last_tes = last_tes or [6, 2, 1]

  # These are pileup counts for specific states
  st_5prime_uts = [0, 0, 0]
  st_5prime_uts_plus = [2, 1, 1]
  st_tis = [3, 2, 1]
  st_tis_plus = [3, 2, 1]
  st_tes = [3, 2, 1]
  st_tts_minus = [3, 2, 1]
  st_tts = [3, 2, 1]
  st_3prime_uts_minus = [2, 1, 1]
  st_3prime_uts = [0, 0, 0]

  # 13 triplets
  # This is the pileup of a transcript with 13 triplets
  seq = np.array([
    st_5prime_uts,
    st_5prime_uts,
    st_5prime_uts_plus,
    st_tis,
    st_tis_plus,
    st_tes,
    st_tes,
    st_tes,
    # [6, 2, 1],  # This is the one triplet which is off
    last_tes,
    st_tts_minus,
    st_tts,
    st_3prime_uts_minus,
    st_3prime_uts
  ]).flatten().tolist()

  riboseq_pileup = np.zeros(shape=(13 * 3, 4))
  riboseq_pileup[:, 0] = seq
  riboseq_pileup[:, 1] = seq
  riboseq_pileup[:, 2] = seq
  riboseq_pileup[:, 3] = seq
  # riboseq_pileup = np.array([
  #   seq,
  #   seq,
  #   seq,
  #   seq,
  # ])

  codon_map = {
    'kozak': None,
    'start': np.array([
      [0, 0, 0],
      [0, 0, 0],
      [1, 0, 0],  # State 2
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]),
    'stop': np.array([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [1, 0, 0],  # State 7
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]),
  }

  # There are 13 triplets and 4 read lengths
  # We want all to be true except for one of the TES
  is_pos_mappable = np.ones(shape=(13 * 3, 4)).astype(bool)
  # Make the first base of the first TES unmappable
  is_pos_mappable[15, 0] = False

  # print(is_pos_mappable)

  data_object = Data(
    riboseq_pileup=riboseq_pileup,
    codon_map=codon_map,
    transcript_normalization_factor=1,
    is_pos_mappable=is_pos_mappable
  )

  return data_object





def get_data_object2(last_tes=None):
  # last_tes = last_tes or [6, 2, 1]

  # These are pileup counts for specific states
  # st_5prime_uts = [0, 0, 0]
  # st_5prime_uts_plus = [2, 1, 1]
  # st_tis = [3, 2, 1]
  # st_tis_plus = [3, 2, 1]
  # st_tes = [3, 2, 1]
  # st_tts_minus = [3, 2, 1]
  # st_tts = [3, 2, 1]
  # st_3prime_uts_minus = [2, 1, 1]
  # st_3prime_uts = [0, 0, 0]

  # 15 triplets
  # This is the pileup of a transcript with 13 triplets
  seq = np.array([
       [0, 1, 2],
       [6, 3, 1],
       [4, 3, 1],
       [0, 5, 1],
       [4, 2, 3],
       [1, 2, 7],
       [1, 4, 7],
       [3, 7, 0],
       [6, 6, 5],
       [7, 7, 4],
       [5, 3, 4],
       [0, 0, 7],
       [0, 6, 1],
       [3, 4, 6],
       [0, 1, 1]]).flatten().tolist()
  # seq = np.array([
  #   st_5prime_uts,
  #   st_5prime_uts,
  #   st_5prime_uts_plus,
  #   st_tis,
  #   st_tis_plus,
  #   st_tes,
  #   st_tes,
  #   st_tes,
  #   # [6, 2, 1],  # This is the one triplet which is off
  #   last_tes,
  #   st_tts_minus,
  #   st_tts,
  #   st_3prime_uts_minus,
  #   st_3prime_uts
  # ]).flatten().tolist()

  riboseq_pileup = np.zeros(shape=(15 * 3, 4))
  riboseq_pileup[:, 0] = seq
  riboseq_pileup[:, 1] = seq
  riboseq_pileup[:, 2] = seq
  riboseq_pileup[:, 3] = seq
  # riboseq_pileup = np.array([
  #   seq,
  #   seq,
  #   seq,
  #   seq,
  # ])

  codon_map = {
    'kozak': None,
    'start': np.array([
       [0, 0, 0],
       [0, 0, 0],
       [1, 0, 0],  # TODO Change this to [1, 1, 0]
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]),
    'stop': np.array([
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [1, 0, 0],
       [0, 0, 0],
       [0, 0, 0],  # TODO Change this to [0, 1, 0]
       [0, 0, 0]])
  }

  # There are 15 triplets and 4 read lengths
  # We want all to be true except for one of the TES
  is_pos_mappable = np.ones(shape=(15 * 3, 4)).astype(bool)
  # Make the first base of the first TES unmappable
  # is_pos_mappable[0, 0] = False  TODO Add this in later

  # print(is_pos_mappable)

  data_object = Data(
    riboseq_pileup=riboseq_pileup,
    codon_map=codon_map,
    transcript_normalization_factor=1,
    is_pos_mappable=is_pos_mappable
  )

  return data_object


# @pytest.mark.parametrize('last_tes,expected_value', [
#   # (None, 0.0385),  TODO Re-enable these once they're fixed
#   # ([3, 2, 1], 0.0)
# ])
# def test_compute_observed_pileup_deviation(last_tes, expected_value):
def test_compute_observed_pileup_deviation():
  test_emission = {
    'logperiodicity': np.log(np.array([
      EMISSIONS,
      EMISSIONS,
      EMISSIONS,
      EMISSIONS,
    ]))
  }
  data = get_data_object2()

  rmse_results = data.compute_observed_pileup_deviation(
    emission=test_emission
  )

  # assert round(rmse_results[0][3], 4) == pytest.approx(0.2751)
  assert round(rmse_results[0][1], 4) == pytest.approx(round(0.2776771339439869, 4))
  # assert round(rmse_results[0][3], 4) == pytest.approx(0.0385)


def test_load_file():
  import os
  print(os.getcwd())
  with open('tests/test_corpus.gtf') as gtf:
    pass


# def test_compute_observed_pileup_deviation_zero(data):
#   test_emission = {
#     'logperiodicity': np.log(np.array([
#       EMISSIONS,
#       EMISSIONS,
#       EMISSIONS,
#       EMISSIONS,
#     ]))
#   }
#
#   rmse_results = data.compute_observed_pileup_deviation(
#     emission=test_emission
#   )
#
#   assert round(rmse_results[0][3], 4) == pytest.approx(0.0)

