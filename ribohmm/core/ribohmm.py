import numpy as np
from numpy.ma.core import MaskedArray
from scipy.special import gammaln, digamma, polygamma
from math import log, exp
import cvxopt as cvx
from cvxopt import solvers
import time, pdb
from copy import deepcopy
import traceback as tb

from ribohmm import utils
from ribohmm.utils import Mappability, States

from collections import namedtuple
import logging
logger = logging.getLogger('viterbi_log')

solvers.options['maxiters'] = 300
solvers.options['show_progress'] = False

# This represents the frame and triplet position of the ORF
# The triplet position is within the transcript, where the index starts at 0
CandidateCDS = namedtuple('CandidateCDS', 'frame start stop')

# Remove this once we're done with debugging RMSE
n_orfs_blank, n_orfs_total = [0, 0, 0, 0], [0, 0, 0, 0]

def logistic(x):
    return 1 / (1 + np.exp(x))


def nplog(x):
    return np.nan_to_num(np.log(x))


def normalize(arr):
    """Compute the log-sum-exp of a real-valued vector,
       avoiding numerical overflow issues.

    Arguments:

        x : numpy vector (float64)

    Returns:

        c : scalar (float64)

    """
    arr_max = np.max(arr)
    c = 0
    for val in arr:
        c += exp(val - arr_max)
    c = log(c) + arr_max
    return c

# cdef np.ndarray[np.float64_t, ndim=1] outsum(np.ndarray[np.float64_t, ndim=2] arr):
def outsum(arr):
    """Fast summation over the 0-th axis.
       Faster than numpy.sum()
    """

    # cdef np.ndarray thesum
    return sum([a for a in arr])


class Data:
    def __init__(self, riboseq_pileup, codon_map, transcript_normalization_factor, is_pos_mappable, seq=''):
        """Instantiates a data object for a transcript and populates it 
        with observed ribosome-profiling data, expression RPKM as its
        scaling factor, and the DNA sequence of each triplet in the 
        transcript in each frame and the missingness-type for each 
        triplet in the transcript in each frame.

        footprint_counts (riboseq_pileup, formerly obs)
        ======================
        List of size num_transcripts, where each element is an array of shape (len_transcript, num_footprint_lengths)
        This is the riboseq pileup at each position of a transcript. This is a list of transcripts, where each
        transcript is represented by an array of integers. The outer element is the position in the transcript, and the
        inner element is for each footprint length.

        codon_map
        ========
        A dictionary with three keys: kozak, start, and stop. The values for each of those keys is an array of size
        (num_codons, 3), where the 3 is for each possible reading frame.

        The start array maps to the list of start codons in utils.STARTCODONS, and the stop array does the same.
        ex:
        [
          [0 1 0],
          [0 0 0],
          [3 0 0]
        ]
        means that there is an AUG codon in the first triplet of the second open reading frame, and an GUG codon in the
        third triplet of the first open reading frame

        rna_counts (scale)
        ==================
        A scaling factor based on observed RNAseq counts. Is a scalar value that is transcript specific.

        obs: footprint counts
        codon_id: codon flags
        scale: rna_counts
        mappable: rna_mappability

        """

        # cdef double r,m,f
        # cdef np.ndarray ma, mapp

        # For backward compatibility
        obs, codon_id, scale, mappable = riboseq_pileup, codon_map, transcript_normalization_factor, is_pos_mappable

        self.seq = seq

        # Save raw riboseq pileup
        self.raw_riboseq_pileup = deepcopy(riboseq_pileup)
        # import pickle
        # with open('riboseq_lengths.pkl', 'wb') as out:
        #     pickle.dump(riboseq_pileup, out)

        # length of transcript
        self.L = self.transcript_length = obs.shape[0]
        # length of HMM
        # For transcript size  9 (remainder 0), result will be 2
        # For transcript size 10 (remainder 1), result will be 2
        # For transcript size 11 (remainder 2), result will be 2
        # For transcript size 12 (remainder 0), result will be 3
        self.M = self.n_triplets = int(self.L/3) - 1  # TODO Why is this -1 here?
        # number of distinct footprint lengths
        self.R = self.n_footprint_lengths = obs.shape[1]
        # observed ribosome-profiling data
        self.obs = self.riboseq_pileup = obs
        # transcript expression RPKM as scaling factor
        self.scale = self.transcript_normalization_factor = scale
        # mappability of each position for each footprint length
        self.mappable = self.is_pos_mappable = mappable
        # ensure that all unmappable positions have a zero footprint count
        self.riboseq_pileup[~self.is_pos_mappable] = 0
        # codon type of each triplet in each frame
        self.codon_id = self.codon_map = codon_id
        # missingness-type of each triplet in each frame for each footprint length
        self.missingness_type = np.zeros((self.n_footprint_lengths, 3, self.n_triplets), dtype=np.uint8)
        # total footprint count for each triplet in each frame for each footprint length
        self.total_pileup = np.empty((3, self.n_triplets, self.n_footprint_lengths), dtype=np.uint64)

        self.rmses = [list(), list(), list()]
        self.ssrmses = [list(), list(), list()]

        # Compute missingness type and total footprint count in each triplet
        for frame_i in range(3):
            for footprint_length_i in range(self.n_footprint_lengths):
                # Get mappability information and reshapes into a 2-dim array, where the number of rows is equal to
                # the number of triplets and the columns are for each frame index
                frame_sequence_positions = slice(frame_i, 3 * self.n_triplets + frame_i)
                per_triplet_mappability = (
                    self.is_pos_mappable[frame_sequence_positions, footprint_length_i]
                    .reshape(self.n_triplets, 3)
                )
                # missingness pattern of a triplet can belong to one of 8 types,
                # depending on which of the 3 positions are unmappable
                # Each triplet will be represented by an integer that corresponds to a value in utils.binarize
                # Each value ma in each iteration represents one triplet (so will be length 3)
                self.missingness_type[footprint_length_i, frame_i, :] = np.array([
                    utils.debinarize[triplet_mappability.tostring()]
                    for triplet_mappability in per_triplet_mappability
                ]).astype(np.uint8)

                # Gets the total pileup count from the riboseq data for each triplet in this frame
                self.total_pileup[frame_i, :, footprint_length_i] = (
                    np.sum(self.riboseq_pileup[frame_sequence_positions, footprint_length_i]
                           .reshape(self.n_triplets, 3), axis=1)
                )

    def compute_log_probability(self, emission):
        """Computes the log probability of the data given model parameters.
           Log probability of data is the sum of log probability at positions
           under the HMM and log probability at extra positions.

           Arguments:
               emission : instance of `Emission`
                          containing estimates of 
                          emission model parameters

        """
        # Create array to store log probability
        # It is 3-dimensional:
        #  - 1st dim is frame index, of size 3
        #  - 2nd dim is triplet index, of size n_triplets
        #  - 3rd dim is state index, of size emission.S, which is 9
        # In this case, log probability is the log of the likelihood that
        self.log_probability = np.zeros((3, self.n_triplets, emission['S']), dtype=np.float64)
        self.periodicity_model = np.zeros((3, self.n_triplets, emission['S']), dtype=np.float64)
        self.occupancy_model = np.zeros((3, self.n_triplets, emission['S']), dtype=np.float64)

        # Same as above TODO
        self.extra_log_probability = np.zeros((3,), dtype=np.float64)

        # missingness-types where 1 out of 3 positions are unmappable
        # This is based utils.binarize, the below indexes all have 1 position unmappable
        # Except that isn't true? That's what the comment from the original code says, but as far as I can
        # tell, this maps to the following, where False means unmappable:
        #  - [False, True,  True]
        #  - [True,  False, True]
        #  - [True,  True,  False]
        #  - [True,  True,  True]
        # Why is 7 included in this?
        # one_base_unmappable = np.array([3, 5, 6, 7]).reshape(4, 1)
        one_base_unmappable = np.array([
            Mappability.UMM,
            Mappability.MUM,
            Mappability.MMU,
            Mappability.MMM
        ]).reshape(4, 1)

        # loop over possible frames
        for frame_i in range(3):
            # loop over footprint lengths
            for footprint_length_i in range(self.n_footprint_lengths):
                # The local log probability for this frame index and footprint length index
                log_probability = np.zeros((self.n_triplets, emission['S']), dtype=np.float64)

                # Riboseq pileup for all the triplets in this frame index, then reshaped to where each row is a
                # triplet with three values
                frame_positions = slice(frame_i, 3 * self.n_triplets + frame_i)
                frame_pileups = self.riboseq_pileup[frame_positions, footprint_length_i].reshape(self.n_triplets, 3)

                # probability under periodicity model, accounting for mappability
                # triplets with at most 1 unmappable position
                # For each triplet in the transcript, for this footprint length and frame index, determine which
                # triplets match one of the missingness types that have one unmappable position
                triplet_one_base_unmappable = np.any(
                    self.missingness_type[footprint_length_i, frame_i, :] == one_base_unmappable,
                    axis=0
                )

                # Calculate the log probability of those triplets which have one unmappable position, for each
                # possible state
                log_probability[triplet_one_base_unmappable, :] = (
                    (
                        # Give total riboseq pileup for each triplet, for this footprint length and frame index, to
                        # the natural log of the gamma function (https://www.desmos.com/calculator/q4evkl4ekm)
                        gammaln(self.total_pileup[frame_i, triplet_one_base_unmappable, footprint_length_i] + 1)
                        # Base pileups for each triplet in this frame index, given to the natural log of the
                        # gamma function
                        - np.sum(gammaln(frame_pileups[triplet_one_base_unmappable, :] + 1), axis=1)
                        # Reshape to put each value in its own row, with 1 column
                    ).reshape(triplet_one_base_unmappable.sum(), 1)
                    # Inner product of the base pileup for each triplet in this frame index and the emission
                    # log periodicity for this footprint length
                    + np.dot(
                        frame_pileups[triplet_one_base_unmappable, :],
                        emission['logperiodicity'][footprint_length_i].T
                    )
                )

                # Iterate through each missingness type that has one unmappable position
                # Subtract some log probability from each triplet which has one unmappable position
                for mtype in one_base_unmappable[:3, 0]:
                    mapAB = self.missingness_type[footprint_length_i, frame_i, :] == mtype
                    log_probability[mapAB, :] -= np.dot(
                        self.total_pileup[frame_i, mapAB, footprint_length_i:footprint_length_i + 1],
                        utils.nplog(emission['rescale'][footprint_length_i:footprint_length_i + 1, :, mtype])
                    )

                # probability under occupancy model, accounting for mappability
                # The alpha and beta model parameters are arrays of size (n_footprint_lengths, n_states)
                # alpha and beta below are 1-D arrays of size n_states
                alpha = emission['rate_alpha'][footprint_length_i]
                beta = emission['rate_beta'][footprint_length_i]
                # The rescale variable ends up being an array of size n_triplets, where each values corresponds to
                # the rescale value for that triplet's missingness type
                rescale = emission['rescale'][footprint_length_i, :, self.missingness_type[footprint_length_i, frame_i, :]]
                # Total pileup for each triplet, where each value is a single-element array
                total = self.total_pileup[frame_i, :, footprint_length_i:footprint_length_i + 1]
                rate_log_probability = (
                    alpha * beta * utils.nplog(beta) +
                    gammaln(alpha*beta + total) -
                    gammaln(alpha*beta) -
                    gammaln(total + 1) +
                    total * utils.nplog(self.transcript_normalization_factor * rescale) -
                    (alpha * beta + total) * utils.nplog(beta + self.transcript_normalization_factor * rescale)
                )

                # ensure that triplets with all positions unmappable
                # do not contribute to the data probability
                mask = self.missingness_type[footprint_length_i, frame_i, :] == Mappability.UUU
                rate_log_probability[mask, :] = 0

                # Store the log probability
                self.periodicity_model[frame_i] += log_probability
                self.occupancy_model[frame_i] += rate_log_probability
                self.log_probability[frame_i] += log_probability + rate_log_probability

                # likelihood of extra positions in transcript
                # Compute likelihood for bases before the core sequence of triplets
                # TODO Consolidate this code, make it DRY
                for extra_base_pos in range(frame_i):
                    if self.is_pos_mappable[extra_base_pos, footprint_length_i]:
                        self.extra_log_probability[frame_i] += (
                            alpha[States.ST_5PRIME_UTS] * beta[States.ST_5PRIME_UTS] * utils.nplog(beta[States.ST_5PRIME_UTS]) -
                            (alpha[States.ST_5PRIME_UTS] * beta[States.ST_5PRIME_UTS] + self.riboseq_pileup[extra_base_pos, footprint_length_i]) *
                            utils.nplog(beta[States.ST_5PRIME_UTS] + self.transcript_normalization_factor / 3.) +
                            gammaln(alpha[States.ST_5PRIME_UTS] * beta[States.ST_5PRIME_UTS]+self.riboseq_pileup[extra_base_pos, footprint_length_i]) -
                            gammaln(alpha[States.ST_5PRIME_UTS] * beta[States.ST_5PRIME_UTS]) +
                            self.riboseq_pileup[extra_base_pos, footprint_length_i] * utils.nplog(self.transcript_normalization_factor / 3.) -
                            gammaln(self.riboseq_pileup[extra_base_pos, footprint_length_i] + 1)
                        )

                # Compute likelihood for bases after the core sequence of triplets
                for extra_base_pos in range(3 * self.n_triplets + frame_i, self.transcript_length):
                    if self.is_pos_mappable[extra_base_pos, footprint_length_i]:
                        self.extra_log_probability[frame_i] += (
                            alpha[States.ST_3PRIME_UTS] * beta[States.ST_3PRIME_UTS] * utils.nplog(beta[States.ST_3PRIME_UTS]) -
                            (alpha[States.ST_3PRIME_UTS] * beta[States.ST_3PRIME_UTS] + self.riboseq_pileup[extra_base_pos, footprint_length_i]) *
                            utils.nplog(beta[States.ST_3PRIME_UTS] + self.transcript_normalization_factor / 3.) +
                            gammaln(alpha[States.ST_3PRIME_UTS] * beta[States.ST_3PRIME_UTS] + self.riboseq_pileup[extra_base_pos, footprint_length_i]) -
                            gammaln(alpha[States.ST_3PRIME_UTS] * beta[States.ST_3PRIME_UTS]) +
                            self.riboseq_pileup[extra_base_pos, footprint_length_i] * utils.nplog(self.transcript_normalization_factor / 3.) -
                            gammaln(self.riboseq_pileup[extra_base_pos, footprint_length_i] + 1)
                        )

        # check for infs or nans in log likelihood
        if np.isnan(self.log_probability).any() or np.isinf(self.log_probability).any():
            print('Warning: Inf/Nan in data log likelihood')
            pdb.set_trace()

        if np.isnan(self.extra_log_probability).any() or np.isinf(self.extra_log_probability).any():
            print('Warning: Inf/Nan in extra log likelihood')
            pdb.set_trace()

    def compute_minimal_ORF(self, candidate_cds, buffer=0):
        """
        Args:
            candidate_cds: A CandidateCDS namedtuple

        Returns:

        """
        n_triplets = self.codon_map['start'].shape[0]
        state_seq = np.array(self.get_state_sequence(n_triplets, candidate_cds.start, candidate_cds.stop))
        try:
            start_minimal_orf = int(np.where(state_seq == States.ST_TIS)[0][0] - buffer)
        except:
            raise ValueError('Could not find start in compute_minimal_ORF()')
        try:
            end_minimal_orf = int(np.where(state_seq == States.ST_3PRIME_UTS_MINUS)[0][0] + buffer)
        except:
            # Where the last triplet is a stop codon
            if state_seq[-1] == States.ST_TTS:
                end_minimal_orf = int(np.where(state_seq == States.ST_TTS)[0][0] + buffer)
            else:
                raise ValueError('Could not find end in compute_minimal_ORF()')
        return start_minimal_orf, end_minimal_orf

    def get_minimal_ORF_overlapping_reads(self, candidate_cds):
        """
        Calculates the total reads that overlap the minimal ORF, for all footprint lengths
        Args:
            candidate_cds:

        Returns:

        """
        try:
            minimal_ORF_start, minimal_ORF_stop = self.compute_minimal_ORF(candidate_cds)
            return int(self.total_pileup[candidate_cds.frame, minimal_ORF_start:minimal_ORF_stop + 1].sum())
        except:
            return None


    def compute_minimal_ORF_log_probability(self):
        N_FRAMES = 3
        orf_state_matrix, candidate_cds_matrix = self.orf_state_matrix()
        # print('#### Size of candidate cds: {}'.format(sum([len(c) for c in candidate_cds_matrix])))

        # orf_periodicity_likelihoods = [list(), list(), list()]
        orf_periodicity_likelihoods = dict()
        # orf_occupancy_likelihoods = [list(), list(), list()]
        orf_occupancy_likelihoods = dict()

        for frame_i in range(N_FRAMES):
            n_orfs = orf_state_matrix[frame_i].shape[0]
            for orf_i in range(n_orfs):
                candidate_cds = candidate_cds_matrix[frame_i][orf_i]
                try:
                    start_minimal_orf = np.where(orf_state_matrix[frame_i][orf_i] == States.ST_TIS)[0][0]
                    end_minimal_orf = np.where(orf_state_matrix[frame_i][orf_i] == States.ST_3PRIME_UTS_MINUS)[0][0]
                except IndexError:
                    continue

                minimal_orf_state_sequence = orf_state_matrix[frame_i][orf_i][start_minimal_orf:end_minimal_orf + 1]
                # orf_periodicity = self.periodicity_model[frame_i, start_minimal_orf:end_minimal_orf + 1,
                #                                          minimal_orf_state_sequence]
                orf_periodicity = np.choose(minimal_orf_state_sequence, self.periodicity_model[frame_i, start_minimal_orf:end_minimal_orf + 1].T)
                # orf_occupancy = self.occupancy_model[frame_i, start_minimal_orf:end_minimal_orf + 1,
                #                                      minimal_orf_state_sequence]
                orf_occupancy = np.choose(minimal_orf_state_sequence, self.occupancy_model[frame_i, start_minimal_orf:end_minimal_orf + 1].T)
                # print('^^^^^^^^^ shape: {}'.format(orf_periodicity.shape))
                import base64
                # base64.b64encode(nparray)
                # print(orf_occupancy)
                # orf_periodicity_likelihoods[frame_i].append(np.sum(orf_periodicity))
                orf_periodicity_likelihoods[candidate_cds] = float(np.sum(orf_periodicity))
                # orf_occupancy_likelihoods[frame_i].append(np.sum(orf_occupancy))
                orf_occupancy_likelihoods[candidate_cds] = float(np.sum(orf_occupancy))

        return orf_periodicity_likelihoods, orf_occupancy_likelihoods


    def compute_observed_pileup_deviation(self, emission, return_sorted=True, normalize_tes=False, transcript_obj=None):
        """
        For each ORF, for each read length, for the first two base positions in each triplet, computes a
        difference between observed and expected pileup

        The start codon to stop codon is called the ORF. Including all the UTR regions is the full transcript.

        RMSE includes only the ORF. That's from the TIS state to the 3'UTS- state, inclusive.
        ssRMSE includes only the ORF, but also takes the average of all TES states as a single TES entry.

        :param candidate_orfs:
        :param emission:
        :return:
        """
        # Identify only triplets which are fully mappable to calculate RMSE
        # fully_mappable_triplets [each frame, each footprint length, each triplet]
        N_FRAMES = 3
        fully_mappable_triplets = np.zeros((N_FRAMES, self.n_footprint_lengths, self.n_triplets))
        for frame_i in range(N_FRAMES):
            for footprint_length_i in range(self.n_footprint_lengths):
                for i in range(self.n_triplets):
                    fully_mappable_triplets[frame_i, footprint_length_i, i] = np.all(
                        self.is_pos_mappable[(i * 3) + frame_i: (i * 3) + 3 + frame_i, footprint_length_i]
                    )

        # Get all the ORFs for this transcript
        orfs_with_errors = list()
        # for candidate_orf in self.get_candidate_cds_simple():
        _, all_candidate_cds = self.orf_state_matrix()
        all_candidate_cds = all_candidate_cds[0] + all_candidate_cds[1] + all_candidate_cds[2]
        # Iterate through all ORFs
        n_blank, n_total = 0, 0
        ltfm, not_ltfm = list(), list()
        for candidate_orf in all_candidate_cds:
            # footprint_errors_with_utr = list()  # Including all the UTR
            footprint_errors_only_orf = list()  # Only from start to stop, inclusive
            # by_triplet_error_with_utr = dict()
            by_triplet_error_only_orf = dict()
            # Get the index of the start and stop triplet
            orf_start_triplet, orf_stop_triplet = self.compute_minimal_ORF(candidate_orf)
            orf_size = orf_stop_triplet - orf_start_triplet + 1  # TODO Is this the correct formula?
            triplets_dropped_for_mappability = list()
            ORF_pileups = dict()
            state_diagram = [2, 3] + [4] * (orf_size - 5) + [5, 6, 7]

            # Iterate over each footprint length in each ORF
            for footprint_length_i in range(self.n_footprint_lengths):
                n_total += 1
                # print('&&&&&&&&&&&&&&&&&&&&&&&&&')
                # expected shape is [9, 3]. 1st dim is the state, 2nd dim is the three base positions
                expected = emission['logperiodicity'][footprint_length_i]
                observed_frame_i = candidate_orf.frame
                observed_start = candidate_orf.start
                observed_stop = candidate_orf.stop

                # Get the pileup matrix, which will be 2 dimensional [n_triplets, 3]
                pileups = self.raw_riboseq_pileup[observed_frame_i:, footprint_length_i]
                # Chop base pairs off the end until we are divisible by 3
                # For transcript size 9 (remainder 0), frame 0, result will be 3 triplets
                # For transcript size 9 (remainder 0), frame 1, result will be 2 triplets
                # For transcript size 9 (remainder 0), frame 2, result will be 2 triplets
                # For transcript size 10 (remainder 1), frame 0, result will be 3 triplets
                # For transcript size 10 (remainder 1), frame 1, result will be 3 triplets
                # For transcript size 10 (remainder 1), frame 1, result will be 2 triplets
                # For transcript size 11 (remainder 2), frame 0, result will be 3 triplets
                # For transcript size 11 (remainder 2), frame 1, result will be 3 triplets
                # For transcript size 11 (remainder 2), frame 2, result will be 3 triplets
                while len(pileups) % 3 in {1, 2}:
                    pileups = pileups[:-1]
                # Reshape into desired form [n_triplets, 3]
                pileups = pileups.reshape(-1, 3)

                # Pull out just the ORF from both pileups and expected emissions
                # Ensure that the number of triplets in the pileup never exceeds self.n_triplets
                pileups = pileups[:self.n_triplets]
                pileups = pileups[observed_start:observed_stop + 1]
                # expected = expected[observed_start:observed_stop + 1]
                expected = np.exp(expected[state_diagram])

                # Remove any triplets with 1 or more unmappable positions
                orf_mappability = fully_mappable_triplets[observed_frame_i, footprint_length_i,
                                                          observed_start:min(observed_stop + 1, pileups.shape[0] - 1)].astype(bool)
                orf_mappability = fully_mappable_triplets[observed_frame_i, footprint_length_i,
                                                          observed_start:observed_stop + 1].astype(bool)
                if footprint_length_i == 0:
                    if not np.any(orf_mappability):
                        ltfm.append(orf_size)
                    else:
                        not_ltfm.append(orf_size)
                n_orfs_total[footprint_length_i] += 1
                pre_mappability_count = pileups.shape[0]
                pileups = pileups[orf_mappability]
                expected = expected[orf_mappability]
                post_mappability_count = pileups.shape[0]

                # Find pileup proportions
                pileups_proportions = np.nan_to_num(
                    pileups / np.tile(pileups.sum(axis=1), (3, 1)).transpose(),
                    nan=1/3
                )

                # Find squared error
                if not np.any(orf_mappability):
                    n_blank += 1
                    n_orfs_blank[footprint_length_i] += 1
                squared_error = (pileups_proportions - expected) ** 2

                # Add error sum of base pair in each triplet
                # Shape goes from (n_triplets, 3) to (n_triplets,)
                try:
                    by_triplet_error_only_orf[footprint_length_i] = np.sum(squared_error, axis=1)
                except:
                    by_triplet_error_only_orf[footprint_length_i] = -1

                # Add ORF pileups
                ORF_pileups[footprint_length_i] = np.sum(pileups)

                # If normalizing TES, find mean of all TES
                if normalize_tes:
                    tes_states = squared_error[2:-3]
                    tes_mean_squared_error = np.mean(tes_states, axis=0)
                    squared_error = np.concatenate([
                        squared_error[:2],
                        tes_mean_squared_error.reshape(1, -1),
                        squared_error[-3:]
                    ])

                # Find RMSE for this footprint length
                rmse = np.sqrt(np.mean(squared_error))
                footprint_errors_only_orf.append(rmse)

            # Take the mean of all RMSE over all footprints lengths
            orf_error_only_orf = np.mean(footprint_errors_only_orf)

            if normalize_tes:
                self.ssrmses[candidate_orf.frame].append(orf_error_only_orf)
            else:
                self.rmses[candidate_orf.frame].append(orf_error_only_orf)

            orfs_with_errors.append((
                candidate_orf,
                orf_error_only_orf,
                by_triplet_error_only_orf,
                triplets_dropped_for_mappability,
                ORF_pileups,
                # n_blank,
                # n_total
            ))

        # Print out transcript level report
        from collections import Counter
        print('===============')
        print('Transcript id: {}'.format(transcript_obj.id))
        print('Normalize TES: {}'.format(normalize_tes))
        print('n total RMSE calculations: {}'.format(n_total))
        print('N ORFs: {}'.format(len(all_candidate_cds)))
        print('n ORFs that were all less than fully mappable (LTFM): {}'.format(len(ltfm)))
        print('mean ORF size for LTFM: {}'.format(np.mean(ltfm)))
        print('median ORF size for LTFM: {}'.format(np.median(ltfm)))
        print('mean ORF size for NOT LTFM: {}'.format(np.mean(not_ltfm)))
        print('median ORF size for NOT LTFM: {}'.format(np.median(not_ltfm)))
        print('distribution for LTFM: {}'.format(Counter(ltfm).most_common()))
        print('\tORF Size: N Occurrences')
        for val, count in Counter(ltfm).most_common():
            print('\t{}: {}'.format(val, count))
        print('===============\n')

        if not return_sorted:
            return orfs_with_errors
        return sorted(orfs_with_errors, key=lambda r: r[1])


    @staticmethod
    def get_all_ORFs(seq):
        """
        This method does not use the codon map, rather it looks in the raw sequence.

        Returns:
        """
        STARTCODONS = {
            'AUG', 'CUG', 'GUG', 'UUG', 'AAG',
            'ACG', 'AGG', 'AUA', 'AUC', 'AUU'
        }
        STOPCODONS = {'UAA', 'UAG', 'UGA'}
        N_FRAMES = 3
        # n_triplets = local_start_codon_map.shape[0]
        # n_triplets = len(seq)
        candidate_cds = list()

        def codon(pos_i, seq):
            return seq[pos_i * 3:(pos_i + 1) * 3]

        for frame_i in range(N_FRAMES):
            frame_seq = seq[frame_i:]
            n_triplets = int(len(frame_seq) / 3)
            for pos_i in range(n_triplets):
                if codon(pos_i, frame_seq) in STARTCODONS:
                    for stop_i in range(pos_i + 1, n_triplets):
                        if codon(stop_i, frame_seq) in STOPCODONS:
                            candidate_cds.append(CandidateCDS(
                                frame=frame_i,
                                start=pos_i,
                                stop=stop_i
                            ))
                            break

        return candidate_cds


    def get_candidate_cds_simple(self, shifted_forward=False):
        """

        Args:
            shifted_forward: If True, shifts forward the codon map to directly represent the codon position

        Returns:

        """
        # return self.get_all_ORFs(self.seq)
        # local_start_codon_map = self.codon_map['discovery_start'].copy()
        local_start_codon_map = self.codon_map.get('discovery_start', self.codon_map['start']).copy()
        local_stop_codon_map = self.codon_map.get('discovery_stop', self.codon_map['stop']).copy()
        # local_stop_codon_map = self.codon_map['discovery_stop'].copy()

        if shifted_forward:
            local_start_codon_map = np.roll(local_start_codon_map, shift=1, axis=0)
            local_start_codon_map[0] = [0, 0, 0]  # np.roll wraps around, so set the first codon to 0s
            local_stop_codon_map = np.roll(local_stop_codon_map, shift=2, axis=0)
            local_stop_codon_map[0] = [0, 0, 0]
            local_stop_codon_map[1] = [0, 0, 0]

        N_FRAMES = 3
        n_triplets = local_start_codon_map.shape[0]
        candidate_cds = list()

        for pos_i in range(n_triplets):
            for frame_i in range(N_FRAMES):
                if local_start_codon_map[pos_i, frame_i] > 0:
                    for stop_i in range(pos_i, n_triplets):
                        if local_stop_codon_map[stop_i, frame_i] > 0:
                            if stop_i - pos_i >= 5:
                                candidate_cds.append(CandidateCDS(
                                    frame=frame_i,
                                    start=pos_i,
                                    stop=stop_i
                                ))
                            break

        return candidate_cds

    def get_state_sequence(self, n_triplets, start, stop):
        seq = np.zeros(n_triplets, dtype=int)
        try:
            # Setting stop codon to TTS
            # seq[start - 1] = 1
            # seq[start] = 2
            # seq[start + 1] = 3
            # seq[start + 2:stop - 1] = 4
            # seq[stop - 1] = 5
            # seq[stop] = 6  # TTS
            # seq[stop + 1] = 7
            # seq[stop + 2:] = 8

            # Setting stop codon to 3'UTS-
            seq[start - 1] = 1
            seq[start] = 2
            seq[start + 1] = 3
            seq[start + 2:stop - 2] = 4
            seq[stop - 2] = 5
            seq[stop - 1] = 6
            seq[stop] = 7  # 3'UTS-
            seq[stop + 1:] = 8
        except:
            pass  # Silently fail
        return list(seq)

    def orf_state_matrix(self):
        """
        Returns () matrix of orf states,

        The ORF state matrix is size 3, one element for each frame. Each of those elements is a list, with each
        state sequence for that frame.
        """
        n_triplets = self.codon_map['start'].shape[0]
        orf_state_matrix_ = [list(), list(), list()]
        candidate_cds_ = [list(), list(), list()]

        for candidate_cds in self.get_candidate_cds_simple(shifted_forward=False):
            state_seq = self.get_state_sequence(n_triplets, candidate_cds.start, candidate_cds.stop)
            if state_seq[0] == 0:
                orf_state_matrix_[candidate_cds.frame].append(state_seq)
                candidate_cds_[candidate_cds.frame].append(candidate_cds)

        return [np.array(m) for m in orf_state_matrix_], candidate_cds_


class Frame(object):
    
    def __init__(self):
        """Instantiates a frame object for a transcript and initializes
        a random posterior probability over all three frames.
        """

        self.posterior = np.random.rand(3)
        """Why on Earth did I add a int() here? It isn't in the original code"""
        # self.posterior = int(self.posterior/self.posterior.sum())
        self.posterior = self.posterior/self.posterior.sum()

    def update(self, data, state):
        """Update posterior probability over the three
        frames for a transcript.

        Arguments:
            data : instance of `Datum`

            state : instance of `State`

        """

        # Sum up the likelihoods for each frame in state.likelihood, then add in data.extra_log_probability
        self.posterior = outsum(state.likelihood) + data.extra_log_probability

        # Perform softmax on the posteriors, so they can be interpreted as probabilities
        self.posterior = self.posterior - self.posterior.max()
        self.posterior = np.exp(self.posterior)
        self.posterior = self.posterior / self.posterior.sum()

        # print(f'Result: {outsum(state.likelihood)} | {data.extra_log_probability} | {outsum(state.likelihood) + data.extra_log_probability} |{self.posterior}')

    def __reduce__(self):
        return (rebuild_Frame, (self.posterior,))

def rebuild_Frame(pos):
    f = Frame()
    f.posterior = pos
    return f


class State(object):
    
    def __init__(self, n_triplets):

        # number of triplets
        self.M = self.n_triplets = n_triplets
        # number of states for the HMM
        self.S = self.n_states = 9
        # stores the (start,stop) and posterior for the MAP state for each frame
        self.best_start = []
        self.best_stop = []
        self.max_posterior = np.empty(shape=(3,), dtype=np.float64)

    def _forward_update(self, data, transition):
        """
        Inflate serialized transition dictionary
        """
        # Define one for each of the 9 States
        logprior = nplog([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # swapidx = np.array([2, 3, 6, 7]).astype(np.uint8)
        swapidx = np.array(
            [States.ST_TIS, States.ST_TIS_PLUS, States.ST_TTS, States.ST_3PRIME_UTS_MINUS]
        ).astype(np.uint8)
        # alpha_k(z_k) is P(z_k | x_1:k)
        # In this case the three frames are three separate HMMs
        self.alpha = np.zeros((3, self.n_triplets, self.n_states), dtype=np.float64)
        self.likelihood = np.zeros((self.n_triplets, 3), dtype=np.float64)

        # data.codon_map['kozak'] is shape (n_triplets, n_frames) and is the kozak values for each triplet in
        # each frame
        # transition['seqparam']['start'] is an 11-element array, one value for each defined Start codon, where 0 is
        # defined as not a Start codon
        # data.codon_map['start'] is an integer value for each triplet and frame that corresponds to a Start codon, or
        # 0 if not any of the Start codons
        # transition['seqparam']['start'][data.codon_map['start']]) is the broadcast of each Start codon seqparam value
        # to the appropriate codon type
        # The shape of P will be (n_triplets, n_frames)
        # Q is essentially the same thing but with Stop codons
        P = logistic(
            -1 * (transition['seqparam']['kozak'] * data.codon_map['kozak'] +
                  transition['seqparam']['start'][data.codon_map['start']])
        )
        Q = logistic(-1 * transition['seqparam']['stop'][data.codon_map['stop']])

        for frame_i in range(3):
            # Determine the likelihood and alpha for the first triplet
            # data.log_probability[frame_i, triplet_i, state_i]
            # Both logprior and the log_probabilty are size (9,)
            newalpha = logprior + data.log_probability[frame_i, 0, :]
            normalized_new_alpha = normalize(newalpha)
            # Set the likelihood and alpha for each state for the first triplet
            for state_i in range(self.n_states):
                self.alpha[frame_i, 0, state_i] = newalpha[state_i] - normalized_new_alpha
            self.likelihood[0, frame_i] = normalized_new_alpha

            # For all triplets after the first triplet
            for triplet_i in range(1, self.n_triplets):

                # states 2,3,6,7
                # TIS, TIS+, TTS, 3'UTS-
                for swap_state_i in swapidx:
                    newalpha[swap_state_i] = (
                        self.alpha[frame_i, triplet_i - 1, swap_state_i - 1]
                        + data.log_probability[frame_i, triplet_i, swap_state_i]
                    )

                # state 0,1
                # 5'UTS, 5'UTS+
                try:
                    # Get the alpha value from the previous triplet, state 5'UTS
                    p = self.alpha[frame_i, triplet_i - 1, States.ST_5PRIME_UTS] + log(1 - P[triplet_i, frame_i])
                    q = self.alpha[frame_i, triplet_i - 1, States.ST_5PRIME_UTS] + log(P[triplet_i, frame_i])
                except ValueError:  # log(x) where x <= 0
                    if P[triplet_i, frame_i] == 0.0:
                        p = self.alpha[frame_i, triplet_i - 1, States.ST_5PRIME_UTS]
                        q = utils.MIN
                    else:
                        p = utils.MIN
                        q = self.alpha[frame_i, triplet_i - 1, States.ST_5PRIME_UTS]
                newalpha[States.ST_5PRIME_UTS] = p + data.log_probability[frame_i, triplet_i, States.ST_5PRIME_UTS]
                newalpha[States.ST_5PRIME_UTS_PLUS] = q + data.log_probability[frame_i, triplet_i, States.ST_5PRIME_UTS_PLUS]

                # state 4
                # TES
                p = self.alpha[frame_i, triplet_i - 1, States.ST_TIS_PLUS]
                try:
                    q = self.alpha[frame_i, triplet_i - 1, States.ST_TES] + log(1 - Q[triplet_i, frame_i])
                except ValueError:
                    q = utils.MIN
                if p > q:
                    newalpha[States.ST_TES] = log(1 + exp(q - p)) + p + data.log_probability[frame_i, triplet_i, States.ST_TES]
                else:
                    newalpha[States.ST_TES] = log(1 + exp(p - q)) + q + data.log_probability[frame_i, triplet_i, States.ST_TES]

                # state 5
                try:
                    newalpha[States.ST_TTS_MINUS] = self.alpha[frame_i, triplet_i - 1, States.ST_TES] + log(Q[triplet_i, frame_i]) + data.log_probability[frame_i, triplet_i, States.ST_TTS_MINUS]
                except ValueError:
                    newalpha[States.ST_TTS_MINUS] = utils.MIN

                # state 8
                p = self.alpha[frame_i, triplet_i - 1, States.ST_3PRIME_UTS_MINUS]
                q = self.alpha[frame_i, triplet_i - 1, States.ST_3PRIME_UTS]
                if p > q:
                    newalpha[States.ST_3PRIME_UTS] = log(1 + exp(q - p)) + p + data.log_probability[frame_i, triplet_i, States.ST_3PRIME_UTS]
                else:
                    newalpha[States.ST_3PRIME_UTS] = log(1 + exp(p - q)) + q + data.log_probability[frame_i, triplet_i, States.ST_3PRIME_UTS]

                normalized_new_alpha = normalize(newalpha)
                # for s from 0 <= s < self.S:
                for s in range(self.n_states):
                    self.alpha[frame_i, triplet_i, s] = newalpha[s] - normalized_new_alpha

                self.likelihood[triplet_i, frame_i] = normalized_new_alpha

        if np.isnan(self.alpha).any() or np.isinf(self.alpha).any():
            print('Warning: Inf/Nan in forward update step')
            pdb.set_trace()

    def _reverse_update(self, data, transition):
        swapidx = np.array([1, 2, 3, 5, 6, 7]).astype(np.uint8)
        self.pos_first_moment = np.empty((3, self.n_triplets, self.n_states), dtype=np.float64)
        self.pos_cross_moment_start = np.empty((3, self.n_triplets, 2), dtype=np.float64)

        P = logistic(
            -1 * (transition.seqparam['kozak'] * data.codon_map['kozak'] +
                  transition.seqparam['start'][data.codon_map['start']])
        )
        Q = logistic(-1 * transition.seqparam['stop'][data.codon_map['stop']])

        # for f from 0 <= f < 3:
        for f in range(3):

            self.pos_first_moment[f, self.n_triplets - 1, :] = np.exp(self.alpha[f, self.n_triplets - 1, :])
            newbeta = np.empty((self.n_states,), dtype=np.float64)
            beta = np.zeros((self.n_states,), dtype=np.float64)

            for m in range(self.n_triplets - 2, -1, -1):

                for s in range(self.n_states):
                    beta[s] = beta[s] + data.log_probability[f, m + 1, s]

                try:
                    pp = beta[0] + log(1 - P[m + 1, f])
                except ValueError:
                    pp = utils.MIN
                try:
                    p = beta[1] + log(P[m + 1, f])
                except ValueError:
                    p = utils.MIN
                try:
                    q = beta[5] + log(Q[m + 1, f])
                except ValueError:
                    q = utils.MIN
                try:
                    qq = beta[4] + log(1 - Q[m + 1, f])
                except ValueError:
                    qq = utils.MIN

                # pos cross moment at start
                a = self.alpha[f, m, 0] - self.likelihood[m + 1, f]
                self.pos_cross_moment_start[f, m + 1, 0] = exp(a + p)
                self.pos_cross_moment_start[f, m + 1, 1] = exp(a + pp)
    
                # states 1,2,3,5,6,7
                for s in swapidx:
                    newbeta[s] = beta[s + 1]
                newbeta[self.n_states - 1] = beta[self.n_states - 1]

                # state 0
                if p > pp:
                    newbeta[0] = log(1 + np.exp(pp - p)) + p
                else:
                    newbeta[0] = log(1 + np.exp(p - pp)) + pp

                # state 4
                if qq > q:
                    newbeta[4] = log(1 + np.exp(q - qq)) + qq
                else:
                    newbeta[4] = log(1 + np.exp(qq - q)) + q

                # for s from 0 <= s < self.S:
                for s in range(self.n_states):
                    beta[s] = newbeta[s] - self.likelihood[m + 1, f]
                    self.pos_first_moment[f, m, s] = exp(self.alpha[f, m, s] + beta[s])

            self.pos_cross_moment_start[f, 0, 0] = 0
            self.pos_cross_moment_start[f, 0, 1] = 0

        if np.isnan(self.pos_first_moment).any() or np.isinf(self.pos_first_moment).any():
            print('Warning: Inf/Nan in first moment')
            pdb.set_trace()

        if np.isnan(self.pos_cross_moment_start).any() or np.isinf(self.pos_cross_moment_start).any():
            print('Warning: Inf/Nan in start cross moment')
            pdb.set_trace()

    def discovery_decode(self, data, transition, transcript, use_minimal_orf=False):
        P = logistic(-1 * (transition['seqparam']['kozak'] * data.codon_map['kozak']
                           + transition['seqparam']['start'][data.codon_map['start']]))
        Q = logistic(-1 * transition['seqparam']['stop'][data.codon_map['stop']])

        N_FRAMES = 3
        # These two matrices are the same dimensions
        orf_state_matrix, candidate_cds_matrix = data.orf_state_matrix()
        orf_posteriors = list()

        for frame_i in range(N_FRAMES):
            n_orfs = orf_state_matrix[frame_i].shape[0]
            orf_posteriors.append(np.zeros(shape=n_orfs))
            for orf_i in range(n_orfs):
                candidate_cds = candidate_cds_matrix[frame_i][orf_i]
                orf_start = candidate_cds.start * 3 + candidate_cds.frame,
                orf_stop = candidate_cds.stop * 3 + candidate_cds.frame
                tis = orf_start  # This is base position, not a state position
                tts = orf_stop
                if transcript.strand == '+':
                    cdstart = transcript.start + np.where(transcript.mask)[0][tis]
                    cdstop = transcript.start + np.where(transcript.mask)[0][tts]
                else:
                    cdstart = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tts]
                    cdstop = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tis]

                # TODO Maybe surround this in a try-except
                try:
                    start_minimal_orf = np.where(orf_state_matrix[frame_i][orf_i] == States.ST_5PRIME_UTS_PLUS)[0][0] - 1
                    end_minimal_orf = np.where(orf_state_matrix[frame_i][orf_i] == States.ST_3PRIME_UTS_MINUS)[0][0] + 1
                except IndexError:
                    continue

                # TODO Need to determine where the first good 0 state is
                # Set initial alpha
                if not use_minimal_orf or orf_state_matrix[frame_i][orf_i, 1] == States.ST_5PRIME_UTS_PLUS:
                    alpha = utils.nplog(1) + data.log_probability[frame_i, 0, 0]
                else:
                    alpha = 0

                # Go through the rest of the states
                for triplet_i in range(1, orf_state_matrix[frame_i].shape[1]):
                    current_state = orf_state_matrix[frame_i][orf_i, triplet_i]
                    prev_state = orf_state_matrix[frame_i][orf_i, triplet_i - 1]
                    # next_state = orf_state_matrix[frame_i][orf_i, triplet_i + 1]  # TODO This may be out of bounds

                    if use_minimal_orf:
                        if triplet_i < start_minimal_orf or triplet_i > end_minimal_orf:
                            # print('Outside minimal ORF')
                            continue
                    if current_state == 0:
                        # print('current state is 0')
                        try:
                            newalpha = alpha + log(1 - P[triplet_i, frame_i])  # What do we do when this is log(0)?
                        except:
                            newalpha = utils.MIN
                    elif current_state == 1:
                        try:
                            newalpha = alpha + log(P[triplet_i, frame_i])
                        except:
                            newalpha = utils.MIN
                    elif current_state == 2:
                        newalpha = alpha + log(1)
                    elif current_state == 3:
                        newalpha = alpha + log(1)
                    elif current_state == 4:
                        if prev_state == 3:
                            newalpha = alpha + log(1)
                        else:
                            try:
                                newalpha = alpha + log(1 - Q[triplet_i, frame_i])
                            except:
                                newalpha = utils.MIN
                    elif current_state == 5:
                        try:
                            newalpha = alpha + log(Q[triplet_i, frame_i])
                        except:
                            newalpha = utils.MIN
                    elif current_state == 6:
                        newalpha = alpha + log(1)
                    elif current_state == 7:
                        newalpha = alpha + log(1)
                    else:  # current_state == 8
                        newalpha = alpha + log(1)  # Is it deterministic?

                    alpha = newalpha + data.log_probability[frame_i, triplet_i, current_state]  # Last element is the state we're on?

                if use_minimal_orf:
                    orf_posteriors[frame_i][orf_i] = np.exp(alpha - np.sum(self.likelihood[start_minimal_orf:end_minimal_orf + 1, frame_i]))
                else:
                    orf_posteriors[frame_i][orf_i] = np.exp(alpha - np.sum(self.likelihood[:, frame_i]))

        return orf_posteriors, candidate_cds_matrix

    def decode(self, data, transition):
        P = logistic(-1*(transition['seqparam']['kozak'] * data.codon_map['kozak']
            + transition['seqparam']['start'][data.codon_map['start']]))
        Q = logistic(-1*transition['seqparam']['stop'][data.codon_map['stop']])

        logprior = utils.nplog([1, 0, 0, 0, 0, 0, 0, 0, 0])
        swapidx = np.array(
            [States.ST_TIS, States.ST_TIS_PLUS, States.ST_TTS, States.ST_3PRIME_UTS_MINUS]
        ).astype(np.uint8)
        pointer = np.zeros((self.n_triplets, self.n_states), dtype=np.uint8)
        pointer[0, 0] = np.array([0])
        alpha = np.zeros((self.n_states,), dtype=np.float64)
        self.decode_alphas = np.zeros((3, self.n_states), dtype=np.float64)
        newalpha = np.zeros((self.n_states,), dtype=np.float64)
        # Most likely hidden state for each triplet
        state = np.zeros((self.n_triplets,), dtype=np.uint8)

        # Iterate over each frame
        for frame_i in range(3):

            # find the state sequence with highest posterior
            # data.log_probability is (frame_i, triplet_i, state_i)
            # This frame, first triplet, all states
            alpha = logprior + data.log_probability[frame_i, 0, :]

            for triplet_i in range(1, self.n_triplets):

                # states 2,3,6,7
                for s in swapidx:
                    newalpha[s] = alpha[s - 1]
                    pointer[triplet_i, s] = s - 1

                # state 0,1
                try:
                    p = alpha[0] + log(1 - P[triplet_i, frame_i])
                    q = alpha[0] + log(P[triplet_i, frame_i])
                except ValueError:
                    if P[triplet_i, frame_i] == 0.0:
                        p = alpha[0]
                        q = utils.MIN
                    else:
                        p = utils.MIN
                        q = alpha[0] + log(1)
                pointer[triplet_i, 0] = 0
                newalpha[0] = p
                pointer[triplet_i, 1] = 0
                newalpha[1] = q

                # state 4
                p = alpha[3]
                try:
                    q = alpha[4] + log(1-Q[triplet_i, frame_i])
                except ValueError:
                    q = utils.MIN
                if p >= q:
                    newalpha[4] = p
                    pointer[triplet_i, 4] = 3
                else:
                    newalpha[4] = q
                    pointer[triplet_i, 4] = 4

                # state 5
                try:
                    newalpha[5] = alpha[4] + log(Q[triplet_i, frame_i])
                except ValueError:
                    newalpha[5] = utils.MIN
                pointer[triplet_i, 5] = 4

                # state 8
                p = alpha[7]
                q = alpha[8]
                if p >= q:
                    newalpha[8] = p
                    pointer[triplet_i, 8] = 7
                else:
                    newalpha[8] = q
                    pointer[triplet_i, 8] = 8

                # for s from 0 <= s < self.n_states:
                for s in range(self.n_states):
                    alpha[s] = newalpha[s] + data.log_probability[frame_i, triplet_i, s]

            self.decode_alphas[frame_i] = alpha

            # constructing the MAP state sequence
            # alpha is 1-dim array of size n_states
            state[self.n_triplets - 1] = np.argmax(alpha)
            # Start on the second-to-last triplet, then count backward to the first triplet
            for triplet_i in range(self.n_triplets - 2, 0, -1):
                # Previous state in the sense that it was the last calculated state
                # Since we're going backward, this is technically the next state in the sequence
                prev_state = state[triplet_i + 1]

                # Get the most likely state of this triplet
                state[triplet_i] = pointer[triplet_i + 1, prev_state]
            state[0] = pointer[0, 0]

            # Calculate the max posterior using the HMM definition
            # max_post = 0
            # for triplet_i, state_ in enumerate(state):
            #     max_post += data.log_probability[frame_i, triplet_i, state_]
            # max_post = np.exp(max_post)
            self.max_posterior[frame_i] = np.exp(np.max(alpha) - np.sum(self.likelihood[:, frame_i]))

            # identifying start codon position
            # state is 1-dim array of size n_triplets
            # This gives the position in the transcript, not the triplet index
            # Returns the most upstream possible start codon
            try:
                self.best_start.append(np.where(state == States.ST_TIS)[0][0] * 3 + frame_i)
            except IndexError:
                self.best_start.append(None)

            # identifying stop codon position
            try:
                # Should this be TTS?
                self.best_stop.append(np.where(state == States.ST_3PRIME_UTS_MINUS)[0][0] * 3 + frame_i)
            except IndexError:
                self.best_stop.append(None)

        # To allow us to extract out this function's alpha value
        self.alpha = np.empty((1, 1, 1), dtype=np.float64)
        self.pos_cross_moment_start = np.empty((1, 1, 1), dtype=np.float64)
        self.pos_cross_moment_stop = np.empty((1, 1, 1), dtype=np.float64)
        self.pos_first_moment = np.empty((1, 1, 1), dtype=np.float64)
        self.likelihood = np.empty((1, 1), dtype=np.float64)

    def joint_probability(self, data, transition, state, frame):

        # cdef long m
        # cdef double p, q, joint_probability

        joint_probability = data.log_probability[frame,0,state[0]]

        # for m from 1 <= m < self.M:
        for m in range(self.n_triplets):

            if state[m-1]==0:
    
                p = transition.seqparam['kozak'] * data.codon_map['kozak'][m,frame] \
                    + transition.seqparam['start'][data.codon_map['start'][m,frame]]
                try:
                    joint_probability = joint_probability - log(1+exp(-p))
                    if state[m]==0:
                        joint_probability = joint_probability - p
                except OverflowError:
                    if state[m]==1:
                        joint_probability = joint_probability - p

            elif state[m-1]==4:

                q = transition.seqparam['stop'][data.codon_id['stop'][m,frame]]
                try:
                    joint_probability = joint_probability - log(1+exp(-q))
                    if state[m]==4:
                        joint_probability = joint_probability - q
                except OverflowError:
                    if state[m]==5:
                        joint_probability = joint_probability - q

            joint_probability = joint_probability + data.log_probability[frame,m,state[m]]

        return joint_probability

    def compute_posterior(self, data, transition, start, stop):

        # cdef long frame
        # cdef double joint_prob, marginal_prob, posterior
        # cdef np.ndarray state

        frame = start%3
        start = int((start-frame)/3)
        stop = int((stop-frame)/3)

        # construct state sequence given a start/stop pair
        state = np.empty((self.n_triplets,), dtype=np.uint8)
        state[:start-1] = 0
        state[start-1] = 1
        state[start] = 2
        state[start+1] = 3
        state[start+2:stop-2] = 4
        state[stop-2] = 5
        state[stop-1] = 6
        state[stop] = 7
        state[stop+1:] = 8

        # compute joint probability
        joint_prob = self.joint_probability(data, transition, state, frame)
        # compute marginal probability
        marginal_prob = np.sum(self.likelihood[:,frame])
        posterior = exp(joint_prob - marginal_prob)

        return posterior

    def __reduce__(self):
        return (rebuild_State, (self.best_start, self.best_stop, self.max_posterior, self.M))

def rebuild_State(bstart, bstop, mposterior, M):
    s = State(M)
    s.best_start = bstart
    s.best_stop = bstop
    s.max_posterior = mposterior
    return s

class Transition(object):

    def __init__(self):
        """Order of the states is
        '5UTS','5UTS+','TIS','TIS+','TES','TTS-','TTS','3UTS-','3UTS'
        """

        # number of states in HMM
        self.S = 9
        self.restrict = True
        self.C = len(utils.STARTCODONS) + 1

        self.seqparam = dict()
        # initialize parameters for translation initiation
        self.seqparam['kozak'] = np.random.rand()
        self.seqparam['start'] = np.zeros((self.C,), dtype='float')
        self.seqparam['start'][0] = utils.MIN
        self.seqparam['start'][1] = 1+np.random.rand()
        self.seqparam['start'][2:] = utils.MIN

        # initialize parameters for translation termination
        self.seqparam['stop'] = utils.MAX * np.ones((4,), dtype=np.float64)
        self.seqparam['stop'][0] = utils.MIN

    def __getitem__(self, item):
        return self.__dict__[item]

    def _serialize(self):
        return {
            'seqparam': {
                'kozak': self.seqparam['kozak'],
                'start': list(self.seqparam['start']),
                'stop': list(self.seqparam['stop'])
            }
        }

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # cdef update(self, list data, list states, list frames):
    # @njit
    def update(self, data, states, frames):

        # cdef bool optimized
        # cdef long id, f, V
        # cdef double p, q
        # cdef np.ndarray[np.float64_t, ndim=1] xo, x_final
        # cdef np.ndarray[np.float64_t, ndim=2] x_init
        # cdef State state
        # cdef Frame frame

        # update 5'UTS -> 5'UTS+ transition parameter
        # warm start for the optimization
        optimized = False
        if self.restrict:
            xo = np.hstack((self.seqparam['kozak'], self.seqparam['start'][1:2]))
        else:
            xo = np.hstack((self.seqparam['kozak'], self.seqparam['start'][1:]))

        V = xo.size
        x_init = xo.reshape(V,1)

        try:
            x_final, optimized = optimize_transition_initiation(x_init, data, states, frames, self.restrict)
            if optimized:
                self.seqparam['kozak'] = x_final[0]
                self.seqparam['start'][1] = x_final[1]
                if not self.restrict:
                    self.seqparam['start'][2:] = x_final[2:]

        except:
            # if any error is thrown, skip updating at this iteration
            pass

    def __reduce__(self):
        return (rebuild_Transition, (self.seqparam,self.restrict))

def rebuild_Transition(seqparam, restrict):
    t = Transition()
    t.seqparam = seqparam
    t.restrict = restrict
    return t

# @njit
def optimize_transition_initiation(x_init, data, states, frames, restrict):

    # @njit
    def func(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).ravel().astype(np.float64)

        if z is None:
            # compute likelihood function and gradient
            results = transition_func_grad(xx, data, states, frames, restrict)
            fd = results[0]
            Df = results[1]

            # check for infs and nans, in function and gradient
            if np.isnan(fd) or np.isinf(fd):
                f = np.array([np.finfo(np.float32).max]).astype(np.float64)
            else:
                f = np.array([fd]).astype(np.float64)

            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo(np.float32).max * np.ones((1, xx.size), dtype=np.float64)
            else:
                Df = Df.reshape(1, xx.size)

            return cvx.matrix(f), cvx.matrix(Df)

        else:
            # compute likelihood function, gradient and hessian
            results = transition_func_grad_hess(xx, data, states, frames, restrict)
            fd = results[0]
            Df = results[1]
            hess = results[2]

            # check for infs and nans, in function and gradient
            if np.isnan(fd) or np.isinf(fd):
                f = np.array([np.finfo(np.float32).max]).astype(np.float64)
            else:
                f = np.array([fd]).astype(np.float64)

            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo(np.float32).max * np.ones((1, xx.size), dtype=np.float64)
            else:
                Df = Df.reshape(1, xx.size)

            # check if hessian is positive semi-definite
            eigs = np.linalg.eig(hess)
            if np.any(eigs[0] < 0):
                raise ValueError
            hess = z[0] * hess
            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(hess)

    # call an unconstrained nonlinear solver
    solution = solvers.cp(func)

    # check if optimal value has been reached;
    # if not, re-optimize with a cold start
    optimized = solution['status'] in {'optimal', 'unknown'}
    x_final = np.array(solution['x']).ravel()

    return x_final, optimized


def transition_func_grad(x, data, states, frames, restrict):

    # cdef Data datum
    # cdef State state
    # cdef Frame frame
    # cdef long j, v, V
    # cdef double func, f
    # cdef np.ndarray arg, df, vec, tmp, xex

    xex = np.zeros((len(utils.STARTCODONS) + 1,), dtype=np.float64)
    xex[0] = utils.MIN
    xex[1] = x[1]
    if restrict:
        xex[2:] = utils.MIN
    else:
        xex[2:] = x[2:]

    V, func = x.size, 0
    df = np.zeros((V,), dtype=float)
    for datum, state, frame in zip(data, states, frames):

        # for j from 0 <= j < 3:
        for j in range(3):

            arg = x[0] * datum.codon_map['kozak'][1:, j] + xex[datum.codon_map['start'][1:, j]]

            # evaluate function
            func += frame.posterior[j] * np.sum(
                state.pos_cross_moment_start[j, 1:, 0] *
                arg - state.pos_cross_moment_start[j].sum(1)[1:] *
                utils.nplog(1 + np.exp(arg))
            )

            # evaluate gradient
            vec = (
                state.pos_cross_moment_start[j, 1:, 0] -
                state.pos_cross_moment_start[j].sum(1)[1:] *
                logistic(-arg)
            )
            df[0] += frame.posterior[j] * np.sum(vec * datum.codon_map['kozak'][1:, j])
            # for v from 1 <= v < V:
            for v in range(1, V):
                df[v] += frame.posterior[j] * np.sum(vec[datum.codon_map['start'][1:, j] == v])

    return -func, -df


def transition_func_grad_hess(x, data, states, frames, restrict):

    # cdef Data datum
    # cdef State state
    # cdef Frame frame
    # cdef long j, v, V
    # cdef double func
    # cdef np.ndarray xex, df, Hf, arg, vec, vec2, tmp

    xex = np.zeros((len(utils.STARTCODONS) + 1,), dtype=np.float64)
    xex[0] = utils.MIN
    xex[1] = x[1]
    if restrict:
        xex[2:] = utils.MIN
    else:
        xex[2:] = x[2:]

    V, func = x.size, 0
    df = np.zeros((V,), dtype=float)
    Hf = np.zeros((V, V), dtype=float)

    for datum, state, frame in zip(data, states, frames):

        # for j from 0 <= j < 3:
        for j in range(3):

            arg = x[0] * datum.codon_map['kozak'][1:, j] + xex[datum.codon_map['start'][1:, j]]

            # evaluate function
            func += frame.posterior[j] * np.sum(
                state.pos_cross_moment_start[j, 1:, 0] *
                arg - state.pos_cross_moment_start[j].sum(1)[1:] *
                utils.nplog(1 + np.exp(arg))
            )

            # evaluate gradient and hessian
            vec = (
                state.pos_cross_moment_start[j, 1:, 0] -
                state.pos_cross_moment_start[j].sum(1)[1:] * logistic(-arg)
            )
            vec2 = state.pos_cross_moment_start[j].sum(1)[1:] * logistic(arg) * logistic(-arg)
            df[0] += frame.posterior[j] * np.sum(vec * datum.codon_map['kozak'][1:, j])
            Hf[0,0] += frame.posterior[j] * np.sum(vec2 * datum.codon_map['kozak'][1:, j] ** 2)
            # for v from 1 <= v < V:
            for v in range(1, V):
                tmp = datum.codon_map['start'][1:, j] == v
                df[v] += frame.posterior[j] * np.sum(vec[tmp])
                Hf[v,v] += frame.posterior[j] * np.sum(vec2[tmp])
                Hf[0,v] += frame.posterior[j] * np.sum(vec2[tmp] * datum.codon_map['kozak'][1:, j][tmp])

    Hf[:, 0] = Hf[0, :]
    return -func, -df, Hf

class Emission(object):

    def __init__(self, scale_beta=10000.0, read_lengths=None):

        # cdef long r
        # cdef np.ndarray[np.float64_t, ndim=1] alpha_pattern
        # cdef np.ndarray[np.float64_t, ndim=2] periodicity

        self.S = 9
        self.R = len(read_lengths)
        self.periodicity = np.empty((self.R,self.S,3), dtype=np.float64)
        self.logperiodicity = np.empty((self.R,self.S,3), dtype=np.float64)
        self.rescale = np.empty((self.R,self.S,8), dtype=np.float64)
        self.rate_alpha = np.empty((self.R,self.S), dtype=np.float64)
        self.rate_beta = np.empty((self.R,self.S), dtype=np.float64)
        alpha_pattern = np.array([20.,100.,1000.,100.,50.,30.,60.,10.,1.])

        # for r from 0 <= r < self.R:
        for r in range(self.R):

            periodicity = np.ones((self.S,3), dtype=np.float64)
            periodicity[1:self.S-1,:] = np.random.rand(self.S-2,1)
            self.periodicity[r] = periodicity / utils.insum(periodicity, [1])
            self.logperiodicity[r] = utils.nplog(self.periodicity[r])

            self.rate_alpha[r] = alpha_pattern*np.exp(np.random.normal(0,0.01,self.S))
            self.rate_beta[r] = scale_beta*np.random.rand(self.S)

        self.compute_rescaling()

    def __getitem__(self, item):
        return self.__dict__[item]

    def _serialize(self):
        return {
            'S': self.S,
            'logperiodicity': {
                'data': list(self.logperiodicity.flatten()),
                'shape': self.logperiodicity.shape
            },
            'rescale': {
                'data': list(self.rescale.flatten()),
                'shape': self.rescale.shape
            },
            'rate_alpha': {
                'data': list(self.rate_alpha.flatten()),
                'shape': self.rate_alpha.shape
            },
            'rate_beta': {
                'data': list(self.rate_beta.flatten()),
                'shape': self.rate_beta.shape
            }
        }

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # @njit
    def update_periodicity(self, data, states, frames):

        # cdef bool optimized
        # cdef long j, r, s, t, f, m, T
        # cdef double ab
        # cdef list constants
        # cdef np.ndarray At, Bt, Ct, Et, result, argA, index
        # cdef Data datum
        # cdef State state
        # cdef Frame frame

        T = len(data)
        Et = np.array([d.transcript_normalization_factor for d in data]).reshape(T, 1)
        index = np.array([[4, 5, 6, 7],[2, 3, 6, 7],[1, 3, 5, 7]]).T

        # for r from 0 <= r < self.R:
        for r in range(self.R):

            # for s from 1 <= s < self.S-1:
            for s in range(1, self.S - 1):

                # compute constants
                At = np.zeros((3,), dtype=np.float64)
                for j in range(3):
                    At[j] = np.sum([
                        frame.posterior[f] * np.sum([
                            state.pos_first_moment[f, m, s] * datum.obs[3 * m + f + j,r]
                            for m in np.where(np.any(datum.missingness_type[r, f, :] == index[:, j:j + 1], 0))[0]
                        ]) for datum, state, frame in zip(data, states, frames)
                        for f in range(3)
                    ])
  
                Bt = np.zeros((T, 3), dtype=np.float64)
                Ct = np.zeros((T, 3), dtype=np.float64)
                for t, (datum, state, frame) in enumerate(zip(data, states, frames)):
                    argA = np.array([
                        frame.posterior[f] * state.pos_first_moment[f,:,s] *
                        (datum.total_pileup[f, :, r] + self.rate_alpha[r, s] * self.rate_beta[r,s])
                        for f in range(3)
                    ])
 
                    Bt[t, 0] += np.sum(argA[datum.missingness_type[r] == 3])
                    Bt[t, 1] += np.sum(argA[datum.missingness_type[r] == 5])
                    Bt[t, 2] += np.sum(argA[datum.missingness_type[r] == 6])
                    Ct[t, 0] += np.sum(argA[datum.missingness_type[r] == 4])
                    Ct[t, 1] += np.sum(argA[datum.missingness_type[r] == 2])
                    Ct[t, 2] += np.sum(argA[datum.missingness_type[r] == 1])

                constants = [At, Bt, Ct, Et, self.rate_beta[r,s]]

                # run optimizer
                try:
                    result, optimized = optimize_periodicity(self.periodicity[r,s,:], constants)
                    if optimized:
                        result[result <= 0] = utils.EPS
                        result = result/result.sum()
                        self.periodicity[r, s, :] = result
                except:
                    # if any error is thrown, skip updating at this iteration
                    pass

        self.logperiodicity = utils.nplog(self.periodicity)

        if np.isinf(self.logperiodicity).any() or np.isnan(self.logperiodicity).any():
            print('Warning: Inf/Nan in periodicity parameter')

        self.compute_rescaling()

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    def compute_rescaling(self):

        # cdef long r, s, j
        # cdef np.ndarray mask

        for r in range(self.R):
            for s in range(self.S):
                """
                TODO Need to use six here
                """
                for j,mask in utils.binarize.items():
                    self.rescale[r, s, j] = np.sum(self.periodicity[r, s, mask])

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # cdef update_beta(self, list data, list states, list frames, double reltol):
    # @njit
    def update_beta(self, data, states, frames, reltol):

        # cdef Data datum
        # cdef State state
        # cdef Frame frame
        # cdef long r, s, f
        # cdef double diff, reldiff
        # cdef np.ndarray beta, newbeta, denom, new_scale, mask

        reldiff = 1
        newbeta = self.rate_beta.copy()

        denom = np.zeros((self.R, self.S), dtype=np.float64)
        for datum,state,frame in zip(data,states,frames):

            for r in range(self.R):

                for s in range(self.S):

                    new_scale = self.rescale[r,s,datum.missingness_type[r]]
                    mask = np.logical_not(np.logical_and(new_scale>0,state.pos_first_moment[:,:,s]>0))
                    if np.all(mask):
                        continue
                    denom[r,s] = denom[r,s] + np.sum(frame.posterior * \
                                 np.sum(MaskedArray(state.pos_first_moment[:,:,s], mask=mask),1))

                denom[r,0] = denom[r,0] + np.sum(frame.posterior * \
                             np.array([datum.is_pos_mappable[:f,r].sum() for f in range(3)]))
                denom[r,8] = denom[r,8] + np.sum(frame.posterior * \
                             np.array([datum.is_pos_mappable[3*datum.M+f:,r].sum() for f in range(3)]))

        while reldiff>reltol:

            beta = newbeta.copy()
            newbeta = self._square_beta_map(beta, data, states, frames, denom)
            diff = np.abs(newbeta-beta).sum()
            reldiff = np.mean(np.abs(newbeta-beta)/beta)

        self.rate_beta = newbeta.copy()

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # cdef np.ndarray _square_beta_map(self, np.ndarray beta, list data, list states, list frames, np.ndarray denom):
    # @njit
    def _square_beta_map(self, beta, data, states, frames, denom):

        # cdef int step
        # cdef bool a_ok
        # cdef np.ndarray R, V, a
        # cdef list vars

        vars = [beta]
        for step in [0,1]:
            beta = self._beta_map(beta, data, states, frames, denom)
            vars.append(beta)

        R = vars[1] - vars[0]
        V = vars[2] - vars[1] - R
        a = -1.*np.sqrt(R**2/V**2)
        a[a>-1] = -1.
        a[np.logical_or(np.abs(R)<1e-4,np.abs(V)<1e-4)] = -1.

        # given two update steps, compute an optimal step that achieves
        # a better likelihood than the two steps.
        a_ok = False
        while not a_ok:

            beta = (1+a)**2*vars[0] - 2*a*(1+a)*vars[1] + a**2*vars[2]

            mask = beta<=0
            if np.any(mask):
                a[mask] = (a[mask]-1)/2.
                a[np.abs(a+1)<1e-4] = -1.
            else:
                a_ok = True
  
        beta = self._beta_map(beta, data, states, frames, denom)

        return beta
 
    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # cdef np.ndarray _beta_map(self, np.ndarray beta, list data, list states, list frames, np.ndarray denom):
    # @njit
    def _beta_map(self, beta, data, states, frames, denom):

        # cdef long f, r, s, l
        # cdef np.ndarray newbeta, argA, argB, new_scale, mask, pos
        # cdef Data datum
        # cdef State state
        # cdef Frame frame

        newbeta = np.zeros((self.R,self.S), dtype=np.float64)
        for datum,state,frame in zip(data,states,frames):

            for r in range(self.R):

                for s in range(self.S):

                    new_scale = self.rescale[r,s,datum.missingness_type[r]]
                    mask = np.logical_not(np.logical_and(new_scale>0,state.pos_first_moment[:,:,s]>0))
                    if np.all(mask):
                        continue

                    argA = datum.transcript_normalization_factor*MaskedArray(new_scale, mask=mask) + beta[r,s]
                    argB = MaskedArray(datum.total_pileup[:,:,r], mask=mask) + self.rate_alpha[r,s]*beta[r,s]
                    pos = MaskedArray(state.pos_first_moment[:,:,s], mask=mask)
                    newbeta[r,s] = newbeta[r,s] + np.sum(frame.posterior * np.sum(pos *
                                                                                  (utils.nplog(argA) - digamma(argB) + argB / argA / self.rate_alpha[r, s]), 1))

                for f in range(3):
                    # add extra terms for first state
                    # for l from 0 <= l < f:
                    for l in range(f):
                        if datum.is_pos_mappable[l,r]:
                            newbeta[r,0] = newbeta[r,0] + frame.posterior[f] * \
                                           ((utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 0]) - \
                                             digamma(datum.obs[l,r]+self.rate_alpha[r,0]*beta[r,0])) + \
                                            (datum.obs[l,r]+self.rate_alpha[r,0]*beta[r,0]) / \
                                            self.rate_alpha[r,0]/(datum.transcript_normalization_factor/3.+beta[r,0]))

                    # add extra terms for last state
                    # for l from 3*datum.M+f <= l < datum.L:
                    for l in range(3*datum.M+f, datum.L):
                        if datum.is_pos_mappable[l,r]:
                            newbeta[r,8] = newbeta[r,8] + frame.posterior[f] * \
                                           ((utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 8]) - \
                                             digamma(datum.obs[l,r]+self.rate_alpha[r,8]*beta[r,8])) + \
                                            (datum.obs[l,r]+self.rate_alpha[r,8]*beta[r,8]) / \
                                            self.rate_alpha[r,8]/(datum.transcript_normalization_factor/3.+beta[r,8]))

        newbeta = np.exp(newbeta / denom + digamma(self.rate_alpha*beta) - 1)
        return newbeta

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # @njit
    def update_alpha(self, data, states, frames):

        # cdef bool optimized
        # cdef long s
        # cdef list constants
        # cdef np.ndarray x_init, x_final

        # warm start for the optimization
        optimized = False
        x_init = self.rate_alpha

        try:
            x_final, optimized = optimize_alpha(x_init, data, states, frames, self.rescale, self.rate_beta)
            if optimized:
                self.rate_alpha = x_final

        except ValueError:
            # if any error is thrown, skip updating at this iteration
            pass

    def __reduce__(self):
        return (rebuild_Emission, (self.periodicity, self.rate_alpha, self.rate_beta))

def rebuild_Emission(periodicity, alpha, beta):
    e = Emission()
    e.periodicity = periodicity
    e.logperiodicity = utils.nplog(periodicity)
    e.rate_alpha = alpha
    e.rate_beta = beta
    e.compute_rescaling()
    return e


def optimize_periodicity(x_init, constants):
    # @njit
    def F(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).reshape(1,3)
        At = constants[0]
        Bt = constants[1]
        Ct = constants[2]
        Et = constants[3]
        ab = constants[4]

        # compute function
        func = np.sum(At * utils.nplog(xx)) - \
               np.sum(Bt * utils.nplog((1 - xx) * Et + ab)) - \
               np.sum(Ct * utils.nplog(xx * Et + ab))
        if np.isnan(func) or np.isinf(func):
            f = np.array([np.finfo(np.float32).max]).astype(np.float64)
        else:
            f = np.array([-1*func]).astype(np.float64)

        # compute gradient
        Df = At/xx[0] + np.sum(Bt * Et/((1-xx)*Et+ab),0) - \
             np.sum(Ct * Et/(xx*Et+ab),0)
        if np.isnan(Df).any() or np.isinf(Df).any():
            Df = -1 * np.finfo(np.float32).max * \
                 np.ones((1,xx.size), dtype=np.float64)
        else:
            Df = -1*Df.reshape(1,xx.size)

        if z is None:
            return cvx.matrix(f), cvx.matrix(Df)

        # compute hessian
        hess = 1.*At/xx[0]**2 - np.sum(Bt * Et**2/((1-xx)*Et+ab)**2,0) - \
               np.sum(Ct * Et**2/(xx*Et+ab)**2,0)

        # check if hessian is positive semi-definite
        if np.any(hess<0) or np.any(np.isnan(hess)) or np.any(np.isinf(hess)):
            raise ValueError
        else:
            hess = z[0] * np.diag(hess)

        return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(hess)

    V = x_init.size
    # specify constraints on variables
    G = cvx.matrix(np.diag(-1*np.ones((V,), dtype=np.float64)))
    h = cvx.matrix(np.zeros((V,1), dtype=np.float64))
    A = cvx.matrix(np.ones((1,V), dtype=np.float64))
    b = cvx.matrix(np.ones((1,1), dtype=np.float64))

    # call a constrained nonlinear solver
    solution = solvers.cp(F, G=G, h=h, A=A, b=b)

    if solution['status'] in ['optimal','unknown']:
        optimized = True
    else:
        optimized = False
    x_final = np.array(solution['x']).ravel()

    return x_final, optimized


def optimize_alpha(x_init, data, states, frames, rescale, beta):
    # @njit
    def F(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init.reshape(V,1))

        xx = np.array(x).reshape(beta.shape)

        if z is None:
            # compute likelihood function and gradient
            results = alpha_func_grad(xx, data, states, frames, rescale, beta)

            # check for infs or nans
            fd = results[0]
            if np.isnan(fd) or np.isinf(fd):
                f = np.array([np.finfo(np.float32).max]).astype(np.float64)
            else:
                f = np.array([fd]).astype(np.float64)

            Df = results[1]
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo(np.float32).max * np.ones((1,xx.size), dtype=np.float64)
            else:
                Df = Df.reshape(1,xx.size)

            return cvx.matrix(f), cvx.matrix(Df)

        else:
            # compute function, gradient, and hessian
            results = alpha_func_grad_hess(xx, data, states, frames, rescale, beta)

            # check for infs or nans
            fd = results[0]
            if np.isnan(fd) or np.isinf(fd):
                f = np.array([np.finfo(np.float32).max]).astype(np.float64)
            else:
                f = np.array([fd]).astype(np.float64)

            Df = results[1]
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo(np.float32).max * np.ones((1,xx.size), dtype=np.float64)
            else:
                Df = Df.reshape(1,xx.size)

            # check if hessian is positive semi-definite
            hess = results[2]
            if np.any(hess)<0:
                raise ValueError
            hess = np.diag(z[0] * hess.ravel())

            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(hess)

    V = x_init.size
    # specify constraints on variables
    G = cvx.matrix(np.diag(-1*np.ones((V,), dtype=np.float64)))
    h = cvx.matrix(np.zeros((V,1), dtype=np.float64))

    # call a constrained nonlinear solver
    solution = solvers.cp(F, G=G, h=h)

    if solution['status'] in ['optimal','unknown']:
        optimized = True
    else:
        optimized = False
    x_final = np.array(solution['x']).reshape(x_init.shape)
    x_final[x_final<=0] = x_init[x_final<=0]

    return x_final, optimized


def alpha_func_grad(x, data, states, frames, rescale, beta):

    # cdef Data datum
    # cdef State state
    # cdef Frame frame
    # cdef long s, f, l, r, R, S
    # cdef double func
    # cdef np.ndarray gradient, mask, new_scale, argA, argB, argC, pos

    R = beta.shape[0]
    S = beta.shape[1]

    func = 0
    gradient = np.zeros((R,S), dtype=np.float64)
    for datum, state, frame in zip(data, states, frames):
                    
        for r in range(R):

            for s in range(S):

                new_scale = rescale[r, s, datum.missingness_type[r]]
                mask = np.logical_not(np.logical_and(new_scale > 0, state.pos_first_moment[:, :, s] > 0))
                if np.all(mask):
                    continue

                argA = MaskedArray(datum.total_pileup[:, :, r], mask=mask, fill_value=1) + x[r, s] * beta[r, s]
                argB = datum.transcript_normalization_factor*MaskedArray(new_scale, mask=mask, fill_value=1) + beta[r, s]
                pos = MaskedArray(state.pos_first_moment[:, :, s], mask=mask, fill_value=1)
                argC = np.sum(pos, 1)

                func = func + np.sum(frame.posterior * np.sum(pos * \
                                                              (gammaln(argA) - argA * utils.nplog(argB)), 1)) + np.sum(frame.posterior * \
                                                                                                                       argC) * (x[r,s] * beta[r,s] * utils.nplog(beta[r, s]) - gammaln(x[r, s] * beta[r, s]))

                gradient[r,s] = gradient[r,s] + beta[r,s] * np.sum(frame.posterior * \
                                                                   np.sum(pos * (digamma(argA) - utils.nplog(argB)), 1)) + \
                                np.sum(frame.posterior * argC) * beta[r,s] * (utils.nplog(beta[r, s]) - \
                                                                              digamma(x[r,s]*beta[r,s]))

            for f in range(3):
                # add extra terms for first state
                # for l from 0 <= l < f:
                for l  in range(f):
                    if datum.is_pos_mappable[l,r]:
                        func = func + frame.posterior[f] * (x[r,0] * beta[r,0] * utils.nplog(beta[r, 0]) +
                                                            gammaln(datum.obs[l,r]+x[r,0]*beta[r,0]) - gammaln(x[r,0]*beta[r,0]) -
                                                            (datum.obs[l,r]+x[r,0]*beta[r,0]) * utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 0]))
                        gradient[r,0] = gradient[r,0] + frame.posterior[f] * beta[r,0] * (utils.nplog(beta[r, 0]) +
                                                                                          digamma(datum.obs[l,r]+x[r,0]*beta[r,0]) - digamma(x[r,0]*beta[r,0]) -
                                                                                          utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 0]))

                # add extra terms for last state
                # for l from 3*datum.M+f <= l < datum.L:
                for l in range(3*datum.M+f, datum.L):
                    if datum.is_pos_mappable[l,r]:
                        func = func + frame.posterior[f] * (x[r,8] * beta[r,8] * utils.nplog(beta[r, 8]) +
                                                            gammaln(datum.obs[l,r]+x[r,8]*beta[r,8]) - gammaln(x[r,8]*beta[r,8]) -
                                                            (datum.obs[l,r]+x[r,8]*beta[r,8]) * utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 8]))
                        gradient[r,8] = gradient[r,8] + frame.posterior[f] * beta[r,8] * (utils.nplog(beta[r, 8]) +
                                                                                          digamma(datum.obs[l,r]+x[r,8]*beta[r,8]) - digamma(x[r,8]*beta[r,8]) -
                                                                                          utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 8]))

    func = -1.*func
    gradient = -1.*gradient

    return func, gradient


def alpha_func_grad_hess(x, data, states, frames, rescale, beta):

    # cdef Data datum
    # cdef State state
    # cdef Frame frame
    # cdef long r, s, f, R, S, l
    # cdef double func
    # cdef np.ndarray gradient, mask, new_scale, argA, argB, argC, pos, hessian

    R = beta.shape[0]
    S = beta.shape[1]

    func = 0
    gradient = np.zeros((R,S), dtype=np.float64)
    hessian = np.zeros((R,S), dtype=np.float64)
    for datum,state,frame in zip(data,states,frames):

        for r in range(R):

            for s in range(S):

                new_scale = rescale[r,s,datum.missingness_type[r]]
                mask = np.logical_not(np.logical_and(new_scale>0,state.pos_first_moment[:,:,s]>0))
                if np.all(mask):
                    continue

                argA = MaskedArray(datum.total_pileup[:,:,r], mask=mask, fill_value=1) + x[r,s]*beta[r,s]
                argB = datum.transcript_normalization_factor*MaskedArray(new_scale, mask=mask, fill_value=1) + beta[r,s]
                pos = MaskedArray(state.pos_first_moment[:,:,s], mask=mask, fill_value=1)
                argC = np.sum(pos, 1)

                func = func + np.sum(frame.posterior * np.sum(pos * \
                                                              (gammaln(argA) - argA * utils.nplog(argB)), 1)) + np.sum(frame.posterior * \
                                                                                                                       argC) * (x[r,s] * beta[r,s] * utils.nplog(beta[r, s]) - gammaln(x[r, s] * beta[r, s]))

                gradient[r,s] = gradient[r,s] + beta[r,s] * np.sum(frame.posterior * \
                                                                   np.sum(pos * (digamma(argA) - utils.nplog(argB)), 1)) + \
                                np.sum(frame.posterior * argC) * (utils.nplog(beta[r, s]) - \
                                                                  digamma(x[r,s]*beta[r,s])) * beta[r,s]
                        
                hessian[r,s] = hessian[r,s] + beta[r,s]**2 * np.sum(frame.posterior * \
                               np.sum(pos * polygamma(1,argA),1)) - beta[r,s]**2 * \
                               np.sum(frame.posterior * argC) * polygamma(1,x[r,s]*beta[r,s])

            for f in range(3):
                # add extra terms for first state
                # for l from 0 <= l < f:
                for l in range(f):
                    if datum.is_pos_mappable[l,r]:
                        func = func + frame.posterior[f] * (x[r,0] * beta[r,0] * utils.nplog(beta[r, 0]) + \
                                                            gammaln(datum.obs[l,r]+x[r,0]*beta[r,0]) - gammaln(x[r,0]*beta[r,0]) - \
                                                            (datum.obs[l,r]+x[r,0]*beta[r,0]) * utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 0]))
                        gradient[r,0] = gradient[r,0] + frame.posterior[f] * beta[r,0] * (utils.nplog(beta[r, 0]) + \
                                                                                          digamma(datum.obs[l,r]+x[r,0]*beta[r,0]) - digamma(x[r,0]*beta[r,0]) - \
                                                                                          utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 0]))
                        hessian[r,0] = hessian[r,0] + frame.posterior[f] * beta[r,0]**2 * \
                                       (polygamma(1,datum.obs[l,r]+x[r,0]*beta[r,0]) - \
                                       polygamma(1,x[r,0]*beta[r,0]))

                # add extra terms for last state
                # for l from 3*datum.M+f <= l < datum.L:
                for l in range(3*datum.M+f, datum.L):
                    if datum.is_pos_mappable[l,r]:
                        func = func + frame.posterior[f] * (x[r,8] * beta[r,8] * utils.nplog(beta[r, 8]) + \
                                                            gammaln(datum.obs[l,r]+x[r,8]*beta[r,8]) - gammaln(x[r,8]*beta[r,8]) - \
                                                            (datum.obs[l,r]+x[r,8]*beta[r,8]) * utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 8]))
                        gradient[r,8] = gradient[r,8] + frame.posterior[f] * beta[r,8] * (utils.nplog(beta[r, 8]) + \
                                                                                          digamma(datum.obs[l,r]+x[r,8]*beta[r,8]) - digamma(x[r,8]*beta[r,8]) - \
                                                                                          utils.nplog(datum.transcript_normalization_factor / 3. + beta[r, 8]))
                        hessian[r,8] = hessian[r,8] + frame.posterior[f] * beta[r,8]**2 * \
                                       (polygamma(1,datum.obs[l,r]+x[r,8]*beta[r,8]) - \
                                       polygamma(1,x[r,8]*beta[r,8]))

    func = -1.*func
    gradient = -1.*gradient
    hessian = -1.*hessian

    return func, gradient, hessian

def learn_parameters(observations, codon_id, scales, mappability, scale_beta, mintol, read_lengths):

    # cdef long restart, i, D
    # cdef double scale, Lmax, L, dL, newL, reltol, starttime, totaltime
    # cdef str start
    # cdef list data, Ls, states, frames, ig
    # cdef dict id
    # cdef np.ndarray observation, mappable
    # cdef Data datum
    # cdef Emission emission, best_emission
    # cdef Transition transition, best_transition
    # cdef State state
    # cdef Frame frame

    data = [
        Data(observation, id, scale, mappable)
        for observation, id, scale, mappable
        in zip(observations, codon_id, scales, mappability)
    ]

    # Initialize latent variables
    states = [State(datum.M) for datum in data]
    frames = [Frame() for _ in range(len(data))]

    print('Stage 1: allow only AUG start codons; only update periodicity parameters ...')
    transition = Transition()
    emission = Emission(scale_beta, read_lengths)

    # compute initial log likelihood
    for datum, state, frame in zip(data, states, frames):

        datum.compute_log_probability(emission)

        state._forward_update(datum, transition)

        frame.update(datum, state)

    # L is log likelihood
    L = np.sum([
        np.sum(frame.posterior*state.likelihood) + np.sum(frame.posterior*datum.extra_log_probability)
        for datum, state, frame in zip(data, states, frames)
    ]) / np.sum([datum.L for datum in data])
    dL = np.inf

    # iterate till convergence
    reltol = dL / np.abs(L)
    while np.abs(reltol) > mintol:

        starttime = time.time()

        for state, datum in zip(states, data):
            state._reverse_update(datum, transition)

        # update periodicity parameters
        emission.update_periodicity(data, states, frames)

        # update transition parameters
        transition.update(data, states, frames)

        # compute log likelihood
        for datum, state, frame in zip(data,states,frames):

            datum.compute_log_probability(emission)

            state._forward_update(datum, transition)

            frame.update(datum, state)

        newL = np.sum([
            np.sum(frame.posterior*state.likelihood) + np.sum(frame.posterior*datum.extra_log_probability)
            for datum,state,frame in zip(data,states,frames)
        ]) / np.sum([datum.L for datum in data])

        # Set change in likelihood and the likelihood for the next round
        dL, L = newL - L, newL
        reltol = dL / np.abs(L)

        print('\nLoss: {}'.format(L))
        print('Relative tolerance (convergence): {} ({})'.format(reltol, mintol))
        print('Time in sec, this iteration: {}'.format(time.time() - starttime))
        # print L, reltol, time.time()-starttime

    print('Stage 2: allow only AUG start codons; update all parameters ...')
    dL = np.inf
    reltol = dL/np.abs(L)
    while np.abs(reltol) > mintol:

        starttime = time.time()

        # update latent states
        for state, datum in zip(states, data):

            state._reverse_update(datum, transition)

        # update periodicity parameters
        emission.update_periodicity(data, states, frames)

        # update occupancy parameters
        emission.update_alpha(data, states, frames)

        emission.update_beta(data, states, frames, 1e-3)

        # update transition parameters
        transition.update(data, states, frames)

        # compute log likelihood
        for datum, state, frame in zip(data, states, frames):

            datum.compute_log_probability(emission)

            state._forward_update(datum, transition)

            frame.update(datum, state)

        newL = np.sum([
            np.sum(frame.posterior*state.likelihood) + np.sum(frame.posterior*datum.extra_log_probability)
            for datum,state,frame in zip(data,states,frames)
        ]) / np.sum([datum.L for datum in data])

        # Set change in likelihood and the likelihood for the next round
        dL, L = newL - L, newL
        reltol = dL / np.abs(L)

        # print L, reltol, time.time()-starttime
        print('\nLoss: {}'.format(L))
        print('Relative tolerance (convergence): {} ({})'.format(reltol, mintol))
        print('Time in sec, this iteration: {}'.format(time.time() - starttime))

    print('Stage 3: allow noncanonical start codons ...')

    transition.restrict = False

    transition.seqparam['start'][2:] = -3 + np.random.rand(transition.seqparam['start'][2:].size)

    # Initial log likelihood again?
    for datum, state, frame in zip(data, states, frames):

        state._forward_update(datum, transition)

        frame.update(datum, state)

    dL = np.inf
    reltol = dL / np.abs(L)
    while np.abs(reltol) > mintol:

        totaltime = time.time()

        # update latent variables
        starttime = time.time()
        for state,datum in zip(states, data):
            state._reverse_update(datum, transition)

        # update transition parameters for noncanonical codons
        transition.update(data, states, frames)

        # compute log likelihood
        for datum, state, frame in zip(data, states, frames):

            state._forward_update(datum, transition)

            frame.update(datum, state)

        newL = np.sum([
            np.sum(frame.posterior*state.likelihood) + np.sum(frame.posterior*datum.extra_log_probability)
            for datum,state,frame in zip(data,states,frames)
        ]) / np.sum([datum.L for datum in data])

        # Set change in likelihood and the likelihood for the next round
        dL, L = newL - L, newL
        reltol = dL / np.abs(L)

        # print L, reltol, time.time()-starttime
        print('\nLoss: {}'.format(L))
        print('Relative tolerance (convergence): {} ({})'.format(reltol, mintol))
        print('Time in sec, this iteration: {}'.format(time.time() - starttime))

    return transition, emission, L

def infer_coding_sequence(riboseq_footprint_pileups, codon_maps, transcript_normalization_factors,
                          mappability, transition, emission):
    """
    Inflate serialized transition and emission dictionaries
    """
    # For backward compatibility
    observations, codon_id, scales = riboseq_footprint_pileups, codon_maps, transcript_normalization_factors
    transition = {
        'seqparam': {
            'kozak': np.array(transition['seqparam']['kozak']),
            'start': np.array(transition['seqparam']['start']),
            'stop': np.array(transition['seqparam']['stop'])
        }
    }

    emission = {
        'S': emission['S'],
        'logperiodicity': np.array(emission['logperiodicity']['data']).reshape(emission['logperiodicity']['shape']),
        'rescale': np.array(emission['rescale']['data']).reshape(emission['rescale']['shape']),
        'rate_alpha': np.array(emission['rate_alpha']['data']).reshape(emission['rate_alpha']['shape']),
        'rate_beta': np.array(emission['rate_beta']['data']).reshape(emission['rate_beta']['shape'])
    }

    data = [
        Data(
            riboseq_footprint_pileup,
            codon_map,
            transcript_normalization_factor,
            is_pos_mappable
        )
        for riboseq_footprint_pileup, codon_map, transcript_normalization_factor, is_pos_mappable
        in zip(riboseq_footprint_pileups, codon_maps, transcript_normalization_factors, mappability)
    ]
    states = [State(datum.n_triplets) for datum in data]
    frames = [Frame() for datum in data]

    # for state,frame,datum in zip(states, frames, data):
    for i, (state, frame, datum) in enumerate(zip(states, frames, data), start=1):
        logger.info('==== Starting inference loop {}'.format(i))
        datum.compute_log_probability(emission)
        state._forward_update(datum, transition)
        frame.update(datum, state)
        state.decode(datum, transition)

    return states, frames


def get_triplet_state(triplet_i, start_pos, stop_pos):
    if triplet_i < start_pos - 1:
        return States.ST_5PRIME_UTS
    if triplet_i == start_pos - 1:
        return States.ST_5PRIME_UTS_PLUS
    if triplet_i == start_pos:
        return States.ST_TIS
    if triplet_i == start_pos + 1:
        return States.ST_TIS_PLUS
    if triplet_i == stop_pos - 2:
        return States.ST_TTS_MINUS
    if triplet_i == stop_pos - 1:
        return States.ST_TTS
    if triplet_i == stop_pos:
        return States.ST_3PRIME_UTS_MINUS
    if triplet_i > stop_pos:
        return States.ST_3PRIME_UTS
    return States.ST_TES


def get_triplet_string(state):
    if state == States.ST_5PRIME_UTS:
        return 'ST_5PRIME_UTS'
    if state == States.ST_5PRIME_UTS_PLUS:
        return 'ST_5PRIME_UTS_PLUS'
    if state == States.ST_TIS:
        return 'ST_TIS'
    if state == States.ST_TIS_PLUS:
        return 'ST_TIS_PLUS'
    if state == States.ST_TES:
        return 'ST_TES'
    if state == States.ST_TTS_MINUS:
        return 'ST_TTS_MINUS'
    if state == States.ST_TTS:
        return 'ST_TTS'
    if state == States.ST_3PRIME_UTS_MINUS:
        return 'ST_3PRIME_UTS_MINUS'
    return 'ST_3PRIME_UTS'



def discovery_mode_data_logprob(riboseq_footprint_pileups, codon_maps, transcript_normalization_factors, mappability,
                                transition, emission, transcripts, sequences):
    emission = {
        'start': 'noncanonical',
        'S': emission['S'],
        'logperiodicity': np.array(emission['logperiodicity']['data']).reshape(emission['logperiodicity']['shape']),
        'rescale': np.array(emission['rescale']['data']).reshape(emission['rescale']['shape']),
        'rate_alpha': np.array(emission['rate_alpha']['data']).reshape(emission['rate_alpha']['shape']),
        'rate_beta': np.array(emission['rate_beta']['data']).reshape(emission['rate_beta']['shape'])
    }
    transition = {
        'start': 'noncanonical',
        'seqparam': {
            'kozak': np.array(transition['seqparam']['kozak']),
            'start': np.array(transition['seqparam']['start']),
            'stop': np.array(transition['seqparam']['stop'])
        }
    }

    transcript_elements = zip(
        riboseq_footprint_pileups,
        codon_maps,
        transcript_normalization_factors,
        mappability,
        sequences,
        transcripts
    )
    # for riboseq_footprint_pileup, codon_map, transcript_normalization_factor, is_pos_mappable, seq, transcript in zip(riboseq_footprint_pileups, codon_maps, transcript_normalization_factors, mappability, sequences, transcripts):
    for elements in transcript_elements:
        (
            riboseq_footprint_pileup,
            codon_map,
            transcript_normalization_factor,
            is_pos_mappable,
            seq,
            transcript
        ) = elements
        transcript.data_obj = Data(
            riboseq_footprint_pileup,
            codon_map,
            transcript_normalization_factor,
            is_pos_mappable,
            seq
        )


    discovery_mode_results = list()
    orf_posteriors = list()
    candidate_cds_matrices = list()
    # For each transcript
    i = 0

    # Just for debugging purposes
    from ribohmm.contrib.load_data import read_annotations
    start_codon_annotated, stop_codon_annotated = read_annotations()

    transcript_id = ''
    for transcript in transcripts:
        try:
            # transcript_id = transcript.raw_attrs.get('reference_id', transcript.raw_attrs.get('transcript_id'))
            try:
                transcript_id = transcript.raw_attrs['transcript_id']
            except KeyError:
                raise KeyError('Could not find transcript id')
            print('!!!!! Looking at transcript {} !!!!!'.format(transcript_id))
            i += 1
            transcript.data_obj.compute_log_probability(emission)
            transcript.state_obj = state = State(transcript.data_obj.n_triplets)
            transcript.state_obj._forward_update(data=transcript.data_obj, transition=transition)
            emission_errors = transcript.data_obj.compute_observed_pileup_deviation(emission, return_sorted=False, transcript_obj=transcript)
            import pickle
            with open('rmse.pkl', 'wb') as out:
                pickle.dump(emission_errors, out)
            emission_errors_normalized_tes = transcript.data_obj.compute_observed_pileup_deviation(emission, return_sorted=False, normalize_tes=True, transcript_obj=transcript)
            with open('ssrmse.pkl', 'wb') as out:
                pickle.dump(emission_errors_normalized_tes, out)
            # ORF_EMISSION_ERROR_MEAN_WITH_UTR = 1
            # ORF_EMISSION_ERROR_BY_TRIPLET_SSE_WITH_UTR = 2
            ORF_EMISSION_ERROR_MEAN_ONLY_ORF = 1
            # ORF_EMISSION_ERROR_BY_TRIPLET_SSE_ONLY_ORF = 4
            ORF_EMISSION_ERROR_TRIPLETS_DROPPED_FOR_MAPPABILITY = 3
            # ORF_EMISSION_ERROR_ORF_PILEUPS = 5

            orf_periodicity_likelihoods, orf_occupancy_likelihoods = transcript.data_obj.compute_minimal_ORF_log_probability()

            candidate_cds_likelihoods = list()
            _, all_candidate_cds = transcript.data_obj.orf_state_matrix()
            all_candidate_cds = all_candidate_cds[0] + all_candidate_cds[1] + all_candidate_cds[2]

            for candidate_cds, orf_emission_error, orf_emission_error_normalized_tes in zip(all_candidate_cds, emission_errors, emission_errors_normalized_tes):
                triplet_likelihoods = list()
                triplet_periodicity_likelihoods = list()
                triplet_occupancy_likelihoods = list()
                triplet_alpha_values = list()
                triplet_state_likelihood_values = list()
                triplet_states = list()

                if transcript.strand == '-':
                    exonic_positions = np.arange(transcript.start, transcript.stop)[::-1][transcript.mask]
                else:
                    exonic_positions = np.arange(transcript.start, transcript.stop)[transcript.mask]
                # Remove initial bases to set the frame
                for _ in range(candidate_cds.frame):
                    exonic_positions = np.delete(exonic_positions, 0)
                # If needed, add placeholder values to make sequence divisible by 3
                if len(exonic_positions) % 3 in {1, 2}:
                    exonic_positions = np.append(exonic_positions, [-2] * (3 - (len(exonic_positions) % 3)))

                # Chunk exonic positions into triplets
                triplet_genomic_positions = exonic_positions.reshape(-1, 3)

                # Get genomic position of start and stop codons
                start_genomic_pos = list(triplet_genomic_positions[candidate_cds.start])
                stop_genomic_pos = list(triplet_genomic_positions[candidate_cds.stop])

                # Get the data log probability for each position in this transcript, with the states defined by
                # the candidate CDS
                for triplet_i in range(transcript.data_obj.log_probability.shape[1]):
                    triplet_state = get_triplet_state(triplet_i, start_pos=candidate_cds.start, stop_pos=candidate_cds.stop)
                    triplet_likelihoods.append(transcript.data_obj.log_probability[candidate_cds.frame, triplet_i, triplet_state])
                    triplet_periodicity_likelihoods.append(transcript.data_obj.periodicity_model[candidate_cds.frame, triplet_i, triplet_state])
                    triplet_occupancy_likelihoods.append(transcript.data_obj.occupancy_model[candidate_cds.frame, triplet_i, triplet_state])
                    triplet_alpha_values.append(transcript.state_obj.alpha[candidate_cds.frame, triplet_i, triplet_state])
                    triplet_state_likelihood_values.append(transcript.state_obj.likelihood[triplet_i, candidate_cds.frame])
                    triplet_states.append(get_triplet_string(triplet_state))

                try:
                    minimum_ORF_start, minimum_ORF_end = transcript.data_obj.compute_minimal_ORF(candidate_cds)
                    minimum_ORF_length = minimum_ORF_end - minimum_ORF_start + 1
                except:
                    minimum_ORF_length = None

                candidate_cds_results = {
                    'definition': candidate_cds,
                    'triplet_states': triplet_states,
                    'start_codon_genomic_position': start_genomic_pos,
                    'stop_codon_genomic_position': stop_genomic_pos,
                    'annotated_start': start_codon_annotated.get(transcript_id),
                    'annotated_stop': stop_codon_annotated.get(transcript_id),
                    'strand': transcript.strand,
                    'exonic_positions': (
                        list(np.arange(transcript.start, transcript.stop)[transcript.mask])
                        if transcript.strand == '+'
                        else list(np.arange(transcript.start, transcript.stop)[::-1][transcript.mask])
                    ),
                    'start_codon_differences': [
                        start_genomic_pos[0] - start_codon_annotated.get(transcript_id, -9999999),
                        start_genomic_pos[1] - start_codon_annotated.get(transcript_id, -9999999),
                        start_genomic_pos[2] - start_codon_annotated.get(transcript_id, -9999999),
                    ],
                    'transcript_id': transcript_id,
                    'data_loglikelihood': {
                        # 'by_pos': triplet_likelihoods,
                        'sum': np.sum(triplet_likelihoods)
                    },

                    'only_ORF_data_loglikelihood': {
                        'periodicity': orf_periodicity_likelihoods.get(candidate_cds),
                        'occupancy': orf_occupancy_likelihoods.get(candidate_cds)
                    },
                    'only_ORF_length': minimum_ORF_length,
                    'only_ORF_riboseq_counts': transcript.data_obj.get_minimal_ORF_overlapping_reads(candidate_cds),

                    'data_loglikelihood_periodicity': {
                        # 'by_pos': triplet_periodicity_likelihoods,
                        'sum': np.sum(triplet_periodicity_likelihoods)
                    },
                    'data_loglikelihood_occupancy': {
                        # 'by_pos': triplet_occupancy_likelihoods,
                        'sum': np.sum(triplet_occupancy_likelihoods)
                    },
                    # 'state_alpha': {
                    #     'by_pos': triplet_alpha_values,
                    #     'sum': np.sum(triplet_alpha_values)
                    # },
                    # 'state_likelihood': {
                    #     'by_pos': triplet_state_likelihood_values,
                    #     'sum': np.sum(triplet_state_likelihood_values)
                    # },

                    # For debug log emission_errors_normalized_tes
                    'orf_emission_error': {
                        # 'mean_rmse_with_UTR': orf_emission_error[ORF_EMISSION_ERROR_MEAN_WITH_UTR],
                        # 'by_triplet_sse': orf_emission_error[ORF_EMISSION_ERROR_BY_TRIPLET_SSE_WITH_UTR],
                        'mean_rmse_only_ORF': orf_emission_error[ORF_EMISSION_ERROR_MEAN_ONLY_ORF],
                        # 'by_triplet_sse_only_ORF': orf_emission_error[ORF_EMISSION_ERROR_BY_TRIPLET_SSE_ONLY_ORF]
                        # 'by_triplet_sse': by_triplet_sse,
                        'triplets_dropped_for_mappability': orf_emission_error[ORF_EMISSION_ERROR_TRIPLETS_DROPPED_FOR_MAPPABILITY]
                    },
                    'orf_emission_error_normalized_tes': {
                        # 'mean_rmse_with_UTR': orf_emission_error_normalized_tes[ORF_EMISSION_ERROR_MEAN_WITH_UTR],
                        # 'by_triplet_sse': orf_emission_error[ORF_EMISSION_ERROR_BY_TRIPLET_SSE_WITH_UTR],
                        'mean_rmse_only_ORF': orf_emission_error_normalized_tes[ORF_EMISSION_ERROR_MEAN_ONLY_ORF],
                        # 'by_triplet_sse_only_ORF': orf_emission_error[ORF_EMISSION_ERROR_BY_TRIPLET_SSE_ONLY_ORF]
                        # 'by_triplet_sse': by_triplet_sse,
                        'triplets_dropped_for_mappability': orf_emission_error_normalized_tes[ORF_EMISSION_ERROR_TRIPLETS_DROPPED_FOR_MAPPABILITY]
                    }
                }
                candidate_cds_likelihoods.append(candidate_cds_results)

            transcript.frame_obj = frame = Frame()
            transcript.frame_obj.update(transcript.data_obj, transcript.state_obj)

            # These two matrices are the same dimensions
            # orf_posteriors_ is a matrix of orf posteriors, where the first dimension is the frame
            # candidate_cds_matrix are all CandidateCDS namedtuples
            orf_posteriors_, candidate_cds_matrix = transcript.state_obj.discovery_decode(data=transcript.data_obj, transition=transition, transcript=transcript)
            orf_posteriors.append(orf_posteriors_)
            candidate_cds_matrices.append(candidate_cds_matrix)
            # print(orf_posteriors)
            transcript.state_obj.decode(data=transcript.data_obj, transition=transition)
            discovery_mode_results.append({
                'candidate_orf': candidate_cds_likelihoods,
                'transcript_normalization_factor': transcript.data_obj.transcript_normalization_factor,
                'n_triplets': transcript.data_obj.n_triplets,
                'orf_posteriors': orf_posteriors_,
                # 'data_logprob_full': transcript.data_obj.log_probability.tolist(),
                # 'periodicity_model_full': transcript.data_obj.periodicity_model.tolist(),
                # 'occupancy_model_full': transcript.data_obj.occupancy_model.tolist(),
                # 'riboseq_counts_total_pileup': transcript.data_obj.total_pileup.tolist(),
                # 'riboseq_counts_total_pileup_sum_footprints': transcript.data_obj.total_pileup.sum(axis=2).tolist(),
                # 'data_logprob_full': transcript.data_obj.log_likelihood.tolist(),
                # 'state_alpha_full': state.alpha.tolist(),
                # 'state_decode_alphas': state.decode_alphas.tolist(),
                # 'triplet_genomic_positions': triplet_genomic_positions,
                'decode': {
                    'max_posterior': [utils.MAX if np.isinf(p) else p for p in transcript.state_obj.max_posterior],
                    'best_start': [int(b) if b is not None else b for b in transcript.state_obj.best_start],
                    'best_stop': [int(b) if b is not None else b for b in transcript.state_obj.best_stop]
                },
                'final_posterior': {
                    'by_frame': transcript.state_obj.max_posterior * transcript.frame_obj.posterior,
                    'max_index': np.argmax(transcript.state_obj.max_posterior * transcript.frame_obj.posterior)
                }
            })
        except Exception as e:
            print('Could not process transcript {}: {}'.format(transcript_id, tb.format_exc()))
    # print('************************')
    # print(f'n_orfs_blank: {n_orfs_blank}')
    # print(f'n_orfs_total: {n_orfs_total}')
    return orf_posteriors, candidate_cds_matrices, discovery_mode_results


"""
Compare against transcript definition
Compare against viterbi output
Get output from state.decode()

To visualize:
  - look at top for each transcript, bottom for each transcript
  - go back and look at original data, see if top group shows periodicity


I want all the output in this format:
[
    {
        'transcript_info': {'chr': 1, 'start': 100, 'stop': 200},
        'transcript_string': 'string',
        'candidate_cds': [{
            'definition': <CandidateCDS>,
            'triplet_states': list()
            'data_loglikelihood': {'by_pos': list(), 'sum': float},
            'state_alpha': {'by_pos': list(), 'sum': float},
            'state_likelihood': {'by_pos': list(), 'sum': float}
        }]
    },
    {}
]
"""


def compare_raw_seq_to_codon_map(genome_track, transcripts, codon_maps):
    """
    A dictionary with three keys: kozak, start, and stop. The values for each of those keys is an array of size
        (num_codons, 3), where the 3 is for each possible reading frame.

        The start array maps to the list of start codons in utils.STARTCODONS, and the stop array does the same.
        ex:
        [
          [0 1 0],
          [0 0 0],
          [3 0 0]
        ]
    """
    rna_sequences = genome_track.get_sequence(transcripts)
    for seq, codon_map in zip(rna_sequences, codon_maps):
        print(seq)
        print(codon_map['start'])
        for frame_i in range(3):
            frame_codon_map = list(codon_map['start'][:, frame_i])
            for state_i, state_value in enumerate(frame_codon_map):
                state_seq = seq[state_i * 3 + frame_i:(state_i + 1) * 3 + frame_i]
                if state_value > 0 and state_seq not in utils.STARTCODONS:
                    pass
                    # print('bad')


def state_matrix_qa(riboseq_footprint_pileups, codon_maps, transcript_normalization_factors, mappability,
                    genome_track, transcripts):
    data = [
        Data(
            riboseq_footprint_pileup,
            codon_map,
            transcript_normalization_factor,
            is_pos_mappable
        )
        for riboseq_footprint_pileup, codon_map, transcript_normalization_factor, is_pos_mappable
        in zip(riboseq_footprint_pileups, codon_maps, transcript_normalization_factors, mappability)
    ]

    rna_sequences = genome_track.get_sequence(transcripts)

    for transcript_i, (data_, seq) in enumerate(zip(data, rna_sequences)):
        correct_starts, correct_stops = [0, 0, 0], [0, 0, 0]
        state_matrix = data_.orf_state_matrix()
        for frame_i, state_matrix_ in enumerate(state_matrix):
            for orf_i, orf_states in enumerate(state_matrix_):
                for state_i, state_value in enumerate(orf_states):
                    state_seq = seq[state_i * 3 + frame_i:(state_i + 1) * 3 + frame_i]
                    if state_value == States.ST_TIS and state_seq in utils.STARTCODONS:
                        correct_starts[frame_i] += 1
                    elif state_value == States.ST_TTS and state_seq in utils.STOPCODONS:
                        correct_stops[frame_i] += 1
        # Print Report
        print(f'Transcript {transcript_i}')
        for frame_i in range(3):
            print('\tFrame {} | Starts {}/{} | Stop {}/{}'.format(
                frame_i, correct_starts[frame_i], len(state_matrix[frame_i]),
                correct_stops[frame_i], len(state_matrix[frame_i])
            ))

