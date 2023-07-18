import os
import argparse
import warnings
import json
import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor

import numpy as np

from ribohmm import core, utils
from ribohmm.core import seq as seq
from ribohmm.core.ribohmm import infer_coding_sequence, discovery_mode_data_logprob, state_matrix_qa, compare_raw_seq_to_codon_map

from ribohmm.core.ribohmm import Data

import logging
logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d|%(levelname)s] %(message)s',
    datefmt='%d%b%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger('viterbi_log')

# ignore warnings with these expressions
warnings.filterwarnings('ignore', '.*overflow encountered.*',)
warnings.filterwarnings('ignore', '.*divide by zero.*',)
warnings.filterwarnings('ignore', '.*invalid value.*',)

N_FRAMES = 3


def write_inferred_cds_discovery_mode(transcript, frame, rna_sequence, candidate_cds, orf_posterior,
                                      orf_start, orf_stop):
    """
    Two records to look at:
    chromosome start stop transcript_id posterior strand cdstart cdstop protein_seq num_exons exon_sizes exon_starts frame
    chr11 6947752 6979079 STRG.6583.8 18680 + 6947755 6947803 VRRGCSAAIYETRSRQ 6 207,579,83,133,96,2159, 6947752,6953324,6962801,6964313,6964776,6976920, 0
    chr11 6947752 6979079 STRG.6583.8 1098 + 6947779 6947803 IYETRSRQ 6 207,579,83,133,96,2159, 6947752,6953324,6962801,6964313,6964776,6976920, 0

    """
    normal_starting_state_sequence = {
        '>3 0s': 0,
        '2 0s': 0,
        '0': 0,
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0,
        '5': 0,
        '6': 0,
        '7': 0,
        '8': 0,
        '2 8s': 0,
        '>3 8s': 0,
    }
    greater_10k_starting_state_sequence = {
        '>3 0s': 0,
        '2 0s': 0,
        '0': 0,
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0,
        '5': 0,
        '6': 0,
        '7': 0,
        '8': 0,
    }
    try:
        # print(f'Transcript: {transcript.chromosome}:{transcript.start}:{transcript.stop}:{transcript.strand}:{transcript.id}'
        #       f' Posteriors: {list(frame.posterior)} | {orf_posterior} * {frame.posterior[candidate_cds.frame]}')
        posterior = int(orf_posterior * frame.posterior[candidate_cds.frame] * 10_000)
        # print(f'###### {posterior}')

        # if posterior > 10_000:
        #     # print(f'###### {posterior}')
        #
        #     # print('&&&&&&&&& {}'.format(state_sequence))
        #     greater_10k_starting_state_sequence[starts_with_state] += 1
        # else:
        #     normal_starting_state_sequence[starts_with_state] += 1
    except:
        posterior = 'NaN'

    state_sequence = Data.get_state_sequence(n_triplets=int(len(transcript.sequence) / 3) - 1,
                                             start=candidate_cds.start, stop=candidate_cds.stop)
    if state_sequence[0] == 0 and state_sequence[1] == 0 and state_sequence[2] == 0:
        starts_with_state = '>3 0s'
    elif state_sequence[0] == 0 and state_sequence[1] == 0:
        starts_with_state = '2 0s'
    elif state_sequence[0] == 0:
        starts_with_state = '0'
    else:
        starts_with_state = str(state_sequence[0])
    tis = orf_start  # This is base position, n151444ot a state position
    tts = orf_stop

    protein = utils.translate(rna_sequence[tis:tts])
    # identify TIS and TTS in genomic coordinates
    if transcript.strand == '+':
        cdstart = transcript.start + np.where(transcript.mask)[0][tis]
        cdstop = transcript.start + np.where(transcript.mask)[0][tts]
    else:
        cdstart = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tts]
        cdstop = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tis]

    # if int(cdstart) == 6947755 and int(cdstop) == 6947803:
    if (int(cdstart), int(cdstop)) == (6947755, 6947803) or (int(cdstart), int(cdstop)) == (6947779, 6947803):
        print('############# {}'.format(cdstart))
        print(orf_posterior)
        print(frame.posterior[candidate_cds.frame])


    towrite = [transcript.chromosome,
               transcript.start,
               transcript.stop,
               transcript.id,
               posterior,
               transcript.strand,
               cdstart,
               cdstop,
               protein,
               len(transcript.exons),
               ','.join(map(str, [e[1] - e[0] for e in transcript.exons])) + ',',
               ','.join(map(str, [transcript.start + e[0] for e in transcript.exons])) + ',',
               candidate_cds.frame
    ]

    # print('==============================')
    # print(normal_starting_state_sequence)
    # print(greater_10k_starting_state_sequence)
    return towrite, ('>10k' if isinstance(posterior, int) and posterior > 10_000 else 'normal', starts_with_state)


def write_inferred_cds(transcript, state, frame, rna_sequence):
    posteriors = state.max_posterior*frame.posterior
    index = np.argmax(posteriors)
    # print(f'Transcript: {transcript.chromosome}:{transcript.start}:{transcript.stop}:{transcript.strand}:{transcript.id}'
    #       f' Posteriors: {list(frame.posterior)} | {frame.posterior[index]}')
    tis = state.best_start[index]
    tts = state.best_stop[index]
    # print(f'Stop codon: {rna_sequence[tts-3:tts]}')
    # print(f'TTS (stop codon): {rna_sequence[tts:tts+3]}')

    # output is not a valid CDS
    if tis is None or tts is None:
        logger.warning(f'Could not find inference for transcript at '
                       f'{transcript.chromosome}:{transcript.start}:{transcript.stop}'
                       f':{transcript.strand}:{transcript.id} | tis: {tis} tts: {tts}')
        return None

    posterior = int(posteriors[index]*10000) 
    protein = utils.translate(rna_sequence[tis:tts])
    # identify TIS and TTS in genomic coordinates
    if transcript.strand=='+':
        cdstart = transcript.start + np.where(transcript.mask)[0][tis]
        cdstop = transcript.start + np.where(transcript.mask)[0][tts]
    else:
        cdstart = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tts]
        cdstop = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tis]

    towrite = [transcript.chromosome, 
               transcript.start, 
               transcript.stop, 
               transcript.id,
               posterior,
               transcript.strand,
               cdstart,
               cdstop,
               protein, 
               len(transcript.exons), 
               ','.join(map(str,[e[1]-e[0] for e in transcript.exons]))+',', 
               ','.join(map(str,[transcript.start+e[0] for e in transcript.exons]))+',',
               index
    ]
    return towrite


def infer_on_transcripts(primary_strand, transcripts, ribo_track, genome_track, rnaseq_track,
                         mappability_tabix_prefix, infer_algorithm, model_params):
    logger.info('!!!!!! Starting infer_on_transcripts()')
    if primary_strand not in {'+', '-'}:
        raise ValueError('primary_strand must be either + or -')
    opposite_strand = '-' if primary_strand == '+' else '+'

    logger.info(f'Looking at transcript {primary_strand} strands')
    for t in transcripts:
        if t.strand == opposite_strand:
            t.mask = t.mask[::-1]
            t.strand = primary_strand

    # check if all exons have at least 5 footprints
    exon_counts = ribo_track.get_exon_total_counts(transcripts)
    transcripts = [t for t, e in zip(transcripts, exon_counts) if np.all(e >= 5)]  # TODO Variable name should reflect change
    logger.info('In {} transcripts, all exons have at least 5 footprints'.format(len(transcripts)))

    records_to_write = list()
    greater_10ks = list()
    discovery_mode_debug_metadata = list()
    if len(transcripts) > 0:
        codon_maps = list()
        logger.info('Loading RNA sequences')
        rna_sequences = genome_track.get_sequence(transcripts)
        logger.info('Setting codon flags')
        for rna_sequence in rna_sequences:
            sequence = seq.RnaSequence(rna_sequence)
            codon_maps.append(sequence.mark_codons())

        # load footprint count data in transcripts
        logger.info('Getting riboseq footprint counts')
        footprint_counts = ribo_track.get_counts(transcripts)

        # load transcript-level rnaseq RPKM
        logger.info('Loading transcript level RNAseq RPKM')
        if rnaseq_track is None:
            rna_counts = np.ones((len(transcripts),), dtype='float')
        else:
            rna_counts = rnaseq_track.get_total_counts(transcripts)

        # load mappability of transcripts; transform mappability to missingness
        logger.info('Loading mappability')
        if mappability_tabix_prefix is not None:
            rna_mappability = genome_track.get_mappability(transcripts)
        else:
            rna_mappability = [np.ones(c.shape, dtype='bool') for c in footprint_counts]

        logger.info('Running inference')
        if infer_algorithm == 'viterbi':
            logger.info(f'Running the Viterbi algorithm over the {primary_strand} strand')
            states, frames = infer_coding_sequence(
                footprint_counts,
                codon_maps,
                rna_counts,
                rna_mappability,
                model_params['transition'],
                model_params['emission']
            )
            for transcript, state, frame, rna_sequence in zip(transcripts, states, frames, rna_sequences):
                record = write_inferred_cds(transcript, state, frame, rna_sequence)
                records_to_write.append(record)
        elif infer_algorithm == 'discovery':
            # transcripts = [t for t in transcripts if t.id == 'STRG.6219.1']
            # transcripts = [t for t in transcripts if t.id == 'STRG.6390.5']
            # transcripts = [t for t in transcripts if t.id == 'STRG.6326.5']
            logger.info(f'Running the Discovery algorithm over the {primary_strand} strand')
            pos_orf_posteriors, pos_candidate_cds_matrices, pos_frames, pos_data_log_probs = discovery_mode_data_logprob(
                riboseq_footprint_pileups=footprint_counts,
                codon_maps=codon_maps,
                transcript_normalization_factors=rna_counts,
                mappability=rna_mappability,
                transition=model_params['transition'],
                emission=model_params['emission'],
                transcripts=transcripts
            )
            discovery_mode_debug_metadata = [
                {
                    'transcript_info': {
                        'chr': t.chromosome,
                        'start': t.start,
                        'stop': t.stop,
                        'strand': t.strand,
                        'length': t.stop - t.start + 1
                    },
                    'transcript_string': str(t.raw_attrs),
                    'exons': {
                        'absolute': [(e[0] + t.start, e[1] + t.start) for e in t.exons],
                        'relative': t.exons
                    },
                    'riboseq_pileup_counts': {
                        read_length: list(f[:, read_length_i])
                        for read_length_i, read_length in enumerate(ribo_track.get_read_lengths())
                    },
                    'results': candidate_cds_likelihoods
                }
                for t, candidate_cds_likelihoods, f in zip(transcripts, pos_data_log_probs, footprint_counts)
            ]

            for transcript, frame, rna_sequence, orf_posterior_matrix, candidate_cds_matrix in zip(
                transcripts, pos_frames, rna_sequences, pos_orf_posteriors, pos_candidate_cds_matrices):
                for frame_i in range(N_FRAMES):
                    for orf_i, orf_posterior in enumerate(orf_posterior_matrix[frame_i]):
                        candidate_cds = candidate_cds_matrix[frame_i][orf_i]
                        record, greater_10k = write_inferred_cds_discovery_mode(
                            transcript=transcript,
                            frame=frame,
                            rna_sequence=rna_sequence,
                            candidate_cds=candidate_cds,
                            orf_posterior=orf_posterior,
                            # This is the same formula used in State.decode()
                            orf_start=candidate_cds.start * 3 + candidate_cds.frame,
                            orf_stop=candidate_cds.stop * 3 + candidate_cds.frame
                        )
                        records_to_write.append(record)
                        greater_10ks.append(greater_10k)
    return records_to_write, discovery_mode_debug_metadata, greater_10ks






def infer_CDS(
    model_file,
    transcript_models,
    genome_track,
    mappability_tabix_prefix,
    ribo_track,
    rnaseq_track,
    output_directory,
    infer_algorithm='viterbi',
    dev_restrict_transcripts_to=None,
    dev_output_debug_data=None,
    n_procs=1,
    n_transcripts_per_proc=10
):
    logger.info('Starting infer_CDS()')
    N_TRANSCRIPTS = dev_restrict_transcripts_to  # Set to None to allow all transcripts
    print(f'!!!!!!!!!! N Transcripts: {N_TRANSCRIPTS}')
    N_FRAMES = 3
    DEBUG_OUTPUT_FILENAME = 'feb07.json'

    """
    Load the model from JSON
    """
    model_params = json.load(open(model_file))

    # load transcripts
    transcript_names = list(transcript_models.keys())[:N_TRANSCRIPTS]
    print(f'!!!!! Size of transcripts: {len(transcript_names)}')
    logger.info('Number of transcripts: {}'.format(len(transcript_names)))

    # open output file handle
    # file in bed12 format
    logger.info('Writing output headers')
    handle = open(os.path.join(output_directory, 'inferred_CDS.bed'), 'w')
    towrite = ["chromosome", "start", "stop", "transcript_id", 
               "posterior", "strand", "cdstart", "cdstop", 
               "protein_seq", "num_exons", "exon_sizes", "exon_starts", "frame"]
    handle.write(" ".join(map(str, towrite))+'\n')

    # Process 1000 transcripts at a time
    debug_metadata = defaultdict(list)
    records_to_write = list()
    futs = list()
    with ProcessPoolExecutor(max_workers=max(1, n_procs)) as executor:
        for n in range(int(np.ceil(len(transcript_names)/n_transcripts_per_proc))):
            logger.info('Processing transcripts {}-{}'.format(n * n_transcripts_per_proc, (n + 1) * n_transcripts_per_proc))
            tnames = transcript_names[n*n_transcripts_per_proc:(n+1)*n_transcripts_per_proc]
            transcripts_chunk = [transcript_models[name] for name in tnames]

            for infer_strand in ['+', '-']:
                futs.append(executor.submit(
                    infer_on_transcripts,
                    primary_strand=infer_strand,
                    transcripts=transcripts_chunk,
                    ribo_track=ribo_track,
                    genome_track=genome_track,
                    rnaseq_track=rnaseq_track,
                    mappability_tabix_prefix=mappability_tabix_prefix,
                    infer_algorithm=infer_algorithm,
                    model_params=model_params
                ))

    wait(futs)
    greater_10k_results = {
        '>10k': {
            '>3 0s': 0,
            '2 0s': 0,
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
            '5': 0,
            '6': 0,
            '7': 0,
            '8': 0,
        },
        'normal': {
            '>3 0s': 0,
            '2 0s': 0,
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
            '5': 0,
            '6': 0,
            '7': 0,
            '8': 0,
        }
    }
    for fut in futs:
        records_to_write_, debug_metadata_, greater_10k_result = fut.result()
        print(f'Got {len(records_to_write_)} records to write')
        records_to_write.extend(records_to_write_)
        if debug_metadata_:
            debug_metadata[infer_strand].extend(debug_metadata_)
        print(greater_10k_result)
        for gt10kr in greater_10k_result:
            greater_10k_results[gt10kr[0]][gt10kr[1]] += 1

    print('=========================')
    print(greater_10k_results)
    print('=========================')

    # Output records
    for record in records_to_write:
        if record is not None:
            try:
                handle.write(" ".join(map(str, record)) + '\n')
            except ValueError as e:
                logger.warning(str(e))
    # Output debug output bundle
    if debug_metadata and dev_output_debug_data:
        with open(os.path.join(output_directory, dev_output_debug_data), 'w') as out:
            json.dump(serialize_output({'pos': debug_metadata['+'], 'neg': debug_metadata['-']}), out)

    logger.info('Closing handles')
    handle.close()
    ribo_track.close()

    if rnaseq_track is not None:
        rnaseq_track.close()
    genome_track.close()

    logger.info('Finished')


def serialize_output(results):
    if isinstance(results, list):
        return [serialize_output(r) for r in results]
    if isinstance(results, dict):
        return {k: serialize_output(v) for k, v in results.items()}
    if isinstance(results, np.int64):
        return int(results)
    if isinstance(results, np.ndarray):
        return list(results)
    return results
# if __name__=="__main__":
#
#     options = parse_args()
#
#     infer(options)
