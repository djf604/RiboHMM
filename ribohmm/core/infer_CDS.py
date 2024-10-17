import os
import argparse
import warnings
import json
import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
from typing import Dict, List

import numpy as np

from ribohmm import core, utils
from ribohmm.core.seq import RnaSequence
from ribohmm.contrib.load_data import Transcript
from ribohmm.core.ribohmm import infer_coding_sequence, discovery_mode_data_logprob, state_matrix_qa, compare_raw_seq_to_codon_map

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
    try:
        # print(f'Transcript: {transcript.chromosome}:{transcript.start}:{transcript.stop}:{transcript.strand}:{transcript.id}'
        #       f' Posteriors: {list(frame.posterior)} | {orf_posterior} * {frame.posterior[candidate_cds.frame]}')
        posterior = int(orf_posterior * frame.posterior[candidate_cds.frame] * 10_000)
        # print(f'###### {posterior}')
    except:
        posterior = 'NaN'
    tis = orf_start  # This is base position, not a state position
    tts = orf_stop

    protein = utils.translate(rna_sequence[tis:tts])
    # identify TIS and TTS in genomic coordinates
    if transcript.strand == '+':
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
               ','.join(map(str, [e[1] - e[0] for e in transcript.exons])) + ',',
               ','.join(map(str, [transcript.start + e[0] for e in transcript.exons])) + ',',
               candidate_cds.frame
    ]
    return towrite


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


def infer_on_transcripts(primary_strand, transcripts: List[Transcript], ribo_track, genome_track, rnaseq_track,
                         mappability_tabix_prefix, infer_algorithm, model_params):
    logger.info('!!!!!! Starting infer_on_transcripts()')
    if primary_strand not in {'+', '-'}:
        raise ValueError('primary_strand must be either + or -')
    opposite_strand = '-' if primary_strand == '+' else '+'

    # Only for debugging ORFs
    # transcripts = [t for t in transcripts if t.id in {
    #     'STRG.6369.3',
    #     'STRG.6877.1',
    #     'STRG.6285.11',
    #     'STRG.6377.18',
    #     'STRG.6667.5'
    # }]

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
    discovery_mode_debug_metadata = list()
    if len(transcripts) > 0:
        codon_maps = list()
        logger.info('Loading RNA sequences')
        rna_sequences = genome_track.get_sequence(transcripts, add_seq_to_transcript=True)
        logger.info('Setting codon flags')
        for rna_sequence in rna_sequences:
            sequence = RnaSequence(rna_sequence)
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
            logger.info(f'Running the Discovery algorithm over the {primary_strand} strand')
            pos_orf_posteriors, pos_candidate_cds_matrices, pos_data_log_probs = discovery_mode_data_logprob(
                riboseq_footprint_pileups=footprint_counts,
                codon_maps=codon_maps,
                transcript_normalization_factors=rna_counts,
                mappability=rna_mappability,
                transition=model_params['transition'],
                emission=model_params['emission'],
                transcripts=transcripts,
                sequences=rna_sequences
            )
            discovery_mode_debug_metadata = [
                {
                    'transcript_info': {
                        'chr': t.chromosome,
                        'start': t.start,
                        'stop': t.stop,
                        'strand': t.strand,
                        'length': t.stop - t.start + 1,
                        'id': t.id
                    },
                    'sequence': seq,
                    'transcript_string': str(t.raw_attrs),
                    'exons': {
                        'absolute': [(e[0] + t.start, e[1] + t.start) for e in t.exons],
                        'relative': t.exons,
                        'mask': list([int(m) for m in t.mask])
                    },
                    # 'riboseq_pileup_counts': {
                    #     read_length: list(f[:, read_length_i])
                    #     for read_length_i, read_length in enumerate(ribo_track.get_read_lengths())
                    # },
                    'results': candidate_cds_likelihoods
                }
                for t, candidate_cds_likelihoods, f, seq in zip(transcripts, pos_data_log_probs, footprint_counts, rna_sequences)
            ]

            for transcript, rna_sequence, orf_posterior_matrix, candidate_cds_matrix in zip(
                transcripts, rna_sequences, pos_orf_posteriors, pos_candidate_cds_matrices):
                for frame_i in range(N_FRAMES):
                    for orf_i, orf_posterior in enumerate(orf_posterior_matrix[frame_i]):
                        candidate_cds = candidate_cds_matrix[frame_i][orf_i]
                        record = write_inferred_cds_discovery_mode(
                            transcript=transcript,
                            frame=transcript.frame_obj,
                            rna_sequence=rna_sequence,
                            candidate_cds=candidate_cds,
                            orf_posterior=orf_posterior,
                            # This is the same formula used in State.decode()
                            orf_start=candidate_cds.start * 3 + candidate_cds.frame,
                            orf_stop=candidate_cds.stop * 3 + candidate_cds.frame
                        )
                        records_to_write.append(record)
    return records_to_write, discovery_mode_debug_metadata






def infer_CDS(
    model_file,
    transcript_models: Dict[str, Transcript],
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
    # print(f'Processing N Transcripts: {N_TRANSCRIPTS}')
    N_FRAMES = 3
    DEBUG_OUTPUT_FILENAME = 'feb07.json'

    """
    Load the model from JSON
    """
    model_params = json.load(open(model_file))

    # load transcripts
    transcript_names: List[str] = list(transcript_models.keys())[:N_TRANSCRIPTS]
    # print(f'!!!!! Size of transcripts: {len(transcript_names)}')
    logger.info('Number of transcripts: {}'.format(len(transcript_names)))

    # open output file handle
    # file in bed12 format
    logger.info('Writing output headers')
    handle = open(os.path.join(output_directory, 'inferred_CDS.bed'), 'w')
    towrite = ["chromosome", "start", "stop", "transcript_id", 
               "posterior", "strand", "cdstart", "cdstop", 
               "protein_seq", "num_exons", "exon_sizes", "exon_starts", "frame"]
    handle.write(" ".join(map(str, towrite))+'\n')

    from ribohmm.contrib.load_data import read_annotations, Transcript
    # genome_track.get_sequence(list(transcript_models.values()), add_seq_to_transcript=True)
    # start_codon_annotations, stop_codon_annotations = read_annotations()
    # print('********** We are computing all ***********')
    # Transcript.compute_all(
    #     [t for t in transcript_models.values()],
    #     start_annotations=start_codon_annotations,
    #     stop_annotations=stop_codon_annotations
    # )
    # n_not_found = sum([1 - int(t.annotated_ORF_found) for t in transcript_models.values()])
    # total_transcripts = len(transcript_models.values())
    # print('{}/{} not found'.format(n_not_found, total_transcripts))


    # Process 1000 transcripts at a time
    debug_metadata = defaultdict(list)
    records_to_write = list()
    futs = list()
    with ProcessPoolExecutor(max_workers=max(1, n_procs)) as executor:
        for n in range(int(np.ceil(len(transcript_names)/n_transcripts_per_proc))):
            logger.info('Processing transcripts {}-{}'.format(n * n_transcripts_per_proc, (n + 1) * n_transcripts_per_proc))
            tnames: List[str] = transcript_names[n*n_transcripts_per_proc:(n+1)*n_transcripts_per_proc]
            transcripts_chunk: List[Transcript] = [transcript_models[name] for name in tnames]

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
    for fut in futs:
        records_to_write_, debug_metadata_ = fut.result()
        # print(f'Got {len(records_to_write_)} records to write')
        records_to_write.extend(records_to_write_)
        for d in debug_metadata_:
            debug_metadata[d['transcript_info']['strand']].append(d)
        # if debug_metadata_:
        #     pos_debug_metadata = [d for d in debug_metadata_ if d['transcript_info']]
        #     debug_metadata[infer_strand].extend(debug_metadata_)

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
            # print('********************')
            print('number of pos: {}'.format(len(debug_metadata['+'])))
            print('number of neg: {}'.format(len(debug_metadata['-'])))
            json.dump(serialize_output({'pos': debug_metadata['+'], 'neg': debug_metadata['-']}), out)

    # Uncomment the below if we ever come back to analyzing annotated start codon
    # # Some debugging for the debug_metadata object
    # debug_object = serialize_output({'pos': debug_metadata['+'], 'neg': debug_metadata['-']})
    # try:
    #     find_start_codon(debug_object)
    # except Exception as e:
    #     print('Could not run find_start_codon(): {}'.format(e))
    #
    # # from ribohmm.contrib.load_data import read_annotations, Transcript
    # all_transcripts: List[Transcript] = [t for t in transcript_models.values()]
    # genome_track.get_sequence(all_transcripts, add_seq_to_transcript=True)
    # start_codon_annotations, stop_codon_annotations = read_annotations()
    # # print('********** We are finding problematic transcripts ***********')
    # Transcript.compute_all(
    #     all_transcripts,
    #     start_annotations=start_codon_annotations,
    #     stop_annotations=stop_codon_annotations
    # )
    #
    # # Analyze the results
    # n_transcripts = len(all_transcripts)
    # missing_annotation_transcripts = [
    #     t for t in all_transcripts
    #     if not t.annotated_ORF_found
    # ]
    # n_missing_annotation_transcripts = len(missing_annotation_transcripts)
    #
    # print('{}/{} annotated stat pos not found'.format(n_missing_annotation_transcripts, n_transcripts))
    # for t in missing_annotation_transcripts:
    #     print('Name: {} | {} | {} | {} | {}'.format(t.id, t.ref_gene_id, t.ref_transcript_id,
    #                                                 t.raw_attrs.get('reference_id'), t.raw_attrs.get('transcript_id')))
    #     print('{}:{}:{}'.format(t.chromosome, t.start, t.stop))
    #     print('Exons:')
    #     print([(e[0] + t.start, e[1] + t.start) for e in t.exons])
    #     print('Annotated start pos: {}'.format(t.annotated_start_pos))
    #     print('Closest ORF: {}'.format(t.closest_orf))
    #     print('Transcript strand: {}'.format(t.strand))
    #     print(t.get_exonic_sequence(genome_track=genome_track, formatted=True))
    #     for orf in t.orfs:
    #         print(orf)
    #     print('**************************************')
    #
    #
    # n_not_found = sum([1 - int(t.annotated_ORF_found) for t in transcript_models.values()])
    # total_transcripts = len(transcript_models.values())
    # print('{}/{} not found'.format(n_not_found, total_transcripts))

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


def find_start_codon(data, only_show_missing=True):
    n_transcripts_without_start = [0, 0]
    print('Running find_start_codon()')
    for strand in ('pos', 'neg'):
        for trns_i, trns in enumerate(data[strand]):
            has_start_codon = list()
            within_five = list()
            #start_codon_index = 0 if orf['strand'] == '+' else 2
            closest_start = (9e10, None)
            if len(trns['results']['candidate_orf']) == 0:
                print('{} transcript {}/None has no ORFS'.format(strand, trns['transcript_info'].get('id')))
                continue
            for i, orf in enumerate(trns['results']['candidate_orf']):
                start_codon_index = 0 if orf['strand'] == '+' else 2
                transcript_id = orf['transcript_id']
                annotated_start = orf['annotated_start']
                # print('one')
                # print(orf['annotated_start'])
                # print(transcript_id)
                try:
                    distance_to_start = abs(orf['start_codon_genomic_position'][start_codon_index] - orf['annotated_start'])
                except:
                    print('Could not find distance to start for transcript {} ORF {}'.format(trns['transcript_info'].get('id'), i))
                    distance_to_start = 9e10
                if orf['start_codon_genomic_position'][start_codon_index] == orf['annotated_start']:
                    has_start_codon.append(orf['definition'])
                if distance_to_start < 5:
                    within_five.append(orf['definition'])
                if distance_to_start < closest_start[0]:
                    closest_start = (distance_to_start, orf['definition'])
            if len(has_start_codon) > 0:
                if not only_show_missing:
                    print('{} transcript {}/{} has detectable annotated start codon'.format(strand, trns['transcript_info'].get('id'), transcript_id))
            else:
                if strand == 'pos':
                    n_transcripts_without_start[0] += 1
                else:
                    n_transcripts_without_start[1] += 1
                if len(within_five) > 0:
                    print('{} transcript {}/{} has detectable annotated start codon within 5bp (index {})'.format(
                        strand, trns['transcript_info'].get('id'), transcript_id, trns_i)
                    )
                else:
                    print('!!!!!! {} transcript {}/{} is missing the annotated start codon (index {})'.format(
                        strand, trns['transcript_info'].get('id'), transcript_id, trns_i)
                    )
                print('Annotated start: {}'.format(annotated_start))
                print('Exons:')
                print(trns['exons']['absolute'])
                print(trns['transcript_info'])

                # Print whether this is a case where the annotated start corresponds with the start of the transcript
                if trns['transcript_info']['strand'] == '+':
                    if int(annotated_start) == int(trns['transcript_info']['start']):
                        print('** Annotated start detected at beginning of pos transcript **')
                else:  # Is on negative strand
                    if int(annotated_start) == int(trns['transcript_info']['stop']) - 3:
                        print('** Annotated start detected at beginning of neg transcript **')

                print(f'closest start: {closest_start}')
                # print(closest_start)
                print(f'annotated start: {annotated_start}\n')

    print('N pos without start: {}'.format(n_transcripts_without_start[0]))
    print('N neg without start: {}'.format(n_transcripts_without_start[1]))




STARTCODONS = [
    'AUG', 'CUG', 'GUG', 'UUG', 'AAG',
    'ACG', 'AGG', 'AUA', 'AUC', 'AUU'
]
STOPCODONS = ['UAA', 'UAG', 'UGA']

def get_codon_from_seq(seq, frame, triplet_i, stop_i=None):
  stop_i = stop_i or triplet_i
  for i in range(triplet_i, stop_i + 1):
      codon = seq[(3 * i) + frame:(3 * i) + frame + 3]
      if codon in STARTCODONS:
          print(f'{codon} [Start]')
      elif codon in STOPCODONS:
          print(f'{codon} [Stop]')
      else:
          print(f'{codon}')

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
  return ' '.join([seq[s:s+3] for s in range(int(len(seq) / 3) + offset)])


"""
[
  ...
  [0 0 0],
  [0 0 0],
  
  [3 0 0],
  [0 0 0]
]


[
  [0 0 0]
  [0 1 0],
  [0 0 0],
  [0 0 0],
  [0 0 0]
]


ACGTC
GTC AAT CCT CT    CT =/= AUG
CGT C

"""






