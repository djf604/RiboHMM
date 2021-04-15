import os
import argparse
import warnings
import json
import datetime

import numpy as np

from ribohmm import core, utils
from ribohmm.core import seq as seq
from ribohmm.core.ribohmm import infer_coding_sequence

import logging
logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%d%b%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger('viterbi_log')

# ignore warnings with these expressions
warnings.filterwarnings('ignore', '.*overflow encountered.*',)
warnings.filterwarnings('ignore', '.*divide by zero.*',)
warnings.filterwarnings('ignore', '.*invalid value.*',)



check_out = ['ENST00000607058.1',
 'ENST00000488123.2',
 'ENST00000540040.1',
 'ENST00000591551.1',
 'ENST00000389680.2',
 'ENST00000387347.2',
 'ENST00000361390.2',
 'ENST00000387405.1',
 'ENST00000387409.1',
 'ENST00000361624.2',
 'ENST00000361739.1',
 'ENST00000362079.2',
 'ENST00000361335.1',
 'ENST00000361381.2',
 'ENST00000387441.1',
 'ENST00000387449.1',
 'ENST00000387456.1',
 'ENST00000361567.2',
 'ENST00000361681.2',
 'ENST00000361789.2',
 'ENST00000387460.2',
 'ENST00000387461.2']

def parse_args():
    parser = argparse.ArgumentParser(description=" infers the translated sequences "
                                     " from ribosome profiling data and RNA sequence data; "
                                    " RNA-seq data can also be used if available ")

    parser.add_argument("--output_file",
                        type=str,
                        default=None,
                        help="output file containing the model parameters")

    parser.add_argument("--rnaseq_file",
                        type=str,
                        default=None,
                        help="prefix of tabix file with counts of RNA-seq reads")

    parser.add_argument("--mappability_file",
                        type=str,
                        default=None,
                        help="prefix of tabix file with mappability information")

    parser.add_argument("model_file",
                        action="store",
                        help="file name containing the model parameters")

    parser.add_argument("fasta_file",
                        action="store",
                        help="fasta file containing the genome sequence")

    parser.add_argument("gtf_file",
                        action="store",
                        help="gtf file containing the assembled transcript models")

    parser.add_argument("riboseq_file",
                        action="store",
                        help="prefix of tabix files with counts of ribosome footprints")

    options = parser.parse_args()

    if options.output_file is None:
        options.output_file = options.model_file+'bed12'

    return options

def write_inferred_cds(handle, transcript, state, frame, rna_sequence):

    posteriors = state.max_posterior*frame.posterior
    index = np.argmax(posteriors)
    tis = state.best_start[index]
    tts = state.best_stop[index]

    # output is not a valid CDS
    if tis is None or tts is None:
        if transcript.id in check_out:
            print('{} write aborted, not valid'.format(transcript.id))
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
               ','.join(map(str,[transcript.start+e[0] for e in transcript.exons]))+',']
    handle.write(" ".join(map(str,towrite))+'\n')

    if transcript.id in check_out:
        print('{} write completed'.format(transcript.id))

    return None

def infer_CDS(model_file, transcript_models, genome_track, mappability_tabix_prefix, ribo_track,
          rnaseq_track, output_directory):
    logger.info('Starting infer_CDS()')

    """
    Load the model from JSON
    """
    model_params = json.load(open(model_file))

    # load transcripts
    transcript_names = list(transcript_models.keys())
    N = len(transcript_names)
    logger.info('Number of transcripts: {}'.format(N))

    # open output file handle
    # file in bed12 format
    logger.info('Writing output headers')
    handle = open(os.path.join(output_directory, 'inferred_CDS.bed'), 'w')
    towrite = ["chromosome", "start", "stop", "transcript_id", 
               "posterior", "strand", "cdstart", "cdstop", 
               "protein_seq", "num_exons", "exon_sizes", "exon_starts"]
    handle.write(" ".join(map(str,towrite))+'\n')

    from collections import Counter, defaultdict
    # Find exon counts for all transcripts, both pos and neg
    alltranscripts = [transcript_models[name] for name in transcript_names]
    for t in alltranscripts:
        if t.strand == '-':
            t.mask = t.mask[::-1]
            t.strand = '+'

    logger.info('Getting total positive exon counts')
    pos_exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
    # pos_transcripts = [t for t, e in zip(alltranscripts, exon_counts) if np.all(e >= 5)]
    for t in alltranscripts:
        t.mask = t.mask[::-1]
        t.strand = '-'
    logger.info('Getting total negative exon counts')
    neg_exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
    # neg_transcripts = [t for t, e in zip(alltranscripts, exon_counts) if np.all(e >= 5)]

    # Process 1000 transcripts at a time
    for n in range(int(np.ceil(N/1000))):
        logger.info('Processing transcripts {}-{}'.format(n * 1000, (n + 1) * 1000))
        tnames = transcript_names[n*1000:(n+1)*1000]
        alltranscripts = [transcript_models[name] for name in tnames]

        # focus on positive strand
        logger.info('Looking at transcript positive strands')
        for t in alltranscripts:
            if t.strand == '-':
                t.mask = t.mask[::-1]
                t.strand = '+'

        # check if all exons have at least 5 footprints
        exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
        transcripts = [t for t,e in zip(alltranscripts,exon_counts) if np.all(e>=5)]
        logger.info('In {} transcripts, all exons have at least 5 footprints'.format(len(transcripts)))

        T = len(transcripts)
        if T>0:
            # load sequence of transcripts and transform sequence data
            codon_flags = []
            logger.info('Loading RNA sequences')
            rna_sequences = genome_track.get_sequence(transcripts)
            logger.info('Setting codon flags')
            for rna_sequence in rna_sequences:
                sequence = seq.RnaSequence(rna_sequence)
                codon_flags.append(sequence.mark_codons())

            # load footprint count data in transcripts
            logger.info('Getting riboseq footprint counts')
            footprint_counts = ribo_track.get_counts(transcripts)

            # load transcript-level rnaseq RPKM
            logger.info('Loading transcript level RNAseq RPKM')
            if rnaseq_track is None:
                rna_counts = np.ones((T,), dtype='float')
            else:
                rna_counts = rnaseq_track.get_total_counts(transcripts)

            # load mappability of transcripts; transform mappability to missingness
            logger.info('Loading mappability')
            if mappability_tabix_prefix is not None:
                rna_mappability = genome_track.get_mappability(transcripts)
            else:
                rna_mappability = [np.ones(c.shape,dtype='bool') for c in footprint_counts]

            logger.info('Running inference')
            states, frames = infer_coding_sequence(footprint_counts, codon_flags, \
                                                        rna_counts, rna_mappability, model_params['transition'], model_params['emission'])

            logger.info('Writing out inferred CDS')
            for transcript,state,frame,rna_sequence in zip(transcripts,states,frames,rna_sequences):
                write_inferred_cds(handle, transcript, state, frame, rna_sequence)

        # focus on negative strand
        logger.info('Looking at transcript negative strands')
        for t in alltranscripts:
            t.mask = t.mask[::-1]
            t.strand = '-'

        # check if all exons have at least 5 footprints
        exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
        transcripts = [t for t,e in zip(alltranscripts,exon_counts) if np.all(e>=5)]
        logger.info('In {} transcripts, all exons have at least 5 footprints'.format(len(transcripts)))

        T = len(transcripts)
        if T>0:

            # load sequence of transcripts and transform sequence data
            codon_flags = []
            logger.info('Loading RNA sequences')
            rna_sequences = genome_track.get_sequence(transcripts)
            logger.info('Setting codon flags')
            for rna_sequence in rna_sequences:
                sequence = seq.RnaSequence(rna_sequence)
                codon_flags.append(sequence.mark_codons())

            # load footprint count data in transcripts
            logger.info('Getting riboseq footprint counts')
            footprint_counts = ribo_track.get_counts(transcripts)

            # load transcript-level rnaseq RPKM
            logger.info('Loading transcript level RNAseq RPKM')
            if rnaseq_track is None:
                rna_counts = np.ones((T,), dtype='float')
            else:
                rna_counts = rnaseq_track.get_total_counts(transcripts)

            # load mappability of transcripts; transform mappability to missingness
            logger.info('Loading mappability')
            if mappability_tabix_prefix is not None:
                rna_mappability = genome_track.get_mappability(transcripts)
            else:
                rna_mappability = [np.ones(c.shape,dtype='bool') for c in footprint_counts]

            # run the learning algorithm
            # states, frames = ribohmm_pure.infer_coding_sequence(footprint_counts, codon_flags, \
            #                        rna_counts, rna_mappability, transition, emission)
            logger.info('Running inference')
            import pickle
            with open('data.pkl', 'wb') as out:
                pickle.dump({
                    'footprint_counts': footprint_counts,
                    'codon_flags': codon_flags,
                    'rna_counts': rna_counts,
                    'rna_mappability': rna_mappability
                }, out)
            import sys
            sys.exit()
            states, frames = infer_coding_sequence(footprint_counts, codon_flags, \
                                                        rna_counts, rna_mappability, model_params['transition'], model_params['emission'])

            logger.info('Writing out inferred CDS')
            for transcript, state, frame, rna_sequence in zip(transcripts, states, frames, rna_sequences):
                write_inferred_cds(handle, transcript, state, frame, rna_sequence)


    logger.info('Closing handles')
    handle.close()
    ribo_track.close()
    # import json
    # import pickle
    #
    # for k in list(duplicates.keys()):
    #     if len(duplicates[k]) <= 1:
    #         del duplicates[k]
    #
    #
    # with open('duplicates.pkl', 'wb') as dup_pickle:
    #     pickle.dump(duplicates, dup_pickle)

    if rnaseq_track is not None:
        rnaseq_track.close()
    genome_track.close()

    logger.info('Finished')


# if __name__=="__main__":
#
#     options = parse_args()
#
#     infer(options)
