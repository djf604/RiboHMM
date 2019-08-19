import os
import json
import argparse
import logging
import time

import pysam

from ribohmm.contrib.bam_to_tbi import convert_riboseq, convert_rnaseq
from ribohmm.core.learn_model import select_transcripts, learn_model_parameters
from ribohmm.core.infer_CDS import infer_CDS
from ribohmm.core.seq import inflate_kozak_model
from ribohmm.utils import which
from ribohmm.contrib import load_data

logger = logging.getLogger('main')


def common_args(parser):
    parser.add_argument('--reference-fasta', required=True,
                        help='Path to a reference genome in fasta format')
    parser.add_argument('--transcriptome-gtf', required=True,
                        help='Path to a transcriptome in gtf format')
    parser.add_argument('--riboseq-bam', required=True,
                        help='Path to a BAM with riboseq mappings')
    parser.add_argument('--rnaseq-bam', help='Path to a BAM with RNAseq mappings')
    parser.add_argument('--riboseq-bam-index', help='Path to samtools index for Riboseq alignment file; if not '
                                                    'provided, will infer as alignment.bam.bai; if that file does '
                                                    'not exist, pysam will create it')
    parser.add_argument('--rnaseq-bam-index', help='Path to samtools index for RNAseq alignment file; if not '
                                                   'provided, will infer as alignment.bam.bai; if that file does '
                                                   'not exist, pysam will create it')
    parser.add_argument('--mappability-tabix-prefix', help='Path to mappability tabix output by '
                                                           '\'ribohmm mappability-calculate\'')
    parser.add_argument('--log-output', help='Path to file to store statistics of the EM algorithm; not output '
                                             'if no path is given')
    parser.add_argument('--read-lengths', nargs='*', type=int, help='Space separated list of riboseq read lengths '
                                                                    '(default: 28 29 30 31)')
    parser.add_argument('--purge-tabix', action='store_true',
                        help='Do not keep the generated tabix files')
    parser.add_argument('--kozak-model',
                        help='Path to kozak model (included with this package)')
    parser.add_argument('--output-directory', required=True,
                        help='Path prefix for all output: generated tabix files, learned '
                             'model parameters, and final inference output')
    parser.add_argument('--bgzip-path', help='Path to bgzip executable, if not in $PATH')
    parser.add_argument('--tabix-path', help='Path to tabix executable, if not in $PATH')



def learn_args(parser):
    parser.add_argument('--model-parameters-output', help='If provided, path to output for JSON record of '
                                                          'learned model paramters')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of transcripts used for learning model parameters (default: 1000)')
    parser.add_argument('--scale-beta', type=float, default=1e4,
                        help='Scaling factor for initial precision values (default: 1e4)')
    parser.add_argument('--min-tolerence', type=float, default=1e-4,
                        help='Convergence criterion for change in per-base marginal likelihood (default: 1e-4)')
    parser.add_argument('--restarts', type=int, default=1,
                        help='Number of re-runs of the algorithm (default: 1)')


def infer_args(parser):
    parser.add_argument('--model-parameters', required=True,
                        help='Path to JSON file of model parameters generated with '
                             '\'ribohmm learn-model\'')


def execute_ribohmm(args=None, learn=True, infer=True):
    if not args:
        parser = argparse.ArgumentParser()
        common_args(parser)
        if learn:
            learn_args(parser)
        if infer and not learn:
            infer_args(parser)
        args = vars(parser.parse_args())

    # Use provided executable paths or try to infer them from user's $PATH
    bgzip_path = args['bgzip_path'] or which('bgzip')
    tabix_path = args['tabix_path'] or which('tabix')

    # Set default read lengths if not provided, or remove duplicates and sort if provided
    if not args['read_lengths']:
        args['read_lengths'] = [28, 29, 30, 31]
    else:
        args['read_lengths'] = sorted(set(args['read_lengths']))

    # Validate path to Kozak model, then inflate
    if args['kozak_model'] is not None and not os.path.isfile(args['kozak_model']):
        raise ValueError('Path to kozak model ({}) is invalid'.format(args['kozak_model']))
    inflate_kozak_model(args['kozak_model'])

    # Verify or generate BAM file indices
    if not args['riboseq_bam_index'] or not os.path.isfile(args['riboseq_bam_index']):
        if not os.path.isfile(args['riboseq_bam'] + '.bai'):
            print('Generating index for {}'.format(args['riboseq_bam']))
            pysam.index(args['riboseq_bam'])
    if args['rnaseq_bam'] and (not args['rnaseq_bam_index'] or not os.path.isfile(args['rnaseq_bam_index'])):
        if not os.path.isfile(args['rnaseq_bam'] + '.bai'):
            print('Generating index for {}'.format(args['rnaseq_bam']))
            pysam.index(args['rnaseq_bam'])

    # Convert riboseq BAM to tabix
    start = time.time()
    riboseq_tabix_prefix = convert_riboseq(
        bam_file=args['riboseq_bam'],
        output_directory=args['output_directory'],
        bgzip_path=bgzip_path,
        tabix_path=tabix_path,
        read_lengths=args['read_lengths']
    )
    logger.debug('convert_riboseq_to_tabix:{}'.format(time.time() - start))

    # Convert RNAseq BAM to tabix
    rnaseq_tabix = None
    if args['rnaseq_bam']:
        start = time.time()
        rnaseq_tabix = convert_rnaseq(
            bam_file=args['rnaseq_bam'],
            output_directory=args['output_directory'],
            bgzip_path=bgzip_path,
            tabix_path=tabix_path
        )
        logger.debug('convert_rnaseq_to_tabix:{}'.format(time.time() - start))


    # Generate major objects once
    print('\n######\nCreating biological models\n######')
    print('Inflating genome model')
    start = time.time()
    genome_track = load_data.Genome(args['reference_fasta'], args['mappability_tabix_prefix'], args['read_lengths'])
    logger.debug('inflate_genome_track:{}'.format(time.time() - start))
    print('Inflating transcript models ')
    start = time.time()
    gtf_model = load_data.load_gtf(args['transcriptome_gtf'])
    logger.debug('inflate_transcriptome:{}'.format(time.time() - start))
    print('Inflating riboseq model')
    start = time.time()
    ribo_track = load_data.RiboSeq(riboseq_tabix_prefix, args['read_lengths'])
    logger.debug('inflate_riboseq_track:{}'.format(time.time() - start))
    rnaseq_track = None
    if rnaseq_tabix:
        print('Inflating RNAseq model')
        start = time.time()
        rnaseq_track = load_data.RnaSeq(rnaseq_tabix)
        logger.debug('inflate_rnaseq_track:{}'.format(time.time() - start))

    if learn:
        print('\n######\nStarting to learn model parameters\n######')
        serialized_model = learn_model_parameters(
            genome_track=genome_track,
            transcripts=select_transcripts(
                transcript_models_dict=gtf_model,
                ribo_track=ribo_track,
                batch_size=args['batch_size']
            ),
            mappability_tabix_prefix=args['mappability_tabix_prefix'],
            ribo_track=ribo_track,  # These two were generated above
            rnaseq_track=rnaseq_track,
            scale_beta=args['scale_beta'],
            restarts=args['restarts'],
            mintol=args['min_tolerence'],
            read_lengths=args['read_lengths']
        )

        print('\n######\nWriting out learned model parameters\n######')
        with open(os.path.join(args['output_directory'], 'model_parameters.json'), 'w') as model_file_out:
            model_file_out.write(json.dumps(serialized_model, indent=2) + '\n')

    if infer:
        print('\n######\nInferring coding sequences\n######')
        model_file = args.get('model_parameters', os.path.join(args['output_directory'], 'model_parameters.json'))
        infer_CDS(
            model_file=model_file,
            transcript_models=gtf_model,
            genome_track=genome_track,
            mappability_tabix_prefix=args['mappability_tabix_prefix'],
            ribo_track=ribo_track,
            rnaseq_track=rnaseq_track,
            output_directory=args['output_directory']
        )

    if args['purge_tabix']:
        for ribo_tbx in riboseq_tabix_prefix:
            try:
                os.remove(ribo_tbx)
            except:
                print('Could not remove {}'.format(ribo_tbx))
            try:
                os.remove(ribo_tbx + '.tbi')
            except:
                print('Could not remove {}'.format(ribo_tbx + '.tbi'))
        if rnaseq_tabix:
            try:
                os.remove(rnaseq_tabix)
            except:
                print('Could not remove {}'.format(rnaseq_tabix))
            try:
                os.remove(rnaseq_tabix + '.tbi')
            except:
                print('Could not remove {}'.format(rnaseq_tabix + '.tbi'))
