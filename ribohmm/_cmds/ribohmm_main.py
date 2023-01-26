import os
import json
import argparse
import logging
import time

import pysam

from ribohmm.contrib.bam_to_tbi import convert_riboseq, convert_rnaseq, convert_bams_to_bed
from ribohmm.core.learn_model import select_transcripts, learn_model_parameters
from ribohmm.core.infer_CDS import infer_CDS
from ribohmm.core.seq import inflate_kozak_model
from ribohmm.utils import which
from ribohmm.contrib import load_data

BEFORE_EXT = 0
logger = logging.getLogger('main')


def populate_parser(parser: argparse.ArgumentParser):
    learn_infer_flag_group = parser.add_argument_group(title='Flow Control')
    learn_infer_flag_mutex = learn_infer_flag_group.add_mutually_exclusive_group()
    learn_infer_flag_mutex.add_argument('--learn-only', action='store_false',
                                        help='If given, will only run the learn step and terminate')
    learn_infer_flag_mutex.add_argument('--infer-only', action='store_false',
                                        help='If given, will accept parameters from a previous run of the learn step '
                                             'and only run the infer step')

    # Common parameters
    common_group = parser.add_argument_group(
        title='Common Arguments',
        description='These arguments apply to both the learn and infer steps'
    )
    # Reference paths
    common_group.add_argument('--reference-fasta', required=True,
                              help='Path to a reference genome in fasta format')
    common_group.add_argument('--transcriptome-gtf', required=True,
                              help='Path to a transcriptome in gtf format')
    common_group.add_argument('--kozak-model',
                              help='Path to kozak model. If not provided, will use the kozak model'
                                   'provided in this package'
                              )

    # Main input data paths
    riboseq_input_mutex = common_group.add_mutually_exclusive_group()
    riboseq_input_mutex.add_argument('--riboseq-bams', nargs='*',
                                     help='Path to a BAM with riboseq mappings')
    riboseq_input_mutex.add_argument('--riboseq-counts-tabix',
                                     help='Point to a RiboSeq counts BED file. See documentation for further details. '
                                          'The counts files are expected to be tabix indexed.'
    )

    rnaseq_input_mutex = common_group.add_mutually_exclusive_group()
    rnaseq_input_mutex.add_argument('--rnaseq-bams', nargs='*', help='Path to a BAM with RNAseq mappings')
    rnaseq_input_mutex.add_argument('--rnaseq-counts-tabix',
                                    help='Point to an RNAseq counts BED file. See documentation for further details. '
                                         'The counts files are expected to be tabix indexed.'
    )
    # TODO Now there may be more than one BAM index
    # common_group.add_argument('--riboseq-bam-index',
    #                           help='Path to samtools index for Riboseq alignment file; if not '
    #                                'provided, will infer as alignment.bam.bai; if that file does '
    #                                'not exist, RiboHMM will create it'
    # )
    # common_group.add_argument('--rnaseq-bam-index',
    #                           help='Path to samtools index for RNAseq alignment file; if not '
    #                                'provided, will infer as alignment.bam.bai; if that file does '
    #                                'not exist, RiboHMM will create it'
    # )

    # Peripheral input data paths
    common_group.add_argument('--mappability-tabix-prefix', help='Path to mappability tabix output by '
                                                           '\'ribohmm mappability-compute\'')

    # Control parameters
    common_group.add_argument('--log-output', help='Path to file to store statistics of the EM algorithm; not output '
                                                   'if no path is given')
    common_group.add_argument('--read-lengths', nargs='*', type=int,
                              help='Space separated list of riboseq read lengths (default: 28 29 30 31)'
    )
    common_group.add_argument('--purge-tabix', action='store_true',
                              help='Do not keep the generated counts and tabix files')
    common_group.add_argument('--output-directory', required=True,
                              help='Path prefix for all output: generated counts and tabix files, learned '
                                   'model parameters, and final inference output'
    )

    # External software paths
    common_group.add_argument('--bgzip-path', help='Path to bgzip executable, if not in $PATH')
    common_group.add_argument('--tabix-path', help='Path to tabix executable, if not in $PATH')

    # Learn parameters
    learn_group = parser.add_argument_group(
        title='Learn Arguments',
        description='These arguments pertain only to the learn step and will be ignored if --infer-only is given'
    )
    learn_group.add_argument('--model-parameters-output',
                             help='If provided, path to output for JSON record of learned model paramters')
    learn_group.add_argument('--batch-size', type=int, default=1000,
                             help='Number of transcripts used for learning model parameters (default: 1000)')
    learn_group.add_argument('--scale-beta', type=float, default=1e4,
                             help='Scaling factor for initial precision values (default: 1e4)')
    learn_group.add_argument('--min-tolerence', type=float, default=1e-4,
                             help='Convergence criterion for change in per-base marginal likelihood (default: 1e-4)')
    learn_group.add_argument('--restarts', type=int, default=1,
                             help='Number of re-runs of the algorithm (default: 1)')

    # Infer parameters
    infer_group = parser.add_argument_group(
        title='Infer Arguments',
        description='These arguments pertain only to the infer step and will be ignored unless --infer-only is given'
    )
    infer_group.add_argument('--model-parameters',
                             help='Path to JSON file of model parameters generated with \'ribohmm learn-model\'')
    # TODO Make the choices from some contstant pick list
    infer_group.add_argument('--infer-algorithm', choices=('viterbi', 'discovery'), default='viterbi')
    infer_group.add_argument('--dev-restrict-transcripts-to', type=int, help=argparse.SUPPRESS)
    infer_group.add_argument('--dev-output-debug-data', help=argparse.SUPPRESS)


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(prog='ribohmm')
        populate_parser(parser)
        args = parser.parse_args()
    # Because the arguments --learn-only and --infer-only are store_false, the below statement works
    # to control which parts of the program run
    execute_ribohmm(args, learn=args['infer_only'], infer=args['learn_only'])


def execute_ribohmm(args, learn=True, infer=True):
    # Use provided executable paths or try to infer them from user's $PATH
    bgzip_path = args['bgzip_path'] or which('bgzip')
    tabix_path = args['tabix_path'] or which('tabix')

    # Set default read lengths if not provided, or remove duplicates and sort if provided
    args['read_lengths'] = sorted(set(args['read_lengths'] or [28, 29, 30, 31]))

    # Validate path to Kozak model, then inflate
    if args['kozak_model'] is not None and not os.path.isfile(args['kozak_model']):
        raise ValueError('Path to kozak model ({}) is invalid'.format(args['kozak_model']))
    inflate_kozak_model(args['kozak_model'])

    riboseq_counts_bed = convert_bams_to_bed(
        bam_files=args['riboseq_bams'],
        output_prefix=os.path.join(
            args['output_directory'],
            os.path.basename(os.path.splitext(args['riboseq_bams'][0])[BEFORE_EXT])
        ),
        read_lengths=args['read_lengths'],
        bgzip_path=bgzip_path,
        tabix_path=tabix_path
    ) if args['riboseq_bams'] else args['riboseq_counts_tabix']
    # Or from the command line arguments

    # Convert RNAseq BAM to tabix
    rnaseq_counts_bed = args['rnaseq_counts_tabix']
    if args['rnaseq_bams']:
        # start = time.time()
        rnaseq_counts_bed = convert_bams_to_bed(
            bam_files=args['rnaseq_bams'],
            output_prefix=os.path.join(
                args['output_directory'],
                os.path.basename(os.path.splitext(args['rnaseq_bams'][0])[BEFORE_EXT])
            ),
            read_lengths=None,
            bgzip_path=bgzip_path,
            tabix_path=tabix_path
        )


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
    ribo_track = load_data.RiboSeq(riboseq_counts_bed, args['read_lengths'])
    logger.debug('inflate_riboseq_track:{}'.format(time.time() - start))
    rnaseq_track = None
    if rnaseq_counts_bed:
        print('Inflating RNAseq model')
        start = time.time()
        rnaseq_track = load_data.RnaSeq(rnaseq_counts_bed)
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
        os.makedirs(args['output_directory'], exist_ok=True)
        with open(os.path.join(args['output_directory'], 'model_parameters.json'), 'w') as model_file_out:
            model_file_out.write(json.dumps(serialized_model, indent=2) + '\n')

    if infer:
        print('\n######\nInferring coding sequences\n######')
        model_file = args.get('model_parameters') or os.path.join(args['output_directory'], 'model_parameters.json')
        infer_CDS(
            model_file=model_file,
            transcript_models=gtf_model,
            genome_track=genome_track,
            mappability_tabix_prefix=args['mappability_tabix_prefix'],
            ribo_track=ribo_track,
            rnaseq_track=rnaseq_track,
            output_directory=args['output_directory'],
            infer_algorithm=args['infer_algorithm'],
            dev_restrict_transcripts_to=args['dev_restrict_transcripts_to'],
            dev_output_debug_data=args['dev_output_debug_data']
        )

    if args['purge_tabix']:
        if riboseq_counts_bed:
            try:
                os.remove(riboseq_counts_bed)
            except:
                print('Could not remove {}'.format(riboseq_counts_bed))
            try:
                os.remove(riboseq_counts_bed + '.tbi')
            except:
                print('Could not remove {}'.format(riboseq_counts_bed + '.tbi'))
        if rnaseq_counts_bed:
            try:
                os.remove(rnaseq_counts_bed)
            except:
                print('Could not remove {}'.format(rnaseq_counts_bed))
            try:
                os.remove(rnaseq_counts_bed + '.tbi')
            except:
                print('Could not remove {}'.format(rnaseq_counts_bed + '.tbi'))
