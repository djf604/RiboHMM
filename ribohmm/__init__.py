import sys
import argparse
import logging
from argparse import RawDescriptionHelpFormatter

from ribohmm._cmds import mappability_generate, mappability_compute, bam_to_counts, ribohmm_main


def execute_utils_from_command_line():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='ribohmm-utils')
    subparsers = parser.add_subparsers()

    subprograms = [
        (mappability_generate, 'mappability-generate', 'Generate a mappability FASTQ for alignment'),
        (mappability_compute, 'mappability-compute', 'Compute mappability from aligned FASTQ '
                                                     'from mappability-generate'),
        (bam_to_counts, 'bam-to-counts', 'Converts one or more BAMs to counts')
    ]

    for module, name, description in subprograms:
        subp = subparsers.add_parser(
            name,
            description=description,
            formatter_class=RawDescriptionHelpFormatter)
        subp.set_defaults(func=module.main)
        module.populate_parser(subp)

    args = parser.parse_args()
    if not vars(args):
        args.print_help()
        sys.exit()
    args.func(vars(args))


def execute_from_command_line():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='ribohmm')
    ribohmm_main.populate_parser(parser)
    ribohmm_main.main(args=vars(parser.parse_args()))
