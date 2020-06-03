import argparse
import logging
from argparse import RawDescriptionHelpFormatter

from ribohmm._cmds import (learn_model, infer_cds, learn_infer, mappability_generate, mappability_compute,
                           bam_to_counts)


def execute_from_command_line():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='ribohmm')
    subparsers = parser.add_subparsers()

    # TODO Break this all up into two separate executables: ribohmm and ribohmm-utils, where the former is the main
    # program, and the latter is where the mappability and bam_to_counts functions live
    # This way I can add --learn-only and --infer-only commands to the ribohmm executable without having an something
    # awkward like `ribohmm run` in order to run the main program

    subprograms = [
        # (infer, 'infer', 'Main RiboHMM algorithm for CDS inference'),
        (learn_model, 'learn-model', 'Only learn model parameters'),
        (infer_cds, 'infer-cds', 'Only infer CDS with previously learned model parameters'),
        (learn_infer, 'learn-infer', 'Learn model parameters then immediately infer CDS'),
        (mappability_generate, 'mappability-generate', 'Generate a mappability FASTQ for alignment'),
        (mappability_compute, 'mappability-compute', 'Compute mappability from aligned FASTQ '
                                                     'from mappability-generate'),
        (bam_to_counts, 'bam-to-counts', 'Converts one or more BAMs to counts')
        # (None, 'merge-bed', 'None'),
        # (None, 'convert', 'None')
    ]

    for module, name, description in subprograms:
        subp = subparsers.add_parser(
                   name,
                   description=description,
                   formatter_class=RawDescriptionHelpFormatter)
        subp.set_defaults(func=module.main)
        module.populate_parser(subp)

    args = parser.parse_args()
    # TODO catch if there's no sub command
    args.func(vars(args))
