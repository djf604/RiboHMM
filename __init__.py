import argparse
from argparse import RawDescriptionHelpFormatter

from ribohmm._cmds import infer, mappability_generate, mappability_compute


def execute_from_command_line():
    parser = argparse.ArgumentParser(prog='ribohmm')
    subparsers = parser.add_subparsers()

    subprograms = [
        (infer, 'infer', 'Main RiboHMM algorithm for CDS inference'),
        (mappability_generate, 'mappability-generate', 'Generate a mappability FASTQ for alignment'),
        (mappability_compute, 'mappability-compute', 'Compute mappability from aligned FASTQ '
                                                     'from mappability-generate')
    ]

    for module, name, description in subprograms:
        subp = subparsers.add_parser(
                   name,
                   description=description,
                   formatter_class=RawDescriptionHelpFormatter)
        subp.set_defaults(func=module.main)
        module.populate_parser(subp)

    args = parser.parse_args()
    args.func(vars(args))
