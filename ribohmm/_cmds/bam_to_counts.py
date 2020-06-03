import argparse

from ribohmm.contrib.bam_to_tbi import convert_bams_to_bed
from ribohmm.utils import which

def populate_parser(parser):
    parser.add_argument('--bams', required=True, nargs='*', help='One or more BAM file paths to convert to counts')
    parser.add_argument('--bam-type', choices=('riboseq', 'rnaseq'), default='riboseq',
                         help='The type of BAMs that are input')
    parser.add_argument('--read-lengths', nargs='*', type=int, help='Space separated list of riboseq read lengths '
                                                                    '(default: 28 29 30 31)')

    parser.add_argument('--bgzip-path', help='Path to bgzip executable, if not in $PATH')
    parser.add_argument('--tabix-path', help='Path to tabix executable, if not in $PATH')

    parser.add_argument('--output-prefix')


def main(args=None):
    if not args:
        parser = argparse.ArgumentParser()
        populate_parser(parser)
        args = vars(parser.parse_args())

    convert_bams_to_bed(
        bam_files=args['bams'],
        read_lengths=(
            None if args['bam_type'] == 'rnaseq'
            else sorted(set(args['read_lengths'] or [28, 29, 30, 31]))
        ),
        output_prefix=args['output_prefix'],
        bgzip_path=args['bgzip_path'] or which('bgzip'),
        tabix_path=args['tabix_path'] or which('tabix')
    )


