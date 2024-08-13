import os
import argparse
import pdb
import pysam
import subprocess
import numpy as np

from ribohmm.utils import which


def populate_parser(parser):
    parser.add_argument('--mappability-bam', help='Path to aligned BAM for fastq generated with '
                                                  '\'ribohmm mappability-generate\'')
    parser.add_argument('--output-tabix', help='Path to output for mappability tabix file')
    parser.add_argument('--bgzip-path', help='Path to bgzip executable, if not in $PATH')
    parser.add_argument('--tabix-path', help='Path to tabix executable, if not in $PATH')
    parser.add_argument('--min-map-qual', type=int, default=10, help='Minimum mapping quality to consider a read')


def main(args=None):
    if not args:
        parser = argparse.ArgumentParser()
        populate_parser(parser)
        args = vars(parser.parse_args())

    bgzip_path = args['bgzip_path'] or which('bgzip')
    tabix_path = args['tabix_path'] or which('tabix')

    # Assert that bgzip and tabix executables are both available
    if bgzip_path is None:
        raise FileNotFoundError('Path to a bgzip executable was not provided and could not be found in PATH')
    if tabix_path is None:
        raise FileNotFoundError('Path to a tabix executable was not provided and could not be found in PATH')

    # file names and handles
    map_file = args['output_tabix']
    if not map_file.endswith('.bed'):
        map_file = f'{map_file}.bed'
    map_handle = open(map_file, 'w')

    # Check for index alongside BAM file, if not then create
    if not os.path.isfile(args['mappability_bam'] + '.bai'):
        print('BAM index does not exist, creating')
        pysam.index(args['mappability_bam'])
    sam_handle = pysam.AlignmentFile(args['mappability_bam'], 'rb')

    for cname, clen in zip(sam_handle.references, sam_handle.lengths):

        # fetch reads in chromosome
        sam_iter = sam_handle.fetch(reference=cname)

        # initialize mappable positions
        mappable_positions = list()
        for read in sam_iter:

            # skip read if unmapped or mapping quality is too low
            if read.is_unmapped or read.mapq < args['min_map_qual']:
                continue

            if not read.is_reverse:
                mapped_site = int(read.positions[0])
                true_chrom, true_site = read.query_name.split(':')[:2]
                if read.reference_name == true_chrom and mapped_site == int(true_site):
                    mappable_positions.append(mapped_site)

        if len(mappable_positions) > 0:

            # get boundaries of mappable portions of the genome
            mappable_positions = np.sort(mappable_positions)

            boundaries = mappable_positions[:-1] - mappable_positions[1:]
            indices = np.where(boundaries < -1)[0]
            ends = (mappable_positions[indices] + 1).tolist()
            try:
                ends.append(mappable_positions[-1] + 1)
            except IndexError:
                pdb.set_trace()

            boundaries = mappable_positions[1:] - mappable_positions[:-1]
            indices = np.where(boundaries > 1)[0] + 1
            starts = mappable_positions[indices].tolist()
            starts.insert(0, mappable_positions[0])

            # write to file
            for start, end in zip(starts, ends):
                map_handle.write('\t'.join([cname, str(start), str(end)]) + '\n')

        print('Completed {}'.format(cname))

    sam_handle.close()
    map_handle.close()

    # compress count file
    pipe = subprocess.Popen('{} -f {}'.format(bgzip_path, map_file), stdout=subprocess.PIPE, shell=True)
    stdout = pipe.communicate()[0]

    # index count file
    pipe = subprocess.Popen('{} -f -b 2 -e 3 -0 {}.gz'.format(tabix_path, map_file), stdout=subprocess.PIPE, shell=True)
    stdout = pipe.communicate()[0]

    print('Completed computing mappability from BAM file {}'.format(args['mappability_bam']))
