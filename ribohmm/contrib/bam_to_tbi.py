import numpy as np
import pysam
import subprocess
import argparse
import os, pdb

from collections import Counter

MIN_MAP_QUAL = 10
BEFORE_EXT = 0


def merge_beds(beds, bedtools_path):
    with open('.combined.bed', 'w') as out:
        subprocess.call(['cat'] + beds, stdout=out)

    with open('.sorted.bed', 'w') as out:
        subprocess.call([bedtools_path, 'sort', '-i', '.combined.bed'], stdout=out)

    with open('combined.bed', 'w') as out:
        subprocess.call([bedtools_path, 'merge', '-d', '-1',
                         '-c', '4', '-o', 'sum'], stdout=out)

    subprocess.call(['rm', '.combined.bed', '.sorted.bed'])


def convert(protocol='riboseq', source='bam', sink='tabix', read_lengths=None):
    """

    :param protocol:
    :param source: {'bam', 'bed'}
    :param sink: {'bed', 'tabix'}
    :param read_lengths:
    :return:
    """
    if protocol == 'riboseq':
        pass
    elif protocol == 'rnaseq':
        pass
    raise ValueError('Protocol is not supported')


def _convert_bam_to_bed(bam_file, output_directory, bgzip_path, read_length=None):
    """
    Providing read_length implies this is a riboseq conversion
    :param bam_file:
    :param output_directory:
    :param bgzip_path:
    :param read_length:
    :return:
    """
    is_riboseq = read_length is not None
    count_file = os.path.basename(os.path.splitext(bam_file)[BEFORE_EXT]) + '.counts.bed'
    os.makedirs(os.path.join(output_directory, 'tabix'), exist_ok=True)  # TODO This is not python3 compatible
    bed_output_path = os.path.join(output_directory, 'tabix', count_file)

    with pysam.AlignmentFile(bam_file, 'rb') as sam_handle, open(bed_output_path) as count_handle:
        counts = Counter()
        for read in sam_handle.fetch(until_eof=True):
            discard_read = (
                read.is_unmapped or
                read.mapping_quality < MIN_MAP_QUAL or
                (is_riboseq and read.query_length != read_length)
            )


            if not discard_read:
                if is_riboseq:
                    asite_index = -13 if read.is_reverse else 12
                    asite = int(read.get_reference_positions()[asite_index])
                    counts[asite] += 1
                else:
                    site = (read.reference_start + read.reference_length - 1
                            if read.is_reverse else read.reference_start)
                    counts[site] += 1




        for cname, clen in zip(sam_handle.references, sam_handle.lengths):
            counts = Counter()
            for read in sam_handle.fetch(reference=cname):
                pass




def convert_rnaseq(bam_file, output_directory, bgzip_path, tabix_path):
    """
    Given a path to an RNAseq bam file, convert to tabix format.
    :param bam_file: str Path to an RNAseq bam file
    :param bgzip_path: str Path to bgzip executable
    :param tabix_path: str Path to tabix executable
    """
    # Assert that bgzip and tabix executables are both available
    if bgzip_path is None:
        raise FileNotFoundError('Path to a bgzip executable was not provided and could not be found in PATH')
    if tabix_path is None:
        raise FileNotFoundError('Path to a tabix executable was not provided and could not be found in PATH')

    count_file = os.path.basename(os.path.splitext(bam_file)[BEFORE_EXT]) + '.counts.bed'
    os.makedirs(os.path.join(output_directory, 'tabix'), exist_ok=True)
    tabix_output_path = os.path.join(output_directory, 'tabix', count_file)
    with pysam.AlignmentFile(bam_file, 'rb') as sam_handle, open(tabix_output_path, 'w') as count_handle:

        for cname, clen in zip(sam_handle.references, sam_handle.lengths):
            # initialize count array
            counts = Counter()
            for read in sam_handle.fetch(reference=cname):

                # skip read if unmapped or if mapping quality is low
                if read.is_unmapped or read.mapq < MIN_MAP_QUAL:
                    continue

                site = read.pos + read.alen - 1 if read.is_reverse else read.pos
                counts[site] += 1

            # write counts to output file
            for i in sorted(counts.keys()):
                count_handle.write('\t'.join([cname, str(i), str(i+1), str(counts[i])]) + '\n')

            print('Completed {}'.format(cname))

    # Compress counts file
    subprocess.call([bgzip_path, '-f', tabix_output_path])

    # Index counts file
    subprocess.call([tabix_path, '-f', '-b', '2', '-e', '3', '-0', '{}.gz'.format(tabix_output_path)])

    print('Compressed file with RNA-seq counts is {}.gz'.format(tabix_output_path))

    return '{}.gz'.format(tabix_output_path)


def convert_riboseq(bam_file, output_directory, bgzip_path, tabix_path, read_lengths):
    """
    Given a path to an Riboseq bam file, convert to tabix format.
    :param bam_file: str Path to an Riboseq bam file
    :param bgzip_path: str Path to bgzip executable
    :param tabix_path: str Path to tabix executable
    """
    # Assert that bgzip and tabix executables are both available
    if bgzip_path is None:
        raise FileNotFoundError('Path to a bgzip executable was not provided and could not be found in PATH')
    if tabix_path is None:
        raise FileNotFoundError('Path to a tabix executable was not provided and could not be found in PATH')

    # file names and handles
    os.makedirs(os.path.join(output_directory, 'tabix'), exist_ok=True)
    count_file_path = os.path.join(output_directory, 'tabix',
                                   os.path.basename(os.path.splitext(bam_file)[BEFORE_EXT]) + '.{}.len{}.counts.bed')
    # rev_count_file = os.path.splitext(bam_file)[BEFORE_EXT] + '_rev.len{}.tbx'
    sam_handle = pysam.AlignmentFile(bam_file, 'rb')
    # fwd_handle = {r: open('{}.{}'.format(fwd_count_file, r), 'w') for r in read_lengths}
    fwd_handle = {r: open(count_file_path.format('fwd', r), 'w') for r in read_lengths}
    # rev_handle = {r: open('{}.{}'.format(rev_count_file, r), 'w') for r in read_lengths}
    rev_handle = {r: open(count_file_path.format('rev', r), 'w') for r in read_lengths}

    for cname, clen in zip(sam_handle.references, sam_handle.lengths):

        # initialize count arrays
        fwd_counts = {r: Counter() for r in read_lengths}
        rev_counts = {r: Counter() for r in read_lengths}

        for read in sam_handle.fetch(reference=cname):

            # skip reads not of the appropriate length, or if unmapped, or if mapping quality is low
            if read.rlen not in read_lengths or read.is_unmapped or read.mapq < MIN_MAP_QUAL:
                continue

            if read.is_reverse:
                asite = int(read.positions[-13])
                rev_counts[read.rlen][asite] += 1
            else:
                asite = int(read.positions[12])
                fwd_counts[read.rlen][asite] += 1

        # write counts to output files
        for r in read_lengths:
            for i in sorted(fwd_counts[r].keys()):
                fwd_handle[r].write('\t'.join([cname, str(i), str(i+1), str(fwd_counts[r][i])]) + '\n')

            for i in sorted(rev_counts[r].keys()):
                rev_handle[r].write('\t'.join([cname, str(i), str(i+1), str(rev_counts[r][i])]) + '\n')

        print('Completed {}'.format(cname))

    sam_handle.close()
    for r in read_lengths:
        fwd_handle[r].close()
        rev_handle[r].close()

    for r in read_lengths:

        # compress count file
        # TODO Add a better error message if either of these programs aren't in path
        subprocess.call([bgzip_path, '-f', count_file_path.format('fwd', r)])
        subprocess.call([bgzip_path, '-f', count_file_path.format('rev', r)])

        subprocess.call([tabix_path, '-f', '-b', '2', '-e', '3', '-0', count_file_path.format('fwd', r) + '.gz'])
        subprocess.call([tabix_path, '-f', '-b', '2', '-e', '3', '-0', count_file_path.format('rev', r) + '.gz'])

        # generated_tabix.append(count_file_path.format('fwd', r) + '.gz')
        # generated_tabix.append(count_file_path.format('rev', r) + '.gz')

        print('Compressed file with ribosome footprint counts '
              'on forward strand is {}.gz'.format(count_file_path.format('fwd', r)))
        print('Compressed file with ribosome footprint counts on '
              'reverse strand is {}.gz'.format(count_file_path.format('rev', r)))

    return os.path.join(output_directory, 'tabix', os.path.splitext(os.path.basename(bam_file))[BEFORE_EXT])
