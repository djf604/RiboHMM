import numpy as np
import pysam
import subprocess
import argparse
import os, pdb

from collections import Counter

from utils import READ_LENGTHS

MIN_MAP_QUAL = 10

# def parse_args():
#     parser = argparse.ArgumentParser(description=" convert bam data format to bigWig data format, "
#                                      " for ribosome profiling and RNA-seq data ")
#
#     parser.add_argument("--dtype",
#                         choices=("rnaseq","riboseq"),
#                         default="riboseq",
#                         help="specifies the type of assay (default: riboseq)")
#
#     parser.add_argument("bam_file",
#                         action="store",
#                         help="path to bam input file")
#
#     options = parser.parse_args()
#
#     options.bgzip = which("bgzip")
#     options.tabix = which("tabix")
#
#     return options

def which(program):

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def convert_rnaseq(bam_file, bgzip_path, tabix_path):
    count_file = os.path.splitext(bam_file)[0]
    sam_handle = pysam.AlignmentFile(bam_file, 'rb')
    count_handle = open(count_file, 'w')

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

    sam_handle.close()
    count_handle.close()

    # Compress counts file
    subprocess.call([bgzip_path, '-f', count_file])

    # Index counts file
    subprocess.call([tabix_path, '-f', '-b', '2', '-e', '3', '-0', '{}.gz'.format(count_file)])

    print('Compressed file with RNA-seq counts is {}.gz'.format(count_file))


def convert_riboseq(bam_file, bgzip_path, tabix_path):

    # file names and handles
    fwd_count_file = os.path.splitext(bam_file)[0] + '_fwd'
    rev_count_file = os.path.splitext(bam_file)[0] + '_rev'
    sam_handle = pysam.AlignmentFile(bam_file, 'rb')
    fwd_handle = {r: open('{}.{}'.format(fwd_count_file, r), 'w') for r in READ_LENGTHS}
    rev_handle = {r: open('{}.{}'.format(rev_count_file, r), 'w') for r in READ_LENGTHS}

    for cname, clen in zip(sam_handle.references, sam_handle.lengths):

        # initialize count arrays
        fwd_counts = {r: Counter() for r in READ_LENGTHS}
        rev_counts = {r: Counter() for r in READ_LENGTHS}

        for read in sam_handle.fetch(reference=cname):

            # skip reads not of the appropriate length, or if unmapped, or if mapping quality is low
            if read.rlen not in READ_LENGTHS or read.is_unmapped or read.mapq < MIN_MAP_QUAL:
                continue

            if read.is_reverse:
                asite = int(read.positions[-13])
                rev_counts[read.rlen][asite] += 1
            else:
                asite = int(read.positions[12])
                fwd_counts[read.rlen][asite] += 1

        # write counts to output files
        for r in READ_LENGTHS:
            for i in sorted(fwd_counts[r].keys()):
                fwd_handle[r].write('\t'.join([cname, str(i), str(i+1), str(fwd_counts[r][i])]) + '\n')

            for i in sorted(rev_counts[r].keys()):
                rev_handle[r].write('\t'.join([cname, str(i), str(i+1), str(rev_counts[r][i])]) + '\n')

        print('Completed {}'.format(cname))

    sam_handle.close()
    for r in READ_LENGTHS:
        fwd_handle[r].close()
        rev_handle[r].close()

    for r in READ_LENGTHS:

        # compress count file
        subprocess.call([bgzip_path, '-f', '{}.{}'.format(fwd_count_file, r)])
        subprocess.call([bgzip_path, '-f', '{}.{}'.format(rev_count_file, r)])

        subprocess.call([tabix_path, '-f', '-b', '2', '-e', '3', '-0', '{}.{}.gz'.format(fwd_count_file, r)])
        subprocess.call([tabix_path, '-f', '-b', '2', '-e', '3', '-0', '{}.{}.gz'.format(rev_count_file, r)])

        print('Compressed file with ribosome footprint counts '
              'on forward strand is {}.{}.gz'.format(fwd_count_file, r))
        print('Compressed file with ribosome footprint counts on '
              'reverse strand is {}.{}.gz'.format(rev_count_file, r))
