import argparse
import datetime
import gzip
import pysam
import numpy as np

from ribohmm.contrib.load_data import load_gtf
from ribohmm.utils import make_complement


def populate_parser(parser):
    parser.add_argument('--gtf-file', help='Path to GTF file containing transcript models')
    parser.add_argument('--fasta-reference', help='FASTA file containing reference genome sequence')
    parser.add_argument('--footprint-length', type=int, default=29,
                        help='Length of ribosome footprint (default: 29)')
    parser.add_argument('--output-fastq', help='Prefix of output fastq file')


def main(args=None):
    if not args:
        parser = argparse.ArgumentParser()
        populate_parser(parser)
        args = vars(parser.parse_args())

    if not args['output_fastq']:
        args['output_fastq'] = '{}_mappability.fq.gz'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )

    # qual = ''.join(['~' for r in range(args['footprint_length'])])
    qual = '~' * args['footprint_length']
    seq_handle = pysam.FastaFile(args['fasta_reference'])

    # load transcripts
    transcripts = load_gtf(args['gtf_file'])
    tnames = transcripts.keys()

    fastq_handle = gzip.open(args['output_fastq'], 'wb')
    for num, tname in enumerate(tnames):

        transcript = transcripts[tname]

        # get transcript DNA sequence
        sequence = seq_handle.fetch(transcript.chromosome, transcript.start, transcript.stop).upper()

        # get forward strand reads
        if transcript.strand == "-":
            transcript.mask = transcript.mask[::-1]
            transcript.strand = "+"

        seq = ''.join(np.array(list(sequence))[transcript.mask].tolist())
        L = len(seq)
        positions = transcript.start + np.where(transcript.mask)[0]
        reads = [seq[i:i + args['footprint_length']]
                 for i in range(L - args['footprint_length'] + 1)]

        # # write synthetic reads
        # s = ''.join(['@{}:{}:{}\n{}\n+\n{}\n'.format(transcript.chromosome, position,transcript.strand,read,qual) for position, read in zip(positions, reads)])
        # fastq_handle.write(s)

        # ["@%s:%d:%s\n%s\n+\n%s\n" %
        #  (transcript.chromosome, position,transcript.strand,read,qual) for position,read in zip(positions,reads)]
        #
        fastq_handle.write(''.join(["@%s:%d:%s\n%s\n+\n%s\n" % (transcript.chromosome, \
                                                                position, transcript.strand, read, qual) \
                                    for position, read in zip(positions, reads)]).encode())

        # get reverse strand reads
        transcript.mask = transcript.mask[::-1]
        transcript.strand = "-"
        seq = seq[::-1]
        seq = ''.join(make_complement(seq))
        positions = transcript.start + transcript.mask.size - np.where(transcript.mask)[0]
        reads = [seq[i:i + args['footprint_length']]
                 for i in range(L - args['footprint_length'] + 1)]

        # write synthetic reads
        fastq_handle.write(''.join(["@%s:%d:%s\n%s\n+\n%s\n" % (transcript.chromosome, \
                                                                position, transcript.strand, read, qual) \
                                    for position, read in zip(positions, reads)]).encode())

    seq_handle.close()
    fastq_handle.close()
