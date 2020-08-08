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


def write_fasta(fastq_handle, sequence, transcript, footprint_length, exon_mask=None, strand=None):
    """
    Writes out transcript sequences to fasta for later alignment.

    If ``strand`` is given, the transcript will be treated as if it's on that strand regardless of the value of
    ``transcript.strand``. Same for ``exon_mask`` as a replacement for ``transcript.mask``.

    :param fastq_handle:
    :param sequence:
    :param transcript:
    :param footprint_length:
    :param exon_mask:
    :param strand:
    :return:
    """
    exon_mask = exon_mask or transcript.mask.copy()
    strand = strand or transcript.strand

    # Get the exonic sequence
    exon_seq = ''.join(np.array(list(sequence))[exon_mask])

    # Depending on the transcript strand, flip the exonic sequence and get positions
    if transcript == '-':
        exon_seq = ''.join(make_complement(exon_seq[::-1]))
        positions = transcript.start + transcript.mask.size - np.where(transcript.mask)[0]
    else:
        # If the strand is positive, get these positions
        positions = transcript.start + np.where(exon_mask)[0]

    # Extract reads based on the footprint length
    reads = [
        exon_seq[i:i + footprint_length]
        for i in range(len(exon_seq) - footprint_length + 1)
    ]

    for position, read in zip(positions, reads):
        fastq_handle.write('@{chrom}:{pos}:{strand}\n{read}\n+\n{qual}\n'.format(
            chrom=transcript.chromosome,
            pos=position,
            strand=strand,
            read=read,
            qual='~' * footprint_length
        ).encode())


def main(args=None):
    if not args:
        parser = argparse.ArgumentParser()
        populate_parser(parser)
        args = vars(parser.parse_args())

    print('Starting mappability generate')
    if not args['output_fastq']:
        args['output_fastq'] = '{}_mappability.fq.gz'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )

    # qual = ''.join(['~' for r in range(args['footprint_length'])])
    # qual = '~' * args['footprint_length']
    seq_handle = pysam.FastaFile(args['fasta_reference'])

    # load transcripts
    transcripts = load_gtf(args['gtf_file'])

    fastq_handle = gzip.open(args['output_fastq'], 'wb')
    for num, tname in enumerate(transcripts.keys()):
        transcript = transcripts[tname]

        # get transcript DNA sequence
        sequence = seq_handle.fetch(transcript.chromosome, transcript.start, transcript.stop).upper()

        if transcript.strand == '.':
            write_fasta(fastq_handle, sequence, transcript,
                footprint_length=args['footprint_length'],
                exon_mask=transcript.mask.copy(),
                strand='+'
            )
            write_fasta(fastq_handle, make_complement(sequence), transcript,
                footprint_length=args['footprint_length'],
                exon_mask=transcript.mask[::-1],
                strand='-'
            )
        else:
            write_fasta(fastq_handle, sequence, transcript, footprint_length=args['footprint_length'])

        # get forward strand reads
        #
        #
        #
        #
        #
        # if transcript.strand == "-":
        #     transcript.mask = transcript.mask[::-1]
        #     transcript.strand = "+"
        #
        # exon_seq = ''.join(np.array(list(sequence))[transcript.mask])
        # positions = transcript.start + np.where(transcript.mask)[0]
        # reads = [exon_seq[i:i + args['footprint_length']]
        #          for i in range(len(exon_seq) - args['footprint_length'] + 1)]
        #
        # # # write synthetic reads
        # # s = ''.join(['@{}:{}:{}\n{}\n+\n{}\n'.format(transcript.chromosome, position,transcript.strand,read,qual) for position, read in zip(positions, reads)])
        # # fastq_handle.write(s)
        #
        # # ["@%s:%d:%s\n%s\n+\n%s\n" %
        # #  (transcript.chromosome, position,transcript.strand,read,qual) for position,read in zip(positions,reads)]
        # #
        # for position, read in zip(positions, reads):
        #     fastq_handle.write('@{chrom}:{pos}:{strand}\n{read}\n+\n{qual}\n'.format(
        #         chrom=transcript.chromosome,
        #         pos=position,
        #         strand=transcript.strand,
        #         read=read,
        #         qual='~' * args['footprint_length']
        #     ).encode())
        #
        #
        # # fastq_handle.write(''.join(["@%s:%d:%s\n%s\n+\n%s\n" % (transcript.chromosome, \
        # #                                                         position, transcript.strand, read, qual) \
        # #                             for position, read in zip(positions, reads)]).encode())
        #
        # # get reverse strand reads
        # transcript.mask = transcript.mask[::-1]
        # transcript.strand = "-"
        # seq = exon_seq[::-1]
        # seq = ''.join(make_complement(seq))
        # positions = transcript.start + transcript.mask.size - np.where(transcript.mask)[0]
        # reads = [seq[i:i + args['footprint_length']]
        #          for i in range(len(exon_seq) - args['footprint_length'] + 1)]
        #
        # # write synthetic reads
        # fastq_handle.write(''.join(["@%s:%d:%s\n%s\n+\n%s\n" % (transcript.chromosome, \
        #                                                         position, transcript.strand, read, qual) \
        #                             for position, read in zip(positions, reads)]).encode())

    seq_handle.close()
    fastq_handle.close()
