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
    parser.add_argument('--footprint-lengths', type=int, nargs='*',
                        help='Length of ribosome footprint (default: 28 29 30 31)')
    parser.add_argument('--output-fastq-stub', help='Prefix of output fastq file')


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

    footprint_lengths = args['footprint_lengths'] or [28, 29, 30, 31]

    print('Loading FASTA')
    seq_handle = pysam.FastaFile(args['fasta_reference'])
    # load transcripts
    print('Loading transcripts')
    transcripts = load_gtf(args['gtf_file'])
    print('Loaded {} transcripts'.format(len(transcripts)))
    for footprint_length in footprint_lengths:
        print('Starting mappability generate for footprint length {}'.format(footprint_length))

        output_fastq_stub = '{stub}_footprint{footprint_length}'.format(
            stub=datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if not args['output_fastq_stub'] else args['output_fastq_stub'],
            footprint_length=footprint_length
        )

        fastq_handle = gzip.open(output_fastq_stub + '.fq.gz', 'wb')
        for num, tname in enumerate(transcripts.keys(), start=1):
            if num % 100 == 0:
                print('Written {} transcripts'.format(num))
            transcript = transcripts[tname]

            # get transcript DNA sequence
            sequence = seq_handle.fetch(transcript.chromosome, transcript.start, transcript.stop).upper()

            if transcript.strand == '.':
                write_fasta(fastq_handle, sequence, transcript,
                    footprint_length=footprint_length,
                    exon_mask=transcript.mask.copy(),
                    strand='+'
                )
                write_fasta(fastq_handle, make_complement(sequence), transcript,
                    footprint_length=footprint_length,
                    exon_mask=transcript.mask[::-1],
                    strand='-'
                )
            else:
                write_fasta(fastq_handle, sequence, transcript, footprint_length=footprint_length)
        fastq_handle.close()
    seq_handle.close()
