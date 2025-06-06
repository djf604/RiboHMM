import os
import hashlib
import json
import sys
from typing import Dict, List, Union, Optional
import time

import numpy as np
import pandas as pd
import pysam
from functools import reduce

import ribohmm.utils as utils
from ribohmm.core.ribohmm import Data

MIN_MAP_QUAL = 10

import logging
logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d|%(levelname)s] %(message)s',
    datefmt='%d%b%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger('viterbi_log')

class Genome():
    def __init__(self, fasta_filename, map_filename, read_lengths):
        self._seq_handle = None
        self.fasta_filename = fasta_filename
        self.map_filename = map_filename
        # Mappability
        self._map_handles = None
        self._read_lengths = read_lengths

    def get_sequence(self, transcripts: List['Transcript'], add_seq_to_transcript=False):
        """
        :param transcripts:
        :param add_seq_to_transcript:
        :return:
        """
        if self._seq_handle is None:
            self._seq_handle = pysam.FastaFile(self.fasta_filename)
        sequences = []

        # print('============')
        # print('Start = 2_398_001 and Stop = 2_398_005')
        # short_seq = self._seq_handle.fetch('chr11', 2_398_001, 2_398_005).upper()
        # print(short_seq)
        # print('============')

        for transcript in transcripts:

            # get DNA sequence
            seq = self._seq_handle.fetch(transcript.chromosome, transcript.start, transcript.stop).upper()
            if add_seq_to_transcript:
                transcript.raw_seq = seq
            # print('transcript stop minus start: {}'.format(transcript.stop - transcript.start))
            # print('transcript stop minus start plus 1: {}'.format(transcript.stop - transcript.start + 1))
            # print('len of seq: {}'.format(len(seq)))

            # get DNA sequence of transcript
            # reverse complement, if necessary
            if transcript.strand == '-':
                seq = seq[::-1]
                seq = ''.join(np.array(list(seq))[transcript.mask].tolist())
                seq = utils.make_complement(seq)
            else:
                seq = ''.join(np.array(list(seq))[transcript.mask].tolist())

            # get RNA sequence
            seq = ''.join(['U' if s == 'T' else s for s in seq])
            sequences.append(seq)

            if add_seq_to_transcript:
                transcript.transcribed_seq = seq
            
        return sequences

    def get_mappability(self, transcripts):
        if self._map_handles is None:
            self._map_handles = [pysam.TabixFile(self.map_filename+'_%d.bed.gz'%r)
                                 for r in self._read_lengths]

        mappabilities = []
        for transcript in transcripts:

            # get mappable positions
            mappables = [np.zeros(transcript.mask.shape, dtype='bool')
                         for r in self._read_lengths]
            tbx_iters = [
                handle.fetch(
                    transcript.chromosome,
                    transcript.start - (0 if transcript.strand == '+' else (self._read_lengths[handle_i] - 2)),
                    transcript.stop
                )
                for handle_i, handle in enumerate(self._map_handles)
            ]
            if transcript.strand=='+':
                offsets = [1,1,1,1]
            else:
                offsets = self._read_lengths

            for tbx_iter,mappable,offset in zip(tbx_iters,mappables,offsets):

                for tbx in tbx_iter:

                    row = tbx.split('\t')
                    start = int(row[1]) - transcript.start + offset - 1
                    end = int(row[2]) - transcript.start + offset - 1
                    # Because start can be negative, set it to 0 in that case
                    mappable[max(0, start):end] = True

            if transcript.strand=='+':
                mappables = np.array(mappables).T.astype('bool')
            else:
                mappables = np.array(mappables).T.astype('bool')[::-1]

            mappabilities.append(mappables[transcript.mask,:])

        return mappabilities
            
    def close(self):
        if self._seq_handle is not None:
            self._seq_handle.close()
        if self._map_handles is not None:
            ig = [handle.close() for handle in self._map_handles]

class RiboSeq():

    def __init__(self, riboseq_counts_bed, read_lengths):
        """
        riboseq_counts_bed is expected to be bgzip compressed with a tabix index file

        :param riboseq_counts_bed:
        :param read_lengths:
        """
        self.riboseq_counts_bed = riboseq_counts_bed
        self._counts_tbx = None
        self._read_lengths = read_lengths

    # TODO I think this could be cached to speed up subsequent access
    def get_counts(self, transcripts, exon_counts=False):
        """
        Fetch RiboSeq pileup counts at each base for each given transcript. If exon_counts is True,
        then this will instead be a sum of pileup counts per exon (instead of per base). The returned
        data structure is a list of np.array objects.

        :param exon_counts: bool Whether to return counts per base or per exon
        :param transcripts: list of load_data.Transcript objects
        :return: list np.Array of shape (n_bases|n_exons, n_read_lengths)


        for each transcript:
            counts_df = dataframe where each row is a base position in the transcript and each column is
                        is a read length
            for each count_record in this transcript region:
                count_record = chr, start, stop, count, fwd/rev, read_length
                assert that count_record['read_length'] is one of the RiboSeq object's count lengths
                assert that we only count "fwd" count_records if the transcript is on the + strand OR the strand
                       is unknown (.) and only "rev" count_records if the transcript is on the - strand
                if both above are true:
                    set count_df[base_position, read_length] = count_record['count']

            if we only want the pileup for each exon:
                calculate sum of counts in each exon for this transcript
                report counts for each exon
            else:
                if transcript is on - strand:
                    counts_df = (counts_df with the row order reversed)
                report counts, but only the exonic base positions

        """
        if self._counts_tbx is None:
            self._counts_tbx = pysam.TabixFile(self.riboseq_counts_bed)
        read_counts = list()
        for transcript in transcripts:
            # Create a data frame where the rows represent each base of the transcript and the columns are the
            # read lengths
            # The initial value for all elements in the dataframe is NaN
            tscpt_counts_df = pd.DataFrame(index=range(transcript.mask.shape[0]), columns=self._read_lengths, data=0)

            # Iterate over all the count records in the riboseq BED file
            for count_record in self._counts_tbx.fetch(transcript.chromosome, transcript.start, transcript.stop):
                # Record format is: chrom start stop counts fwd/rev read_length
                chrom, start, stop, asite_count, strandedness, read_length = count_record.split('\t')
                read_length = int(read_length)
                count_pos = int(start) - transcript.start
                # This may change, but for now unknown transcript only have positive reads counted
                # Ensure that:
                # 1) The read length of the record is one of the read lengths of this RiboSeq object
                # 2) If the transcript is on the + strand, only read "fwd" reads. If the transcript is on the -
                #    strand, only read "rev" reads. If the transcript strand is unknown (its value is .) then only
                #    read "fwd" reads.
                if (
                    read_length in self._read_lengths
                    and (
                        (transcript.strand in {'+', '.'} and strandedness == 'fwd')
                        or (transcript.strand == '-' and strandedness == 'rev')
                    )
                ):
                    # If for some reason there were multiple entries in the reads BED file, only the
                    # last count value would persist
                    tscpt_counts_df.loc[count_pos, read_length] = int(asite_count)

            if exon_counts:
                tscpt_exons_counts = tscpt_counts_df.values
                # For each exon, find the sum of all bases within that exon
                read_counts.append(np.array(
                    [tscpt_exons_counts[start:end].sum() for start, end in transcript.exons]
                ))
            else:
                # tscpt_counts_df = tscpt_counts_df.fillna(0).astype(int)
                if transcript.strand == '-':
                    # Reverse the row order of the dataframe
                    tscpt_counts_df = tscpt_counts_df.iloc[::-1]

                # Report counts at each base, but only for exonic bases
                read_counts.append(tscpt_counts_df.loc[transcript.mask.astype(bool)].values)

        return read_counts

    def get_total_counts(self, transcripts):
        """
        Get the total number of RiboSeq reads that fall within each transcript.

        :param transcripts: list of load_data.Transcript objects
        :return: np.array[len(transcripts),]
        """
        return np.array([counts.sum() for counts in self.get_counts(transcripts)])

    def get_exon_total_counts(self, transcripts):
        return self.get_counts(transcripts, exon_counts=True)

    def get_read_lengths(self):
        return self._read_lengths

    def close(self):
        if self._counts_tbx is not None:
            self._counts_tbx.close()


def format_elapsed(elapsed):
    minutes, seconds = divmod(elapsed, 60)
    return '{} minutes, {} seconds'.format(int(minutes), seconds)

class RnaSeq():
    PILEUP_COUNT = 3

    def __init__(self, rnaseq_counts_bed, use_cache=True, cache_dir=None):
        # Counts
        self.rnaseq_counts_bed = rnaseq_counts_bed
        self._counts_tbx = pysam.TabixFile(self.rnaseq_counts_bed)
        self.total = 0

        if use_cache:
            logger.info('Trying to load from cache')
            cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.ribohmm')
            if not os.path.isdir(cache_dir):
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                except:
                    raise OSError(f'Could not create directory {cache_dir}')
            rna_counts_bed_md5 = hashlib.md5(open(rnaseq_counts_bed, 'rb').read()).hexdigest()
            try:
                with open(os.path.join(cache_dir, 'rna_counts.{}.json'.format(rna_counts_bed_md5))) as cache_in:
                    self.total = json.load(cache_in)
                logger.info('Loaded from cache file {}'.format('rna_counts.{}.json'.format(rna_counts_bed_md5)))
            except:
                logger.info('Tried to use cache, but could not find one')
                pass  # Silently fail, the cache does not exist

        if self.total == 0:
            import time
            start = time.perf_counter()
            for chrom in self._counts_tbx.contigs:
                start0 = time.perf_counter()
                # a = (int(record.split('\t')[RnaSeq.PILEUP_COUNT]) for record in self._counts_tbx.fetch(chrom))
                logger.info(
                    'Elapsed time for pileup count: {}'.format(format_elapsed(time.perf_counter() - start0)))
                start0 = time.perf_counter()
                sum_total = sum((int(record.split('\t')[RnaSeq.PILEUP_COUNT]) for record in self._counts_tbx.fetch(chrom)))
                logger.info('Elapsed time for __init__ sum: {}, {}'.format(format_elapsed(time.perf_counter() - start0), sum_total))
                # start0 = time.perf_counter()
                # r = reduce(
                #     lambda x, y: x + y,
                #     (int(record.split('\t')[RnaSeq.PILEUP_COUNT]) for record in self._counts_tbx.fetch(chrom))
                # )
                # self.total += r
                self.total += sum_total
                # logger.info('Elapsed time for reduce: {}, {}'.format(format_elapsed(time.perf_counter() - start0), r))
            logger.info('Elapsed time for __init__ loading counts tbx: {}'.format(format_elapsed(time.perf_counter() - start)))

            if use_cache:
                try:
                    logger.info('Trying to write to the cache')
                    cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.ribohmm')
                    os.makedirs(cache_dir, exist_ok=True)
                    rna_counts_bed_md5 = hashlib.md5(open(rnaseq_counts_bed, 'rb').read()).hexdigest()
                    # transcr_model_md5 = hashlib.md5(open(filename).read().encode()).hexdigest()
                    with open(os.path.join(cache_dir, 'rna_counts.{}.json'.format(rna_counts_bed_md5)), 'w') as cache_out:
                        logger.info('Writing to {}'.format(os.path.join(cache_dir, 'rna_counts.{}.json'.format(rna_counts_bed_md5))))
                        json.dump(self.total, cache_out)
                except:
                    logger.info('Could not write to cache')
                    pass  # Silently fail, this is not an essential feature

        self._counts_tbx = None

    def get_count(self, transcript: 'Transcript'):
        if self._counts_tbx is None:
            self._counts_tbx = pysam.TabixFile(self.rnaseq_counts_bed)
        mask = transcript.mask if transcript.strand == '+' else transcript.mask[::-1]
        counts = 0
        for count_record in self._counts_tbx.fetch(transcript.chromosome, transcript.start, transcript.stop):
            chrom, start, stop, asite_count, strandedness, _ = count_record.split('\t')
            count_pos = int(start) - transcript.start
            if mask[count_pos]:
                counts += int(asite_count)

        transcript.rnaseq_count = counts
        return counts

    def get_total_counts(self, transcripts):
        start_ = time.perf_counter()
        total_counts = list()
        for transcript in transcripts:
            counts = self.get_count(transcript)
            total_counts.append(max(1, counts) * 1e6 / (transcript.L * self.total))

        logger.info('Elapsed time for getting counts: {}'.format(format_elapsed(time.perf_counter() - start_)))
        return np.array(total_counts)

    def close(self):
        if self._counts_tbx is not None:
            self._counts_tbx.close()


class ORF:
    def __init__(self, frame=None, start_triplet=None, stop_triplet=None, start_pos=None, stop_pos=None,
                 start_seq='', stop_seq=''):
        self.frame = frame
        self.start_triplet = start_triplet
        self.stop_triplet = stop_triplet
        self.start_pos = start_pos
        self.stop_pos = stop_pos
        self.start_seq = start_seq
        self.stop_seq = stop_seq
        self.exonic_positions = None
        self.exonic_intervals = None

        # Legacy values to match the namedtuple CandidateCDS
        self.start = self.start_triplet
        self.stop = self.stop_triplet

    def __str__(self):
        return 'Frame: {} | Start: {}/{}/{} | Stop: {}/{}/{}'.format(
            self.frame,
            self.start_triplet, self.start_pos, self.start_seq,
            self.stop_triplet, self.stop_pos, self.stop_seq
        )


class Transcript():
    def __init__(self, chrom, start, stop, strand, attrs):
        self.chromosome = chrom if chrom.startswith('c') else 'chr{}'.format(chrom)
        self.start = int(start)
        self.stop = int(stop)
        self.raw_attrs = attrs
        self.mask = None

        self.strand = strand if strand in {'+', '-'} else '.'

        """
        This puts into attr a dictionary of GTF attributes
        ex.
        ['gene_id "ENSG00000223972.4"',
         ' transcript_id "ENSG00000223972.4"',
         ' gene_type "pseudogene"',
         ' gene_status "KNOWN"',
         ' gene_name "DDX11L1"',
         ' transcript_type "pseudogene"',
         ' transcript_status "KNOWN"',
         ' transcript_name "DDX11L1"',
         ' level 2',
         ' havana_gene "OTTHUMG00000000961.2"']
        """
        if isinstance(attrs, str):
            attrs = dict([
                (ln.split()[0], eval(ln.split()[1]))
                for ln in attrs.split(';')[:-1]
            ])
        assert isinstance(attrs, dict)

        self.id = attrs['transcript_id']
        self.cdstart = None
        self.cdstop = None
        self.exons = list()
        self.has_CDS = False
        self.proteinid = ''

        # add attribute fields that are available
        self.type = attrs.get('transcript_type')
        self.gene_biotype = attrs.get('gene_biotype')
        self.geneid = attrs.get('gene_id')
        self.genename = attrs.get('gene_name')
        self.ref_transcript_id = attrs.get('reference_id')
        self.ref_gene_id = attrs.get('ref_gene_id')
        self.genename = attrs.get('ref_gene_name')

        # For annotated start/stop debugging
        self.raw_seq = None
        self.raw_seq_positions = np.arange(self.start, self.stop)[::-1 if self.strand == '-' else 1]
        self.transcribed_seq = None
        self.transcribed_seq_positions = None
        self.annotated_start_pos = None
        self.annotated_stop_pos = None
        self.orfs: Union[List[ORF], None] = None
        # Outcomes
        self.annotated_ORF_found = False
        self.closest_orf = None
        self.closest_distance = sys.maxsize

        # For core RiboHMM
        self.data_obj: Optional[Data] = None
        self.state_obj = None
        self.frame_obj = None

        # For calculated attribute storage
        self.rnaseq_count = None
        self.orf_data = dict()

    @staticmethod
    def compute_all(transcripts: List['Transcript'], start_annotations: dict, stop_annotations: dict):
        # print('len transcripts: {}'.format(len(transcripts)))
        Transcript.populate_annotated_start_stop(transcripts, start_annotations, stop_annotations)
        Transcript.populate_transcribed_seq_positions(transcripts)
        for t in transcripts:
            try:
                t.compute_all_ORFs()
                t.compute_start_stop_pos()
                t.analyze_ORFs()
            except Exception as e:
                print('Transcript {} could not be analyzed: {}'.format(t.id, str(e)))

    def get_exonic_sequence(self, genome_track: Genome, formatted=False):
        # if self.strand == '-':
        #     exonic_positions = np.arange(self.start, self.stop)[::-1][self.mask]
        # else:
        #     exonic_positions = np.arange(self.start, self.stop)[self.mask]

        absolute_exons = [(e[0] + self.start, e[1] + self.start) for e in self.exons]
        exon_seqs = list()
        for exon in absolute_exons:
            exon_seq = genome_track._seq_handle.fetch(
                self.chromosome,
                exon[0],
                exon[1]
            ).upper()[::(-1 if self.strand == '-' else 1)]
            if self.strand == '-':
                exon_seq = utils.make_complement(exon_seq[::-1])
            exon_seqs.append(exon_seq)

        if not formatted:
            return ''.join(exon_seqs)

        formatted_exon_seqs = list()
        for exon_seq, exon_info in zip(exon_seqs, absolute_exons):
            # exon_start, exon_stop = int(exon_seq[0]), int(exon_seq)
            for pos in range(exon_info[0], exon_info[1], 3):
                seq = exon_seq[(pos - exon_info[0]):(pos - exon_info[0] + 3)]
                formatted_exon_seqs.append('{} {}'.format(pos, seq))

        return '\n'.join(formatted_exon_seqs)


    def analyze_ORFs(self):
        # TODO Need to find the closest ORF
        # print('Transcript is: {}'.format(self.id))
        # print('Annotated start: {}'.format(self.annotated_start_pos))
        annotated_triplet_i = 0 if self.strand == '+' else 2
        # closest_orf, closest_distance = None, sys.maxsize
        for orf_i, orf in enumerate(self.orfs):
            if orf.start_pos[annotated_triplet_i] == self.annotated_start_pos:
                self.annotated_ORF_found = True
                self.closest_orf = orf
                break
            if abs(orf.start_pos[annotated_triplet_i] - self.annotated_start_pos) < abs(self.closest_distance):
                self.closest_orf = orf
                self.closest_distance = orf.start_pos[annotated_triplet_i] - self.annotated_start_pos

        # else:
        #     print('It was never found!')

    def compute_start_stop_pos(self):
        if self.orfs is None:
            raise ValueError('self.orfs cannot be None')

        for orf in self.orfs:
            if self.strand == '-':
                exonic_positions = np.arange(self.start, self.stop)[::-1][self.mask]
                # exonic_positions = np.arange(self.start, self.stop)[self.mask][::-1]
            else:
                exonic_positions = np.arange(self.start, self.stop)[self.mask]
            # Remove initial bases to set the frame
            for _ in range(orf.frame):
                exonic_positions = np.delete(exonic_positions, 0)

            # If needed, add placeholder values to make sequence divisible by 3
            if len(exonic_positions) % 3 in {1, 2}:
                reshaped_exonic_positions = np.append(exonic_positions, [-2] * (3 - (len(exonic_positions) % 3)))

            # Chunk exonic positions into triplets
            # triplet_genomic_positions = np.array(np.split(exonic_positions, 3))
            triplet_genomic_positions = reshaped_exonic_positions.reshape(-1, 3)
            # TODO This is splitting into 3 sets of even size, I want however many chunks all of size 3

            # Get genomic position of start and stop codons
            orf.start_pos = list(triplet_genomic_positions[orf.start_triplet])
            orf.stop_pos = list(triplet_genomic_positions[orf.stop_triplet])

            # Record exonic positions for this ORF
            orf.exonic_positions = exonic_positions[
                (exonic_positions >= min(orf.start_pos + orf.stop_pos))
                & (exonic_positions <= max(orf.start_pos + orf.stop_pos))
            ]

            # Get the exonic intervals
            orf.exonic_intervals = list()
            exonic_pos = orf.exonic_positions[::-1 if self.strand == '-' else 1]
            interval = [exonic_pos[0]]
            for pos in exonic_pos[1:]:
                if pos == interval[-1] + 1:
                    interval.append(pos)
                else:
                    orf.exonic_intervals.append(interval)
                    interval = [pos]
            orf.exonic_intervals.append(interval)

    def compute_all_ORFs(self):
        """
        This method does not use the codon map, rather it looks in the raw sequence.

        Returns:
        """
        if self.transcribed_seq is None:
            raise ValueError('populate_transcribed_seq_positions() must be called before compute_all_ORFs()')

        STARTCODONS = {
            'AUG', 'CUG', 'GUG', 'UUG', 'AAG',
            'ACG', 'AGG', 'AUA', 'AUC', 'AUU'
        }
        STOPCODONS = {'UAA', 'UAG', 'UGA'}
        N_FRAMES = 3
        # n_triplets = local_start_codon_map.shape[0]
        # n_triplets = len(seq)
        candidate_cds = list()

        def codon(pos_i, seq):
            return seq[pos_i * 3:(pos_i + 1) * 3]

        for frame_i in range(N_FRAMES):
            frame_seq = self.transcribed_seq[frame_i:]
            n_triplets = int(len(frame_seq) / 3)
            for pos_i in range(n_triplets):
                if codon(pos_i, frame_seq) in STARTCODONS:
                    for stop_i in range(pos_i + 1, n_triplets):
                        if codon(stop_i, frame_seq) in STOPCODONS:
                            candidate_cds.append(ORF(
                                frame=frame_i,
                                start_triplet=pos_i,
                                stop_triplet=stop_i,
                                start_seq=codon(pos_i, frame_seq),
                                stop_seq=codon(stop_i, frame_seq)
                            ))
                            break

        self.orfs = candidate_cds

    @staticmethod
    def populate_transcribed_seq_positions(transcripts: List['Transcript']):
        for transcript in transcripts:
            if transcript.transcribed_seq is not None and transcript.mask is not None:
                transcript.transcribed_seq_positions = transcript.raw_seq_positions[transcript.mask]

    @staticmethod
    def populate_annotated_start_stop(transcripts: List['Transcript'], start_annotations: dict, stop_annotations: dict):
        for t in transcripts:
            # transcript_id = t.raw_attrs.get('reference_id', t.raw_attrs.get('transcript_id'))
            transcript_id = t.raw_attrs.get('transcript_id')
            t.annotated_start_pos = start_annotations.get(transcript_id)
            t.annotated_stop_pos = stop_annotations.get(transcript_id)

    def triplet_i_to_base_pos(self, frame_i, triplet_i):
        return self.transcribed_seq_positions[triplet_i * 3 + frame_i]

    def add_exon(self, start, stop):
        self.exons.append((int(start), int(stop)))

    def generate_transcript_model(self):
        """

        :return:
        """

        if self.exons:

            # order exons
            """Sort self.exons::list by start coordinate"""
            # order = np.argsort(np.array([e[0] for e in self.exons]))
            # self.exons = [[self.exons[o][0],self.exons[o][1]] for o in order]
            self.exons = sorted(self.exons)

            # extend transcript boundaries, if needed
            self.start = min([self.start, self.exons[0][0]])
            self.stop = max([self.stop, self.exons[-1][-1]])

            # set transcript model
            """Shift all exons by self.start, so the exon model starts at 0 (or at least the transcript does)"""
            self.exons = [(e[0]-self.start, e[1]-self.start) for e in self.exons]
            """Set np.zeros for the length of the transcript"""
            self.mask = np.zeros((self.stop-self.start,), dtype='bool')
            """TODO This is not commented out in the original code"""
            """Equivalent to self.mask[start:stop] = True, except deprecated"""
            """This is just making a mask across the length of the transcript for locations of exons"""
            # ig = [self.mask.__setslice__(start,stop,True) for (start,stop) in self.exons]
            for start, stop in self.exons:
                self.mask[start:stop] = True
            if self.strand=='-':
                self.mask = self.mask[::-1]

            """Total number of nucleotides that are considered part of an exon"""
            self.L = self.mask.sum()

        else:

            # no exons for transcript; remove
            raise ValueError

    def get_seq_slice(self):
        pass


def load_gtf(filename, use_cache=True, cache_dir=None) -> Dict[str, Transcript]:
    """
    Returns a dictionary of transcript_id::str -> Transcript
    :param filename:
    :return:
    """

    # Check cache at ~/.ribohmm
    import os
    import dill
    import hashlib
    if use_cache:
        cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.ribohmm')
        if not os.path.isdir(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except:
                raise OSError(f'Could not create directory {cache_dir}')
        transcr_model_md5 = hashlib.md5(open(filename).read().encode()).hexdigest()
        try:
            with open(os.path.join(cache_dir, 'transcr.{}.dill'.format(transcr_model_md5)), 'rb') as cache_in:
                logger.info('Attemping to load GTF cache from {}'.format('transcr.{}.dill'.format(transcr_model_md5)))
                return dill.load(cache_in)
        except:
            pass  # Silently fail, the cache does not exist

    # Read GTF file
    transcripts, exon_cache = dict(), list()
    print('Reading in GTF file')
    with open(filename, "r") as handle:
        for line in handle:
            # remove comments
            if line.startswith('#'):
                continue

            # read data
            """
            This puts into attr a dictionary of GTF attributes
            ex.
            ['gene_id "ENSG00000223972.4"',
             ' transcript_id "ENSG00000223972.4"',
             ' gene_type "pseudogene"',
             ' gene_status "KNOWN"',
             ' gene_name "DDX11L1"',
             ' transcript_type "pseudogene"',
             ' transcript_status "KNOWN"',
             ' transcript_name "DDX11L1"',
             ' level 2',
             ' havana_gene "OTTHUMG00000000961.2"']
            """
            gtf_record = line.strip().split('\t')
            attrs = dict([
                (ln.split()[0], eval(ln.split()[1]))
                for ln in gtf_record[8].split(';')[:-1]
            ])

            chrom = gtf_record[0] if gtf_record[0].startswith('c') else 'chr{}'.format(gtf_record[0])
            start = int(gtf_record[3]) - 1
            stop = int(gtf_record[4])
            strand = gtf_record[6].strip()

            transcript_id = attrs.get('transcript_id')

            # Add a Transcript or exon to a Transcript
            if gtf_record[2].strip() == 'exon':
                if transcript_id in transcripts:
                    transcripts[transcript_id].add_exon(start, stop)
                else:
                    exon_cache.append((transcript_id, start, stop))
            elif transcript_id not in transcripts and gtf_record[2].strip() == 'transcript':
                transcripts[transcript_id] = Transcript(
                    chrom, start, stop,
                    strand=strand,
                    attrs=attrs
                )

    # Apply exon cache
    for exon_spec in exon_cache:
        transcript_id, start, stop = exon_spec
        transcripts[transcript_id].add_exon(start, stop)

    # generate transcript models
    print('Generating transcript models ({})'.format(len(transcripts)))
    no_exons = list()
    for i, (transcript_id, transcript) in enumerate(transcripts.items(), start=1):
        if i % 1000 == 0:
            print('Processed {}/{}'.format(i, len(transcripts)))
        try:
            transcript.generate_transcript_model()
        except ValueError:
            """If this happens that means there were no exons in the Transcript object"""
            print('There were no exons in Transcript {}'.format(transcript_id))
            no_exons.append(transcript_id)
    for transcript_id_no_exons in no_exons:
        del transcripts[transcript_id_no_exons]

    # Store model in a cache
    if use_cache:
        try:
            # cache_dir = os.path.join(os.path.expanduser('~'), '.ribohmm')
            os.makedirs(cache_dir, exist_ok=True)
            # transcr_model_md5 = hashlib.md5(open(filename).read().encode()).hexdigest()
            with open(os.path.join(cache_dir, 'transcr.{}.dill'.format(transcr_model_md5)), 'wb') as cache_out:
                dill.dump(transcripts, cache_out)
        except:
            pass  # Silently fail, this is not an essential feature

    return transcripts


def read_annotations(annotations_path=None):
    annotations_path = annotations_path or '/home1/08246/dfitzger/riboHMM_chr11_example_YRI_Data/annotated_start_codons.gtf'
    start_codon_annotated = dict()
    stop_codon_annotated = dict()

    try:
        with open(annotations_path) as annot:
            for line in annot:
                try:
                    # record = line.strip().split('\t')
                    transcript_id = line.strip().split("\t")[-1].split(";")[1].strip().split()[1].replace('"', "")
                    offset = 1 if line.strip().split('\t')[1] != 'start_codon' else 0
                    strand = line.strip().split('\t')[4 + offset]
                    start_pos = int(line.strip().split('\t')[2 + offset]) - 1
                    # if strand.strip() == '+':
                    #     start_pos -= 1  # To convert it to 0-based half open
                    stop_pos = int(line.strip().split('\t')[3 + offset])
                    start_codon_annotated[transcript_id] = start_pos
                    stop_codon_annotated[transcript_id] = stop_pos
                except:
                    print('Could not process line: {}'.format(line))
    except:
        print('Could not open file: {}'.format(annotations_path))

    return start_codon_annotated, stop_codon_annotated
