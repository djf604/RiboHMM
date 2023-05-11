import os
import hashlib
import json
import numpy as np
import pandas as pd
import pysam
from functools import reduce

import ribohmm.utils as utils

MIN_MAP_QUAL = 10

import logging
logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] %(message)s',
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

    def get_sequence(self, transcripts):
        """
        :param transcripts:
        :return:
        """
        if self._seq_handle is None:
            self._seq_handle = pysam.FastaFile(self.fasta_filename)
        sequences = []
        for transcript in transcripts:

            # get DNA sequence
            seq = self._seq_handle.fetch(transcript.chromosome, transcript.start, transcript.stop).upper()

            # get DNA sequence of transcript
            # reverse complement, if necessary
            if transcript.strand=="-":
                seq = seq[::-1]
                seq = ''.join(np.array(list(seq))[transcript.mask].tolist())
                seq = utils.make_complement(seq)
            else:
                seq = ''.join(np.array(list(seq))[transcript.mask].tolist())

            # get RNA sequence
            seq = ''.join(['U' if s=='T' else s for s in seq])
            sequences.append(seq)
            
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
            tbx_iters = [handle.fetch(transcript.chromosome, transcript.start, transcript.stop)
                         for handle in self._map_handles]
            if transcript.strand=='+':
                offsets = [1,1,1,1]
            else:
                offsets = self._read_lengths

            for tbx_iter,mappable,offset in zip(tbx_iters,mappables,offsets):

                for tbx in tbx_iter:

                    row = tbx.split('\t')
                    start = int(row[1]) - transcript.start + offset - 1
                    end = int(row[2]) - transcript.start + offset - 1
                    mappable[start:end] = True

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
        """
        if self._counts_tbx is None:
            self._counts_tbx = pysam.TabixFile(self.riboseq_counts_bed)
        read_counts = list()
        for transcript in transcripts:
            tscpt_counts_df = pd.DataFrame(index=range(transcript.mask.shape[0]), columns=self._read_lengths)
            for count_record in self._counts_tbx.fetch(transcript.chromosome, transcript.start, transcript.stop):
                # chrom start stop counts fwd/rev read_length
                chrom, start, stop, asite_count, strandedness, read_length = count_record.split('\t')
                read_length = int(read_length)
                count_pos = int(start) - transcript.start
                # This may change, but for now unknown transcript only have positive reads counted
                if (
                    read_length in self._read_lengths
                    and (
                        (transcript.strand in {'+', '.'} and strandedness == 'fwd')
                        or (transcript.strand == '-' and strandedness == 'rev')
                    )
                ):
                    tscpt_counts_df.loc[count_pos, read_length] = int(asite_count)

            if exon_counts:
                tscpt_exons_counts = tscpt_counts_df.fillna(0).astype(int).values
                read_counts.append(np.array(
                    [tscpt_exons_counts[start:end].sum() for start, end in transcript.exons]
                ))
            else:
                tscpt_counts_df = tscpt_counts_df.fillna(0).astype(int)
                if transcript.strand == '-':
                    tscpt_counts_df = tscpt_counts_df.iloc[::-1]

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

    def get_total_counts(self, transcripts):
        import time
        start = time.perf_counter()
        if self._counts_tbx is None:
            self._counts_tbx = pysam.TabixFile(self.rnaseq_counts_bed)
            # for chrom in self._counts_tbx.contigs:
            #     start0 = time.perf_counter()
            #     # a = (int(record.split('\t')[RnaSeq.PILEUP_COUNT]) for record in self._counts_tbx.fetch(chrom))
            #     logger.info(
            #         'Elapsed time for pileup count: {}'.format(format_elapsed(time.perf_counter() - start0)))
            #     start0 = time.perf_counter()
            #     sum_total = sum((int(record.split('\t')[RnaSeq.PILEUP_COUNT]) for record in self._counts_tbx.fetch(chrom)))
            #     logger.info('Elapsed time for sum: {}, {}'.format(format_elapsed(time.perf_counter() - start0), sum_total))
            #     start0 = time.perf_counter()
            #     r = reduce(
            #         lambda x, y: x + y,
            #         (int(record.split('\t')[RnaSeq.PILEUP_COUNT]) for record in self._counts_tbx.fetch(chrom))
            #     )
            #     self.total += r
            #     logger.info('Elapsed time for reduce: {}, {}'.format(format_elapsed(time.perf_counter() - start0), r))
        logger.info('Elapsed time for loading counts tbx: {}'.format(format_elapsed(time.perf_counter() - start)))
        start_ = time.perf_counter()
        total_counts = list()
        for transcript in transcripts:
            mask = transcript.mask if transcript.strand == '+' else transcript.mask[::-1]
            counts = 0
            for count_record in self._counts_tbx.fetch(transcript.chromosome, transcript.start, transcript.stop):
                chrom, start, stop, asite_count, strandedness, _ = count_record.split('\t')
                count_pos = int(start) - transcript.start
                if mask[count_pos]:
                    counts += int(asite_count)

            total_counts.append(max(1, counts) * 1e6 / (transcript.L * self.total))

        logger.info('Elapsed time for getting counts: {}'.format(format_elapsed(time.perf_counter() - start_)))
        return np.array(total_counts)

    def close(self):
        if self._counts_tbx is not None:
            self._counts_tbx.close()


class Transcript():
    def __init__(self, chrom, start, stop, strand, attrs):
        self.chromosome = chrom if chrom.startswith('c') else 'chr{}'.format(chrom)
        self.start = int(start)
        self.stop = int(stop)
        self.raw_attrs = attrs

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

    def add_exon(self, start, stop):
        self.exons.append((int(start), int(stop)))

    def generate_transcript_model(self):
        """

        :return:
        """

        if self.exons:

            # order exons
            """Sort self.exons::list by start coordinate"""
            order = np.argsort(np.array([e[0] for e in self.exons]))
            self.exons = [[self.exons[o][0],self.exons[o][1]] for o in order]

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


def load_gtf(filename, use_cache=True, cache_dir=None):
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
                return dill.load(cache_in)
        except:
            pass  # Silently fail, the cache does not exist

    transcripts = dict()
    handle = open(filename, "r")

    print('Reading in GTF file')
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

        """
        I think the eval() here is to get rid of surrounding double quotes
        ex. '"ENSG00000223972.4"' => 'ENSG00000223972.4'
        """
        attrs = dict([
            (ln.split()[0], eval(ln.split()[1]))
            for ln in gtf_record[8].split(';')[:-1]
        ])

        # # identify chromosome of the transcript
        # if data[0].startswith('c'):
        #     chrom = data[0]
        # else:
        #     chrom = 'chr%s'%data[0]
        # data[0] = chrom
        chrom = gtf_record[0] if gtf_record[0].startswith('c') else 'chr{}'.format(gtf_record[0])
        start = int(gtf_record[3]) - 1
        stop = int(gtf_record[4])
        strand = gtf_record[6].strip()

        transcript_id = attrs.get('transcript_id')
        if transcript_id in transcripts and gtf_record[2].strip() == 'exon':
            # TODO I think it would be a good idea to have an 'exon cache' and add exons after all main transcript
            # records have been added
            transcripts[transcript_id].add_exon(start, stop)
        elif transcript_id not in transcripts and gtf_record[2].strip() == 'transcript':
            transcripts[transcript_id] = Transcript(
                chrom, start, stop,
                # TODO In a later version we may want to implement a more sophisticated approach, adding two
                # transcripts if the strand is unknown
                strand=strand,
                attrs=attrs
            )
                
    handle.close()

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
