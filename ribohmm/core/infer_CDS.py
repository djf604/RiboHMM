import os
import argparse
import warnings
import json
import datetime

import numpy as np

from ribohmm import core, utils
from ribohmm.core import seq as seq
from ribohmm.core.ribohmm import infer_coding_sequence

# ignore warnings with these expressions
warnings.filterwarnings('ignore', '.*overflow encountered.*',)
warnings.filterwarnings('ignore', '.*divide by zero.*',)
warnings.filterwarnings('ignore', '.*invalid value.*',)



check_out = ['ENST00000607058.1',
 'ENST00000488123.2',
 'ENST00000540040.1',
 'ENST00000591551.1',
 'ENST00000389680.2',
 'ENST00000387347.2',
 'ENST00000361390.2',
 'ENST00000387405.1',
 'ENST00000387409.1',
 'ENST00000361624.2',
 'ENST00000361739.1',
 'ENST00000362079.2',
 'ENST00000361335.1',
 'ENST00000361381.2',
 'ENST00000387441.1',
 'ENST00000387449.1',
 'ENST00000387456.1',
 'ENST00000361567.2',
 'ENST00000361681.2',
 'ENST00000361789.2',
 'ENST00000387460.2',
 'ENST00000387461.2']

def parse_args():
    parser = argparse.ArgumentParser(description=" infers the translated sequences "
                                     " from ribosome profiling data and RNA sequence data; "
                                    " RNA-seq data can also be used if available ")

    parser.add_argument("--output_file",
                        type=str,
                        default=None,
                        help="output file containing the model parameters")

    parser.add_argument("--rnaseq_file",
                        type=str,
                        default=None,
                        help="prefix of tabix file with counts of RNA-seq reads")

    parser.add_argument("--mappability_file",
                        type=str,
                        default=None,
                        help="prefix of tabix file with mappability information")

    parser.add_argument("model_file",
                        action="store",
                        help="file name containing the model parameters")

    parser.add_argument("fasta_file",
                        action="store",
                        help="fasta file containing the genome sequence")

    parser.add_argument("gtf_file",
                        action="store",
                        help="gtf file containing the assembled transcript models")

    parser.add_argument("riboseq_file",
                        action="store",
                        help="prefix of tabix files with counts of ribosome footprints")

    options = parser.parse_args()

    if options.output_file is None:
        options.output_file = options.model_file+'bed12'

    return options

def write_inferred_cds(handle, transcript, state, frame, rna_sequence):

    posteriors = state.max_posterior*frame.posterior
    index = np.argmax(posteriors)
    tis = state.best_start[index]
    tts = state.best_stop[index]

    # output is not a valid CDS
    if tis is None or tts is None:
        if transcript.id in check_out:
            print('{} write aborted, not valid'.format(transcript.id))
        return None

    posterior = int(posteriors[index]*10000) 
    protein = utils.translate(rna_sequence[tis:tts])
    # identify TIS and TTS in genomic coordinates
    if transcript.strand=='+':
        cdstart = transcript.start + np.where(transcript.mask)[0][tis]
        cdstop = transcript.start + np.where(transcript.mask)[0][tts]
    else:
        cdstart = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tts]
        cdstop = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tis]

    towrite = [transcript.chromosome, 
               transcript.start, 
               transcript.stop, 
               transcript.id, 
               posterior, 
               transcript.strand, 
               cdstart, 
               cdstop,
               protein, 
               len(transcript.exons), 
               ','.join(map(str,[e[1]-e[0] for e in transcript.exons]))+',', 
               ','.join(map(str,[transcript.start+e[0] for e in transcript.exons]))+',']
    handle.write(" ".join(map(str,towrite))+'\n')

    if transcript.id in check_out:
        print('{} write completed'.format(transcript.id))

    return None

def infer_CDS(model_file, transcript_models, genome_track, mappability_tabix_prefix, ribo_track,
          rnaseq_track, output_directory):

    # load the model
    # handle = open(options.model_file, 'rb')
    # transition = pickle.load(handle)
    # emission = pickle.load(handle)
    # handle.close()

    """
    Load the model from JSON
    """
    model_params = json.load(open(model_file))

    # load transcripts
    # transcript_models = load_data.load_gtf(transcriptome_gtf)
    transcript_names = list(transcript_models.keys())
    N = len(transcript_names)
    n = int(np.ceil(N/1000))
    print('N: {}'.format(N))
    print('n: {}'.format(n))
    # with open('transcript_names.out', 'w') as tout:
    #     tout.write('\n'.join(transcript_names) + '\n')
    
    # load data tracks
    # genome_track = load_data.Genome(genome_fasta, mappability_tabix_prefix)
    # ribo_track = load_data.RiboSeq(riboseq_tabix_prefix, read_lengths)
    # if rnaseq_tabix is not None:
    #     rnaseq_track = load_data.RnaSeq(rnaseq_tabix)

    # open output file handle
    # file in bed12 format
    handle = open(os.path.join(output_directory, 'inferred_CDS.bed'), 'w')
    towrite = ["chromosome", "start", "stop", "transcript_id", 
               "posterior", "strand", "cdstart", "cdstop", 
               "protein_seq", "num_exons", "exon_sizes", "exon_starts"]
    handle.write(" ".join(map(str,towrite))+'\n')

    from collections import Counter, defaultdict
    written_out = Counter()
    duplicates = defaultdict(list)

    dups = {'ENST00000361567.2',
             'ENST00000591551.1',
             'ENST00000540040.1',
             'ENST00000361681.2',
             'ENST00000361624.2',
             'ENST00000387347.2',
            'ENST00000565981.1',
            'ENST00000472787.1'}


    good_transcript = 'ENST00000565981.1'
    bad_transcript = 'ENST00000540040.1'



    check_out = ['ENST00000607058.1',
 'ENST00000488123.2',
 'ENST00000540040.1',
 'ENST00000591551.1',
 'ENST00000389680.2',
 'ENST00000387347.2',
 'ENST00000361390.2',
 'ENST00000387405.1',
 'ENST00000387409.1',
 'ENST00000361624.2',
 'ENST00000361739.1',
 'ENST00000362079.2',
 'ENST00000361335.1',
 'ENST00000361381.2',
 'ENST00000387441.1',
 'ENST00000387449.1',
 'ENST00000387456.1',
 'ENST00000361567.2',
 'ENST00000361681.2',
 'ENST00000361789.2',
 'ENST00000387460.2',
 'ENST00000387461.2']



    # Find exon counts for all transcripts, both pos and neg
    alltranscripts = [transcript_models[name] for name in transcript_names]
    for t in alltranscripts:
        if t.strand == '-':
            t.mask = t.mask[::-1]
            t.strand = '+'
    pos_exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
    # pos_transcripts = [t for t, e in zip(alltranscripts, exon_counts) if np.all(e >= 5)]
    for t in alltranscripts:
        t.mask = t.mask[::-1]
        t.strand = '-'
    neg_exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
    # neg_transcripts = [t for t, e in zip(alltranscripts, exon_counts) if np.all(e >= 5)]

    # with open('exon_counts.csv', 'w') as out:
    #     out.write('transcript_id,pos_exon_count,neg_exon_count\n')
    #     for t, pos_exon, neg_exon in zip(alltranscripts, pos_exon_counts, neg_exon_counts):
    #         out.write(','.join([str(t.id), str(pos_exon).replace('\n', ''), str(neg_exon).replace('\n', '')]) + '\n')
    # print('%%%%%%%%%%%%%%%%%Done Writing%%%%%%%%%%%%%%')




    for n in range(int(np.ceil(N/1000))):

        tnames = transcript_names[n*1000:(n+1)*1000]

        # if good_transcript in tnames:
        #     print('Good transcript made it into this round')
        # if bad_transcript in tnames:
        #     print('Bad transcript in this round')

        # _fast = set(tnames).intersection(dups)
        # if _fast:
        #     for _f in _fast:
        #         print('{} was examined at {}'.format(_f, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))




        # with open('tnames' + str(n), 'w') as out:
        #     out.write('\n'.join(tnames) + '\n')
        alltranscripts = [transcript_models[name] for name in tnames]
        """
        This next line adds a couple duplicates at this step
        """
        # alltranscripts = alltranscripts + alltranscripts[:5]

        # run inference on both strands independently

        # focus on positive strand
        for t in alltranscripts:
            if t.strand=='-':
                t.mask = t.mask[::-1]
                t.strand = '+'

        # check if all exons have at least 5 footprints
        exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
        """
        The duplication MUST be happening on this next line
        It gets all transcripts which have an exon_count greater than 5
        The transcripts themselves come from alltranscripts
        """
        # print('Pos alltranscripts: {}'.format(len(alltranscripts)))
        # print('Pos exon counts: {}'.format(len(exon_counts)))


        transcripts = [t for t,e in zip(alltranscripts,exon_counts) if np.all(e>=5)]


        for c in check_out:
            if c in [t.id for t in transcripts]:
                print('{} made it past pos exon filter'.format(c))

        # in_good, in_bad = False, False
        # if good_transcript in [t.id for t in transcripts]:
        #     in_good = True
        #     print('Good transcript made it past pos exon filter')
        #     print('Len of transcripts: {}'.format(len(transcripts)))
        #
        # if bad_transcript in [t.id for t in transcripts]:
        #     in_bad = True
        #     print('Bad transcript made it past pos exon filter')
        #     print('Len of transcripts: {}'.format(len(transcripts)))

        T = len(transcripts)
        # print('Pos T: {}'.format(T))
        if T>0:

            # load sequence of transcripts and transform sequence data
            codon_flags = []
            rna_sequences = genome_track.get_sequence(transcripts)
            for rna_sequence in rna_sequences:
                sequence = seq.RnaSequence(rna_sequence)
                codon_flags.append(sequence.mark_codons())

            # load footprint count data in transcripts
            footprint_counts = ribo_track.get_counts(transcripts)

            # load transcript-level rnaseq RPKM
            if rnaseq_track is None:
                rna_counts = np.ones((T,), dtype='float')
            else:
                rna_counts = rnaseq_track.get_total_counts(transcripts)

            # load mappability of transcripts; transform mappability to missingness
            if mappability_tabix_prefix is not None:
                rna_mappability = genome_track.get_mappability(transcripts)
            else:
                rna_mappability = [np.ones(c.shape,dtype='bool') for c in footprint_counts]

            # run the learning algorithm
            # states, frames = ribohmm_pure.infer_coding_sequence(footprint_counts, codon_flags, \
            #                        rna_counts, rna_mappability, transition, emission)


            states, frames = infer_coding_sequence(footprint_counts, codon_flags, \
                                                        rna_counts, rna_mappability, model_params['transition'], model_params['emission'])

            # write results
            # ig = [write_inferred_cds(handle, transcript, state, frame, rna_sequence) \
            #       for transcript,state,frame,rna_sequence in zip(transcripts,states,frames,rna_sequences)]

            pos_writes = 0


            # pos_dups = [t.id for t in transcripts]
            # for p in pos_dups:
            #     if p in dups:
            #         print('{} was pos about to be written at {}'.format(p, datetime.datetime.now().strftime(
            #             '%Y-%m-%d %H:%M:%S')))

            # if good_transcript in [t.id for t in transcripts]:
            #     print('Good transcript about to be pos written')
            # if bad_transcript in [t.id for t in transcripts]:
            #     print('Bad transcript about to be pos written')



            for transcript,state,frame,rna_sequence in zip(transcripts,states,frames,rna_sequences):
                if transcript.id in check_out:
                    print('{} is being written out positive'.format(transcript.id))


                duplicates[transcript.id].append(('positive_strand', transcript))
                write_inferred_cds(handle, transcript, state, frame, rna_sequence)
                pos_writes += 1
                written_out[transcript.id] += 1
            # print('Positive strand writes: {}'.format(pos_writes))


        # focus on negative strand
        for t in alltranscripts:
            t.mask = t.mask[::-1]
            t.strand = '-'

        # check if all exons have at least 5 footprints
        exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
        # print('neg alltranscripts: {}'.format(len(alltranscripts)))
        # print('neg exon counts: {}'.format(len(exon_counts)))
        transcripts = [t for t,e in zip(alltranscripts,exon_counts) if np.all(e>=5)]


        for c in check_out:
            if c in [t.id for t in transcripts]:
                print('{} made it past neg exon filter'.format(c))

        # in_good, in_bad = False, False
        # if good_transcript in [t.id for t in transcripts]:
        #     in_good = True
        #     print('Good transcript made it past neg exon filter')
        #     print('Len of transcripts: {}'.format(len(transcripts)))
        #
        # if bad_transcript in [t.id for t in transcripts]:
        #     in_bad = True
        #     print('Bad transcript made it past neg exon filter')
        #     print('Len of transcripts: {}'.format(len(transcripts)))

        T = len(transcripts)
        # print('Neg T: {}'.format(T))
        if T != len(set(transcripts)):
            # print('!!!!!!!!!!!!!MISMATCH')
            # print(transcripts)
            exit()
        if T>0:

            # load sequence of transcripts and transform sequence data
            codon_flags = []
            rna_sequences = genome_track.get_sequence(transcripts)
            for rna_sequence in rna_sequences:
                sequence = seq.RnaSequence(rna_sequence)
                codon_flags.append(sequence.mark_codons())

            # load footprint count data in transcripts
            footprint_counts = ribo_track.get_counts(transcripts)

            # load transcript-level rnaseq RPKM
            if rnaseq_track is None:
                rna_counts = np.ones((T,), dtype='float')
            else:
                rna_counts = rnaseq_track.get_total_counts(transcripts)

            # load mappability of transcripts; transform mappability to missingness
            if mappability_tabix_prefix is not None:
                rna_mappability = genome_track.get_mappability(transcripts)
            else:
                rna_mappability = [np.ones(c.shape,dtype='bool') for c in footprint_counts]

            # run the learning algorithm
            # states, frames = ribohmm_pure.infer_coding_sequence(footprint_counts, codon_flags, \
            #                        rna_counts, rna_mappability, transition, emission)
            states, frames = infer_coding_sequence(footprint_counts, codon_flags, \
                                                        rna_counts, rna_mappability, model_params['transition'], model_params['emission'])

            # write results
            # ig = [write_inferred_cds(handle, transcript, state, frame, rna_sequence) \
            #       for transcript,state,frame,rna_sequence in zip(transcripts,states,frames,rna_sequences)]

            neg_writes = 0
            # print('Len of transcripts: {}'.format(len(transcripts)))
            # print('Len of states: {}'.format(len(states)))
            # print('Len of frames: {}'.format(len(frames)))
            # print('Len of rna_sequences: {}'.format(len(rna_sequences)))

            # neg_dups = [t.id for t in transcripts]
            # for n in neg_dups:
            #     if n in dups:
            #         print('{} was neg about to be written at {}'.format(n, datetime.datetime.now().strftime(
            #             '%Y-%m-%d %H:%M:%S')))

            for transcript, state, frame, rna_sequence in zip(transcripts, states, frames, rna_sequences):
                if transcript.id in check_out:
                    print('{} is being written out negative'.format(transcript.id))
                duplicates[transcript.id].append(('minus_strand', transcript))
                write_inferred_cds(handle, transcript, state, frame, rna_sequence)
                neg_writes += 1
                written_out[transcript.id] += 1
            # print('Neg strand writes: {}'.format(neg_writes))


    handle.close()
    ribo_track.close()
    # import json
    import pickle

    for k in list(duplicates.keys()):
        if len(duplicates[k]) <= 1:
            del duplicates[k]


    with open('duplicates.pkl', 'wb') as dup_pickle:
        pickle.dump(duplicates, dup_pickle)

    if rnaseq_track is not None:
        rnaseq_track.close()
    genome_track.close()


# if __name__=="__main__":
#
#     options = parse_args()
#
#     infer(options)
