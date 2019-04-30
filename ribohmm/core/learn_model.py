from collections import defaultdict

import numpy as np
from ribohmm.core import ribohmm, seq


def select_transcripts(transcript_models_dict, ribo_track, batch_size):
    """
    Select and return top k transcripts based on the transcript translation
    rate, where k = batch_size
    :param transcript_models_dict: dict Transcript models as returned by load_data.load_gtf()
    :param ribo_track: load_data.RiboSeq Model for RiboSeq data
    :param batch_size: int Return this many transcripts
    :return: list of Transcript models
    """
    
    # load all transcripts
    # print('Loading GTF')
    # transcript_models_dict = load_data.load_gtf(transcriptome_gtf)
    """This is a list of load_data.Transcript objects"""
    print('Getting Transcript objects')
    transcript_models = list(transcript_models_dict.values())

    # get translation level in all transcripts
    print('Loading riboseq file')
    # ribo_track = load_data.RiboSeq(riboseq_tabix_prefix, read_lengths)
    """
    For each transcript, divide the total number of counts in the exons by the length of all exons
    """
    print('Calculating transcript translation rate')
    import time
    start = time.time()
    transcript_translation_rate = [
        c / float(t.mask.sum())
        for c, t in zip(ribo_track.get_total_counts(transcript_models), transcript_models)
    ]
    print('{}'.format(time.time() - start))

    # select top transcripts
    transcripts, transcript_bounds = list(), defaultdict(list)
    """Iterate through the load_data.Transcript objects in order from highest 
    transcript_translation_rate to the lowest"""
    print('Selecting top k transcripts')
    for index in reversed(np.argsort(transcript_translation_rate)):
        transcript = transcript_models[index]
 
        # check if all exons have at least 5 footprints
        exon_counts = ribo_track.get_exon_total_counts([transcript])[0]
        if np.any(exon_counts < 5):
            continue

        # check if transcript overlaps any previous transcript
        # filter out strict overlaps
        overlap = False
        try:
            for bound in transcript_bounds[transcript.chromosome]:
                if not (transcript.stop < bound[0] or transcript.start > bound[1]):
                    overlap = True
                    break
        except KeyError:
            pass
        if overlap:
            continue

        transcripts.append(transcript)
        transcript_bounds[transcript.chromosome].append([transcript.start, transcript.stop])

        # select fixed number of transcripts for learning
        if len(transcripts) >= batch_size:
            break

    return transcripts


def learn_model_parameters(genome_track, transcripts, mappability_tabix_prefix, ribo_track,
                           rnaseq_track, scale_beta, restarts, mintol, read_lengths):

    # select transcripts for learning parameters
    # transcripts = select_transcripts(options)
    """Won't T = options.batch always? I guess it could be less"""
    print('{} transcripts selected'.format(len(transcripts)))

    # load sequence of transcripts and transform sequence data
    codon_flags, total_bases = list(), 0

    print('Getting RNAseq sequences of transcripts')
    for i, rna_sequence in enumerate(genome_track.get_sequence(transcripts)):
        try:
            sequence = seq.RnaSequence(rna_sequence)
            codon_flags.append(sequence.mark_codons())
            total_bases += len(rna_sequence)
        except:
            print('Failed on transcript {}'.format(i))
            raise
    print('{} bases covered'.format(total_bases))

    # load footprint count data in transcripts
    footprint_counts = ribo_track.get_counts(transcripts)
    # ribo_track.close()
    for i, read_len in enumerate(read_lengths):
        print('{} ribosome footprints of length {}bp'.format(
            np.sum([c[:, i].sum() for c in footprint_counts]),
            read_len
        ))

    # load transcript-level rnaseq RPKM
    """
    Ensure this is still correct now that the rnaseq path isn't coming from argparse
    """
    if rnaseq_track is None:
        rna_counts = np.ones((len(transcripts),), dtype='float')
    else:
        # rnaseq_track = load_data.RnaSeq(rnaseq_tabix)
        rna_counts = rnaseq_track.get_total_counts(transcripts)
        # rnaseq_track.close()
    print('Median RNA-seq RPKM in data is {:.2f}'.format(np.sum(rna_counts)))

    # load mappability of transcripts; transform mappability to missingness
    if mappability_tabix_prefix is not None:
        rna_mappability = genome_track.get_mappability(transcripts)
    else:
        rna_mappability = [np.ones(c.shape,dtype='bool') for c in footprint_counts]
    # genome_track.close()
    for i,read_len in enumerate(read_lengths):
        print('{} bases have missing counts for {} bp footprints'.format(
            np.sum([
                m.shape[0] - np.sum(m[:, i])
                for m in rna_mappability
            ]),
            read_len
        ))

    # run the learning algorithm
    print('About to run learn_parameters')
    transition, emission, L = ribohmm.learn_parameters(
        footprint_counts,
        codon_flags,
        rna_counts,
        rna_mappability,
        scale_beta,
        mintol,
        read_lengths
    )

    """
    Convert to JSON instead of pickling
    """
    serialized_model = {
        'transition': transition._serialize(),
        'emission': emission._serialize()
    }

    return serialized_model
