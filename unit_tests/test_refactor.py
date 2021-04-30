from ribohmm.utils import STARTCODONS, STOPCODONS
from ribohmm.core.seq import RnaSequence, inflate_kozak_model


def test_codon_dict():
    STARTS1 = dict([(s, i + 1) for i, s in enumerate(STARTCODONS)])
    STARTS2 = {codon_name: i for i, codon_name in enumerate(STARTCODONS, start=1)}
    assert STARTS1 == STARTS2


def test_rna_sequence():
    inflate_kozak_model()
    rna_seq = RnaSequence('TTTACTAUGTTGGGGGGGAUGUGAUUU')  # 8 triplets
    # rna_seq = RnaSequence('AUGGGGCCTT')
    start_codons = rna_seq._mark_start_codons()
    assert start_codons[1][0] == 1
    assert start_codons[5][0] == 0

    stop_codons = rna_seq._mark_stop_codons()
    assert stop_codons[5][0] == 3

    kozak_model = rna_seq._compute_kozak_scores()
