import json
import pickle
import pytest

from ribohmm.core.ribohmm import infer_coding_sequence


def test_states_frames():
    data = pickle.load(open('data.pkl', 'rb'))
    model_params = json.load(open('riboHMM_chr11_example_YRI_Data/model_parameters.json'))

    states, frames = infer_coding_sequence(
        data['footprint_counts'][:5],
        data['codon_flags'][:5],
        data['rna_counts'][:5],
        data['rna_mappability'][:5],
        model_params['transition'],
        model_params['emission']
    )

    # Check frames
    assert frames[0].posterior[0] == pytest.approx(1, rel=1e-2)
    assert frames[1].posterior[1] == pytest.approx(1, rel=1e-2)
    assert frames[2].posterior[0] == pytest.approx(1, rel=1e-2)
    assert frames[3].posterior[1] == pytest.approx(1, rel=1e-2)
    assert frames[4].posterior[1] == pytest.approx(1, rel=1e-2)

    # Check states
    assert states[0].n_triplets == 1492
    assert states[0].n_states == 9
    assert states[0].best_start == [18, 349, 1982]
    assert states[0].best_stop == [36, 394, 2003]
    assert states[0].alpha[0][0][0] == pytest.approx(3.295e-321)
    assert states[0].likelihood[0][0] == pytest.approx(4.68750801e-310)

    assert states[4].n_triplets == 795
    assert states[4].n_states == 9
    assert states[4].best_start == [441, 28, 77]
    assert states[4].best_stop == [468, 43, 104]
    assert states[4].alpha[0][0][0] == pytest.approx(1.7e-322)
    assert states[4].likelihood[0][0] == pytest.approx(4.68750801e-310)
