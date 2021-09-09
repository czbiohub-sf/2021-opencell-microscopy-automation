import numpy as np


KWARGS = dict(min_otsu_thresh=700, min_num_nuclei=10)


def test_prediction_score(trained_fov_scorer, fov_snap_and_score):
    snap, expected_score = fov_snap_and_score
    result = trained_fov_scorer.score_raw_fov(snap, **KWARGS)
    assert np.round(result.get('score'), 2) == expected_score
    

def test_prediction_high_score(trained_fov_scorer, fov_snaps_high_score):
    for snap in fov_snaps_high_score:
        result = trained_fov_scorer.score_raw_fov(snap, **KWARGS)
        score = result.get('score')
        assert score is not None
        assert score > 0.5


def test_prediction_low_score(trained_fov_scorer, fov_snaps_low_score):
    for snap in fov_snaps_low_score:
        result = trained_fov_scorer.score_raw_fov(snap, **KWARGS)
        score = result.get('score')
        assert score is not None
        assert score < 0


def test_prediction_too_few_nuclei(trained_fov_scorer, fov_snaps_too_few):
    for snap in fov_snaps_too_few:
        result = trained_fov_scorer.score_raw_fov(snap, **KWARGS)
        assert result.get('score') is None
        assert result.get('comment') == 'Too few nuclei in the FOV'


def test_prediction_no_nuclei(trained_fov_scorer, fov_snap_no_nuclei):
    result = trained_fov_scorer.score_raw_fov(fov_snap_no_nuclei, **KWARGS)
    assert result.get('score') is None
    assert result.get('comment') == 'No nuclei in the FOV'


def test_prediction_invalid_input(trained_fov_scorer):

    # the FOV scorer should check that the input image is a 1024x1024 uint16 numpy array
    invalid_inputs = [None, [], np.array([]), np.zeros((1024, 1024), dtype=float)]

    for invalid_input in invalid_inputs:
        result = trained_fov_scorer.score_raw_fov(invalid_input, **KWARGS)
        assert result.get('score') is None
