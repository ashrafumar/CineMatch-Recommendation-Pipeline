"""Assignment-required retrieval and genre-diversity metric checks.

CineMatch | Roll number 2025AG05135.
"""
import numpy as np
import pytest
from cinematch.stage5 import precision_at_k, intra_list_diversity, Stat

def test_precision_at_10_fixed_denominator():
    assert precision_at_k([1, 2, 3], {1, 3}, 10) == pytest.approx(0.2)
    assert precision_at_k([], {1}, 10) == 0.0
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, 0)

def test_diversity_identical_and_disjoint():
    lookup = {10: 0, 11: 1, 12: 2}
    genres = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert intra_list_diversity([10, 11], lookup, genres) == 0.0
    assert intra_list_diversity([10, 12], lookup, genres) == 1.0
    assert intra_list_diversity([10, 11, 12], lookup, genres) == pytest.approx(2/3)

def test_missing_genres_excluded():
    assert intra_list_diversity([1, 2], {1: 0, 2: 1},
                               np.array([[1, 0], [0, 0]])) == 0.0

def test_stats_include_both_assignment_metrics():
    s = Stat()
    s.add([10, 11], {10}, 10, diversity=0.75)
    result = s.output(100)
    assert result['precision_at_k'] == pytest.approx(.1)
    assert result['diversity_at_k'] == pytest.approx(.75)
