"""Stage 4 unit tests | CineMatch 2025AG05135."""
import numpy as np
import pytest
from scipy import sparse
from cinematch.stage4 import (positive_only_score,genre_scores,normalized_scores,
                              blend_scores,top_k,make_factors)


def test_positive_threshold():
    assert positive_only_score(3.5)==0
    assert positive_only_score(4.0)==1


def test_genres_missing_fallback():
    m=sparse.csr_matrix([[1,1],[0,0],[0,1]],dtype=np.float32)
    assert np.allclose(genre_scores(m,np.array([0.8,0.2],dtype=np.float32)),[0.5,0,0.2])


def test_normalization_constant():
    assert np.all(normalized_scores(np.array([5,5]))==0)


def test_blend_weight_validation():
    with pytest.raises(ValueError):
        blend_scores(np.array([1,2]),np.array([2,3]),np.array([3,4]),(0.5,0.5,0.5))


def test_seen_excluded_and_ties_deterministic():
    assert top_k([3,1,2],[0.8,0.8,0.9],seen_movie_ids=[2],k=2)==[(1,pytest.approx(0.8)),(3,pytest.approx(0.8))]


def test_svd_sparse_factorization():
    mat=sparse.csr_matrix([[1,1,0],[0,1,1],[1,0,0]],dtype=np.float32)
    u,v=make_factors(mat,rank=2)
    assert u.shape==(3,2) and v.shape==(3,2)
    assert np.isfinite(u).all() and np.isfinite(v).all()
