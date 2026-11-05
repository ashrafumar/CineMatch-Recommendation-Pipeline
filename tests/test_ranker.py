"""CineMatch learned hybrid tests | 2025AG05135."""
import numpy as np
import pytest
from cinematch.ranker import fit_cutoff, feature_vector

def test_cutoff():
    assert fit_cutoff(100,200)==170

def test_bad_cutoff():
    with pytest.raises(ValueError):fit_cutoff(200,100)

def test_features_missing_user():
    x=feature_vector([0.3,0.5],[0,2],[0,0.7],[False,True],[False,True])
    assert x.shape==(2,5)
    assert x[0,1]==0 and x[0,3]==0
    assert 0.8<x[1,1]<1.0

def test_features_finite():
    x=feature_vector(np.array([0.5]),np.array([1e5]),np.array([1.0]),True,True)
    assert np.isfinite(x).all()
