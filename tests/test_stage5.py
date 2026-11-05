"""CineMatch Stage 5 tests | Roll Number 2025AG05135."""
import numpy as np
import pytest
from cinematch.stage5 import (
    user_metrics, effective_weights, candidate_scores, _truth_sql,
    _validate_weights, Stat, CANDIDATE_WEIGHTS,
)


def test_metrics_perfect():
    r,n,h=user_metrics([2,3,1],{2,3},3)
    assert r==1 and n==1 and h==1


def test_metrics_no_hits():
    assert user_metrics([4,5],{2,3},2)==(0.,0.,0.)


def test_metrics_discount_and_recall():
    r,n,h=user_metrics([9,4,3],{3,4,5},3)
    assert r==pytest.approx(2/3)
    expected=(1/np.log2(3)+1/np.log2(4))/(1+1/np.log2(3)+1/np.log2(4))
    assert n==pytest.approx(expected)
    assert h==1


def test_no_ground_truth_rejected():
    with pytest.raises(ValueError):user_metrics([1,2],set(),10)


def test_weights_adapt_to_cold_start():
    w=effective_weights((.15,.60,.25),False,False)
    assert np.allclose(w,[1,0,0])
    w2=effective_weights((.15,.60,.25),True,False)
    assert np.allclose(w2,[.2,.8,0])


def test_bad_weights_rejected():
    with pytest.raises(ValueError):_validate_weights((.2,.2,.2))
    with pytest.raises(ValueError):effective_weights((-1,2,0),True,True)
    assert all(abs(sum(w)-1)<1e-9 for w in CANDIDATE_WEIGHTS)


def test_split_sql_no_unbounded_label_read():
    for split in ('test','validation'):
        s=_truth_sql(split)
        assert 'rating_timestamp>%s' in s
        assert 'rating_timestamp<=%s' in s
        assert 'rating>=4' in s
    with pytest.raises(ValueError):_truth_sql('train')


def test_stats_coverage():
    s=Stat()
    s.add([1,2,3],{2},3)
    result=s.output(10)
    assert result['evaluated_users']==1
    assert result['catalog_coverage']==pytest.approx(.3)
    assert result['recall_at_k']==1


def test_candidate_scores_unknown_user():
    ctx={'user_ids':np.array([1]),'movie_ids':np.array([10,11]),
         'user_factors':np.ones((1,1),dtype=np.float32),
         'movie_factors':np.ones((2,1),dtype=np.float32),
         'popularity':np.array([1.,.25],dtype=np.float32),
         'movie_genres':np.eye(2,dtype=np.float32),
         'movie_genre_count':np.ones(2),
         'preferences':np.array([[.6,.4]],dtype=np.float32),
         'pref_nonempty':np.array([True])}
    known,has_genre,sc,raw=candidate_scores(ctx,22,np.ones(5),0.)
    assert not known and not has_genre
    assert len(sc['learned_hybrid'])==2
    assert np.all(raw[1]==0.) and np.all(raw[2]==0.)
