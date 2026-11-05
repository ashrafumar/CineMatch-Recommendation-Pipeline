"""CineMatch Stage 3 checks | Roll Number: 2025AG05135."""
from pathlib import Path
from cinematch.stage3 import SPEC_VERSION, PRIOR_STRENGTH, split_for_timestamp
from cinematch.cli import make_parser
from cinematch.config import PROJECT_ROOT


def test_temporal_boundary_nonoverlap():
    assert [split_for_timestamp(x,100,200) for x in (90,100,101,200,201)] == [
        'train','train','validation','validation','test']


def test_bad_boundaries_rejected():
    import pytest
    with pytest.raises(ValueError):
        split_for_timestamp(100,200,200)


def test_pinned_spec_and_prior():
    assert SPEC_VERSION == 'global_time_80_10_10_v1'
    assert PRIOR_STRENGTH == 10.0


def test_stage3_sql_has_run_version_and_integrity():
    sql=(PROJECT_ROOT / 'sql' / '003_stage3.sql').read_text()
    assert 'CREATE TABLE IF NOT EXISTS features.feature_runs' in sql
    assert 'CREATE OR REPLACE VIEW features.ratings_split' in sql
    assert 'UNIQUE(preparation_id,specification_version,train_end,validation_end)' in sql
    assert 'REFERENCES core.movies(movie_id)' in sql


def test_stage3_cli_commands():
    parser=make_parser()
    for cmd in ('init-stage3','build-features','stage3-status'):
        assert parser.parse_args([cmd]).command == cmd


def test_train_features_have_explicit_cutoff():
    from cinematch import stage3
    import inspect
    source=inspect.getsource(stage3._build_snapshot)
    # All three interaction-derived feature families use a train cutoff.
    assert source.count('rating_timestamp<=%s') >= 4
    assert 'JOIN features.movie_genres' in source
