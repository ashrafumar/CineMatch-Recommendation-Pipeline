"""Stage 2 focused unit tests | Roll Number: 2025AG05135."""
from cinematch.stage2 import CHECKS, DISTRIBUTIONS, metric_pass


def test_check_names_unique():
    names = [c.name for c in CHECKS]
    assert len(names) == len(set(names))


def test_expected_counts_and_quality_gates():
    requirements = {c.name: c.expected for c in CHECKS}
    assert requirements['ratings_rows'] == 32_000_204
    assert requirements['movies_rows'] == 87_585
    assert requirements['links_rows'] == 87_585
    assert requirements['duplicate_user_movie_pairs'] == 0
    assert requirements['orphan_ratings'] == 0
    assert requirements['no_genres_listed'] is None


def test_metric_pass():
    assert metric_pass(0, 0)
    assert not metric_pass(2, 0)
    assert metric_pass(900, None)


def test_sql_queries_reference_staging_not_core():
    for check in CHECKS:
        assert 'staging.' in check.sql
        assert 'DELETE' not in check.sql.upper()
        assert 'UPDATE ' not in check.sql.upper()
    assert 'user_history_distribution' in DISTRIBUTIONS


def test_cli_commands_registered():
    from cinematch.cli import make_parser
    parser = make_parser()
    for cmd in ('init-stage2', 'profile', 'prepare', 'stage2-status'):
        assert parser.parse_args([cmd]).command == cmd
