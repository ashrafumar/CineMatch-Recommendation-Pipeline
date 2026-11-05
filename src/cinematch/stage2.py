"""Stage 2 profiling and canonical preparation. CineMatch | Roll Number: 2025AG05135.

Profiles staging in SQL. Never deletes or changes Stage 1 data. Canonicalization
runs in one transaction and fails on quality-contract breaches, with no sampling.
No NLP, tag analysis, or media processing.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .db import connect
from .config import PROJECT_ROOT
from .verify import EXPECTED_COUNTS, EXPECTED_MD5

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class QualityCheck:
    name: str
    sql: str
    expected: int | None
    description: str
    # None => descriptive statistic; otherwise metric must equal expected.

CHECKS = [
    QualityCheck('ratings_rows', 'SELECT COUNT(*) FROM staging.ratings', EXPECTED_COUNTS['ratings.csv'], 'All rating rows from the source'),
    QualityCheck('movies_rows', 'SELECT COUNT(*) FROM staging.movies', EXPECTED_COUNTS['movies.csv'], 'All movie rows from the source'),
    QualityCheck('links_rows', 'SELECT COUNT(*) FROM staging.links', EXPECTED_COUNTS['links.csv'], 'All link rows from the source'),
    QualityCheck('duplicate_movie_ids', '''SELECT COUNT(*) FROM (SELECT movie_id FROM staging.movies GROUP BY movie_id HAVING COUNT(*) > 1) t''', 0, 'Movie IDs with duplicate records'),
    QualityCheck('duplicate_link_ids', '''SELECT COUNT(*) FROM (SELECT movie_id FROM staging.links GROUP BY movie_id HAVING COUNT(*) > 1) t''', 0, 'Movie IDs with duplicate link records'),
    QualityCheck('duplicate_user_movie_pairs', '''SELECT COUNT(*) FROM (SELECT user_id, movie_id FROM staging.ratings GROUP BY user_id, movie_id HAVING COUNT(*) > 1) t''', 0, 'Repeated user-movie keys (no silent deduplication)'),
    QualityCheck('invalid_rating', '''SELECT COUNT(*) FROM staging.ratings WHERE rating < 0.5 OR rating > 5.0 OR (rating * 2) <> trunc(rating * 2)''', 0, 'Rating must be one of 0.5, 1.0, ..., 5.0'),
    QualityCheck('invalid_rating_ids', '''SELECT COUNT(*) FROM staging.ratings WHERE user_id <= 0 OR movie_id <= 0''', 0, 'User/movie identifiers must be positive'),
    QualityCheck('invalid_rating_timestamp', '''SELECT COUNT(*) FROM staging.ratings WHERE rating_timestamp <= 0''', 0, 'UNIX timestamps must be positive'),
    QualityCheck('invalid_movie_ids', '''SELECT COUNT(*) FROM staging.movies WHERE movie_id <= 0''', 0, 'Movie identifiers must be positive'),
    QualityCheck('blank_movie_titles', "SELECT COUNT(*) FROM staging.movies WHERE btrim(title) = ''", 0, 'Blank movie titles'),
    QualityCheck('orphan_ratings', '''SELECT COUNT(*) FROM staging.ratings r WHERE NOT EXISTS (SELECT 1 FROM staging.movies m WHERE m.movie_id = r.movie_id)''', 0, 'Ratings with missing catalog entry'),
    QualityCheck('orphan_links', '''SELECT COUNT(*) FROM staging.links l WHERE NOT EXISTS (SELECT 1 FROM staging.movies m WHERE m.movie_id = l.movie_id)''', 0, 'Links with missing catalog entry'),
    QualityCheck('missing_link_movies', '''SELECT COUNT(*) FROM staging.movies m WHERE NOT EXISTS (SELECT 1 FROM staging.links l WHERE l.movie_id = m.movie_id)''', 0, 'Movies lacking a links row'),
    QualityCheck('no_genres_listed', "SELECT COUNT(*) FROM staging.movies WHERE genres = '(no genres listed)' OR btrim(genres) = ''", None, 'Movies with no classified genres'),
    QualityCheck('distinct_users', 'SELECT COUNT(DISTINCT user_id) FROM staging.ratings', None, 'Unique rating users'),
    QualityCheck('distinct_rated_movies', 'SELECT COUNT(DISTINCT movie_id) FROM staging.ratings', None, 'Unique rated movies'),
    QualityCheck('min_rating_timestamp', 'SELECT MIN(rating_timestamp) FROM staging.ratings', None, 'Earliest observed rating Unix timestamp'),
    QualityCheck('max_rating_timestamp', 'SELECT MAX(rating_timestamp) FROM staging.ratings', None, 'Latest observed rating Unix timestamp'),
]

DISTRIBUTIONS: dict[str, str] = {
    'ratings_distribution': '''SELECT rating::text AS label, COUNT(*) AS n FROM staging.ratings GROUP BY rating ORDER BY rating''',
    'user_history_distribution': '''WITH c AS (SELECT user_id, COUNT(*) AS n FROM staging.ratings GROUP BY user_id)
         SELECT CASE WHEN n=1 THEN '1' WHEN n<5 THEN '2-4' WHEN n<10 THEN '5-9'
                     WHEN n<20 THEN '10-19' WHEN n<50 THEN '20-49' WHEN n<100 THEN '50-99' ELSE '100+' END AS label,
                COUNT(*) AS n FROM c GROUP BY 1 ORDER BY 1''',
    'movie_history_distribution': '''WITH c AS (SELECT movie_id, COUNT(*) AS n FROM staging.ratings GROUP BY movie_id)
         SELECT CASE WHEN n=1 THEN '1' WHEN n<5 THEN '2-4' WHEN n<10 THEN '5-9'
                     WHEN n<20 THEN '10-19' WHEN n<50 THEN '20-49' WHEN n<100 THEN '50-99' ELSE '100+' END AS label,
                COUNT(*) AS n FROM c GROUP BY 1 ORDER BY 1''',
    'genres_distribution': '''SELECT btrim(g.genre) AS label, COUNT(*) AS n FROM staging.movies m
        CROSS JOIN LATERAL unnest(string_to_array(m.genres, '|')) g(genre)
        GROUP BY 1 ORDER BY n DESC''',
    'ratings_by_year': '''SELECT to_char(to_timestamp(rating_timestamp), 'YYYY') AS label, COUNT(*) AS n
        FROM staging.ratings GROUP BY 1 ORDER BY 1''',
}

def init_stage2() -> None:
    """Create quality audit and canonical movie/rating tables."""
    sql = (PROJECT_ROOT / 'sql' / '002_stage2.sql').read_text(encoding='utf-8')
    with connect() as conn:
        conn.execute(sql)

def source_fingerprint(conn: Any) -> str:
    """Require the exact Stage 1 dataset: all three completed MD5-pinned sources."""
    rows = conn.execute('''SELECT source_name, source_md5, expected_records, loaded_records, status
                          FROM staging.ingestion_batches ORDER BY source_name''').fetchall()
    source = {row[0]: row[1:] for row in rows}
    for name in ('movies.csv', 'links.csv', 'ratings.csv'):
        expected_md5, expected_n = EXPECTED_MD5[name], EXPECTED_COUNTS[name]
        if name not in source:
            raise ValueError(f'Missing Stage 1 ingestion ledger: {name}')
        md5, exp, loaded, status = source[name]
        if md5 != expected_md5 or exp != expected_n or loaded != expected_n or status != 'COMPLETED':
            raise ValueError(f'Stage 1 ledger not reconciled: {name}: {source[name]}')
    return '|'.join(f'{name}:{source[name][0]}:{source[name][2]}' for name in ('movies.csv', 'links.csv', 'ratings.csv'))

def metric_pass(value: int, expected: int | None) -> bool:
    """Evaluate a measured quality check against its expected value."""
    return expected is None or value == expected

def _run_sql_profile(conn: Any) -> dict[str, Any]:
    """Execute SQL data-quality checks and record an auditable profile run."""
    metrics: list[dict[str, Any]] = []
    for check in CHECKS:
        value = conn.execute(check.sql).fetchone()[0]
        if value is None:
            raise ValueError(f'Unexpected empty metric {check.name}')
        value = int(value)
        passed = metric_pass(value, check.expected)
        metrics.append({'name': check.name, 'value': value, 'expected': check.expected,
                        'severity': 'BLOCKER' if check.expected is not None else 'INFO',
                        'passed': passed, 'description': check.description})
        LOG.info('%s %s=%s', 'PASS' if passed else 'FAIL', check.name, f'{value:,}')
    distributions = {}
    for name, query in DISTRIBUTIONS.items():
        distributions[name] = [{'label': str(label), 'count': int(count)} for label, count in conn.execute(query).fetchall()]
    return {'metrics': metrics, 'distributions': distributions,
            'passed': all(m['passed'] for m in metrics)}

def run_profile(output_dir: Path | None = None) -> dict[str, Any]:
    """Read staging and persist an auditable profile; no edits to staging or core.

    Stage 1 ingestion must not run concurrently with profiling/preparation.
    """
    init_stage2()
    with connect() as conn:
        fp = source_fingerprint(conn)
        run_id = conn.execute('''INSERT INTO quality.profile_runs(status,source_fingerprint)
            VALUES ('RUNNING',%s) RETURNING run_id''', (fp,)).fetchone()[0]
    report: dict[str, Any] = {}
    try:
        with connect() as conn:
            # A consistent read snapshot for the entire profiling pass.
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            report = _run_sql_profile(conn)
        report.update({'roll_number': '2025AG05135', 'stage': 2, 'run_id': run_id,
                       'source_fingerprint': fp, 'generated_at': datetime.now(timezone.utc).isoformat()})
        with connect() as conn:
            for item in report['metrics']:
                conn.execute('''INSERT INTO quality.profile_metrics
                    (run_id,metric_name,metric_value,severity,passed,description)
                    VALUES (%s,%s,%s,%s,%s,%s)''',
                    (run_id, item['name'], item['value'], item['severity'], item['passed'], item['description']))
            conn.execute('''UPDATE quality.profile_runs SET status=%s,completed_at=now(),summary=%s::jsonb
                WHERE run_id=%s''', ('PASSED' if report['passed'] else 'FAILED', json.dumps(report), run_id))
        destination = output_dir or PROJECT_ROOT / 'artifacts' / 'stage2'
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / f'profile_run_{run_id}_2025AG05135.json'
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        LOG.info('Profile saved to %s', output)
        return report
    except Exception as exc:
        with connect() as conn:
            conn.execute("UPDATE quality.profile_runs SET status='FAILED',completed_at=now(),error_message=%s WHERE run_id=%s",
                         (str(exc)[:4000], run_id))
        raise

def prepare(force: bool = False) -> dict[str, int]:
    """Profile and rebuild canonical tables atomically; never alter staging."""
    report = run_profile()
    if not report['passed']:
        blockers = [m['name'] for m in report['metrics'] if not m['passed']]
        raise ValueError(f'Blocking quality checks failed: {blockers}; canonicalization stopped')
    init_stage2()
    run_id = report['run_id']
    fp = report['source_fingerprint']
    with connect() as conn:
        with conn.transaction():
            # Guard against concurrent prepare (session advisory lock until transaction end).
            conn.execute('SELECT pg_advisory_xact_lock(2025, 5135)')
            if not force:
                existing = conn.execute('''SELECT preparation_id FROM quality.preparation_runs
                    WHERE status='COMPLETED' AND source_fingerprint=%s LIMIT 1''', (fp,)).fetchone()
                if existing:
                    counts = {key: conn.execute(f'SELECT COUNT(*) FROM core.{key}').fetchone()[0]
                              for key in ('movies', 'links', 'ratings')}
                    if counts == {'movies': EXPECTED_COUNTS['movies.csv'],
                                  'links': EXPECTED_COUNTS['links.csv'], 'ratings': EXPECTED_COUNTS['ratings.csv']}:
                        LOG.info('Canonical tables already complete (preparation_id=%s)', existing[0])
                        return counts
    with connect() as conn:
        preparation_id = conn.execute('''INSERT INTO quality.preparation_runs
            (profile_run_id, source_fingerprint,status) VALUES (%s,%s,'RUNNING')
            RETURNING preparation_id''', (run_id, fp)).fetchone()[0]
    try:
        with connect() as conn:
            # All 3 tables and completion ledger commit or roll back together.
            conn.execute('SELECT pg_advisory_xact_lock(2025, 5135)')
            conn.execute('TRUNCATE core.ratings, core.links, core.movies')
            conn.execute('''INSERT INTO core.movies(movie_id,title,genres,has_genres)
                SELECT movie_id,title,
                       CASE WHEN genres = '(no genres listed)' OR btrim(genres) = '' THEN ARRAY[]::text[]
                            ELSE string_to_array(genres,'|') END,
                       (genres <> '(no genres listed)' AND btrim(genres) <> '')
                FROM staging.movies''')
            conn.execute('''INSERT INTO core.links(movie_id,imdb_id,tmdb_id)
                SELECT movie_id, NULLIF(btrim(imdb_id),''), NULLIF(btrim(tmdb_id),'') FROM staging.links''')
            conn.execute('''INSERT INTO core.ratings(user_id,movie_id,rating,rating_timestamp)
                SELECT user_id,movie_id,rating,rating_timestamp FROM staging.ratings''')
            counts = {key: conn.execute(f'SELECT COUNT(*) FROM core.{key}').fetchone()[0]
                      for key in ('movies', 'links', 'ratings')}
            expected = {'movies': EXPECTED_COUNTS['movies.csv'],
                        'links': EXPECTED_COUNTS['links.csv'], 'ratings': EXPECTED_COUNTS['ratings.csv']}
            if counts != expected:
                raise ValueError(f'Canonical reconciliation mismatch: expected={expected} actual={counts}')
            conn.execute('''UPDATE quality.preparation_runs
                SET status='COMPLETED',finished_at=now(),movie_rows=%s,link_rows=%s,rating_rows=%s
                WHERE preparation_id=%s''', (counts['movies'], counts['links'], counts['ratings'], preparation_id))
        return counts
    except Exception as exc:
        with connect() as conn:
            conn.execute("UPDATE quality.preparation_runs SET status='FAILED',finished_at=now(),error_message=%s WHERE preparation_id=%s",
                         (str(exc)[:4000], preparation_id))
        raise

def stage2_status() -> dict[str, list[tuple]]:
    """List recent profiling and canonical preparation results."""
    with connect() as conn:
        return {
            'profiles': conn.execute('''SELECT run_id,status,started_at,completed_at,error_message
                FROM quality.profile_runs ORDER BY run_id DESC LIMIT 5''').fetchall(),
            'preparations': conn.execute('''SELECT preparation_id,status,movie_rows,link_rows,rating_rows,error_message
                FROM quality.preparation_runs ORDER BY preparation_id DESC LIMIT 5''').fetchall(),
        }
