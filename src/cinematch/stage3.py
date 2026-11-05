"""Stage 3: point-in-time-safe global split and versioned feature snapshots.

CineMatch DMML Assignment 2 | Roll Number / Watermark 2025AG05135
No NLP, tags, or video processing. Never modifies canonical ratings.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from .config import PROJECT_ROOT
from .db import connect
from .verify import EXPECTED_COUNTS

LOG = logging.getLogger(__name__)
SPEC_VERSION = 'global_time_80_10_10_v1'
PRIOR_STRENGTH = 10.0


def init_stage3() -> None:
    """Create versioned feature tables and the temporal split view if absent."""
    with connect() as conn:
        conn.execute((PROJECT_ROOT / 'sql' / '003_stage3.sql').read_text(encoding='utf-8'))


def split_for_timestamp(ts: int, train_end: int, validation_end: int) -> str:
    """Assign an interaction timestamp to a nonoverlapping chronological partition."""
    if train_end >= validation_end:
        raise ValueError('train_end must be smaller than validation_end')
    if ts <= train_end:
        return 'train'
    return 'validation' if ts <= validation_end else 'test'


def source_preparation(conn):
    """Return the prepared canonical dataset version required for feature construction."""
    row = conn.execute('''SELECT preparation_id FROM quality.preparation_runs
        WHERE status='COMPLETED' ORDER BY preparation_id DESC LIMIT 1''').fetchone()
    if row is None:
        raise ValueError('Run cinematch prepare before Stage 3')
    expected = EXPECTED_COUNTS['ratings.csv']
    count = conn.execute('SELECT COUNT(*) FROM core.ratings').fetchone()[0]
    if count != expected:
        raise ValueError(f'Core ratings count mismatch: {count} != {expected}')
    return row[0]


def compute_cutoffs(conn) -> tuple[int, int]:
    """Exact 80th/90th percentile of event timestamps, no random split."""
    row = conn.execute('''SELECT percentile_disc(ARRAY[0.8,0.9]::double precision[])
        WITHIN GROUP (ORDER BY rating_timestamp) FROM core.ratings''').fetchone()
    if row is None or row[0] is None or len(row[0]) != 2:
        raise ValueError('No timestamps available for global temporal split')
    train_end, validation_end = map(int, row[0])
    if train_end >= validation_end:
        raise ValueError('Timestamp ties prevent distinct train/validation boundaries')
    return train_end, validation_end


def _build_snapshot(conn, run_id: int, train_end: int, validation_end: int) -> dict:
    """Build each feature only from interactions at or before train_end."""
    LOG.info('Building genre dictionary and genre features')
    conn.execute('''INSERT INTO features.genre_catalog(run_id,genre)
        SELECT %s, g.genre FROM core.movies m CROSS JOIN LATERAL unnest(m.genres) AS g(genre)
        GROUP BY g.genre''', (run_id,))
    conn.execute('''INSERT INTO features.movie_genres(run_id,movie_id,genre)
        SELECT %s, m.movie_id, g.genre FROM core.movies m
        CROSS JOIN LATERAL unnest(m.genres) AS g(genre)''', (run_id,))
    LOG.info('Building train-period user feature snapshot')
    conn.execute('''INSERT INTO features.user_features
        (run_id,user_id,rating_count,mean_rating,positive_count,last_rating_timestamp)
        SELECT %s, user_id, COUNT(*), AVG(rating)::double precision,
          COUNT(*) FILTER (WHERE rating>=4), MAX(rating_timestamp)
        FROM core.ratings WHERE rating_timestamp<=%s GROUP BY user_id''', (run_id,train_end))
    LOG.info('Building train-period movie statistics and Bayesian-smoothed means')
    global_mean = conn.execute('''SELECT AVG(rating)::double precision
        FROM core.ratings WHERE rating_timestamp<=%s''', (train_end,)).fetchone()[0]
    if global_mean is None:
        raise ValueError('Training split has no ratings')
    conn.execute('''INSERT INTO features.movie_features
        (run_id,movie_id,rating_count,mean_rating,positive_count,bayesian_rating,has_genres)
        SELECT %s,m.movie_id,COALESCE(a.n,0),a.avg_rating,COALESCE(a.positive_n,0),
          CASE WHEN a.n IS NULL THEN NULL
               ELSE (a.n*a.avg_rating + %s*%s)/(a.n+%s) END,m.has_genres
        FROM core.movies m LEFT JOIN
          (SELECT movie_id,COUNT(*)::bigint AS n,AVG(rating)::double precision AS avg_rating,
                  COUNT(*) FILTER (WHERE rating>=4) AS positive_n
           FROM core.ratings WHERE rating_timestamp<=%s GROUP BY movie_id) a
        USING(movie_id)''', (run_id,PRIOR_STRENGTH,global_mean,PRIOR_STRENGTH,train_end))
    LOG.info('Building train-period user-genre preferences (genre weight normalized per film)')
    conn.execute('''INSERT INTO features.user_genre_preferences
        (run_id,user_id,genre,positive_weight,rating_weight)
        SELECT %s,r.user_id,mg.genre,
               SUM(CASE WHEN r.rating>=4 THEN 1.0 / gc.genre_count ELSE 0 END)::double precision,
               SUM(r.rating::double precision / gc.genre_count)::double precision
        FROM core.ratings r
        JOIN (SELECT movie_id,COUNT(*) AS genre_count
              FROM features.movie_genres WHERE run_id=%s GROUP BY movie_id) gc ON gc.movie_id=r.movie_id
        JOIN features.movie_genres mg ON mg.movie_id=r.movie_id AND mg.run_id=%s
        WHERE r.rating_timestamp<=%s
        GROUP BY r.user_id,mg.genre''', (run_id,run_id,run_id,train_end))
    rows = conn.execute('''SELECT
      COUNT(*) FILTER (WHERE rating_timestamp<=%s),
      COUNT(*) FILTER (WHERE rating_timestamp>%s AND rating_timestamp<=%s),
      COUNT(*) FILTER (WHERE rating_timestamp>%s)
      FROM core.ratings''', (train_end,train_end,validation_end,validation_end)).fetchone()
    train_rows, validation_rows, test_rows = map(int, rows)
    if min(rows) <= 0 or sum(rows) != EXPECTED_COUNTS['ratings.csv']:
        raise ValueError(f'Invalid chronological split counts: {rows}')
    counts = {
        'train_rows': train_rows, 'validation_rows': validation_rows, 'test_rows': test_rows,
        'user_feature_rows': conn.execute('SELECT COUNT(*) FROM features.user_features WHERE run_id=%s',(run_id,)).fetchone()[0],
        'movie_feature_rows': conn.execute('SELECT COUNT(*) FROM features.movie_features WHERE run_id=%s',(run_id,)).fetchone()[0],
        'genre_feature_rows': conn.execute('SELECT COUNT(*) FROM features.movie_genres WHERE run_id=%s',(run_id,)).fetchone()[0],
        'user_genre_rows': conn.execute('SELECT COUNT(*) FROM features.user_genre_preferences WHERE run_id=%s',(run_id,)).fetchone()[0],
        'global_mean': float(global_mean),
    }
    if counts['movie_feature_rows'] != EXPECTED_COUNTS['movies.csv']:
        raise ValueError('Movie feature coverage mismatch')
    if counts['user_feature_rows'] <= 0:
        raise ValueError('No train-period users')
    return counts


def build_features(force: bool=False) -> dict:
    """Idempotent versioned snapshot. --force is unsupported: history remains immutable."""
    if force:
        raise ValueError('Feature snapshots are immutable; change the spec version for a new snapshot')
    init_stage3()
    with connect() as conn:
        # Global advisory transaction lock avoids two concurrent Stage 3 workers.
        conn.execute('SELECT pg_advisory_xact_lock(2025,5136)')
        preparation_id = source_preparation(conn)
        train_end, validation_end = compute_cutoffs(conn)
        existing = conn.execute('''SELECT run_id,status FROM features.feature_runs WHERE
            preparation_id=%s AND specification_version=%s AND train_end=%s AND validation_end=%s''',
            (preparation_id,SPEC_VERSION,train_end,validation_end)).fetchone()
        if existing and existing[1] == 'COMPLETED':
            LOG.info('Existing completed feature snapshot %s; reusing',existing[0])
            return feature_status(run_id=existing[0], conn=conn)
        if existing:
            run_id = existing[0]
            # If an earlier attempt failed, all feature inserts were rolled back.
            conn.execute('''UPDATE features.feature_runs SET status='RUNNING',started_at=now(),
                completed_at=NULL,error_message=NULL WHERE run_id=%s''',(run_id,))
        else:
            run_id = conn.execute('''INSERT INTO features.feature_runs
                (preparation_id,specification_version,status,train_end,validation_end)
                VALUES (%s,%s,'RUNNING',%s,%s) RETURNING run_id''',
                (preparation_id,SPEC_VERSION,train_end,validation_end)).fetchone()[0]
        # One transaction: on any failure, no partial snapshot is committed.
        result = _build_snapshot(conn,run_id,train_end,validation_end)
        conn.execute('''UPDATE features.feature_runs SET status='COMPLETED',completed_at=now(),
            train_rows=%s,validation_rows=%s,test_rows=%s,user_feature_rows=%s,
            movie_feature_rows=%s,genre_feature_rows=%s,user_genre_rows=%s,global_mean=%s
            WHERE run_id=%s''',(
            result['train_rows'],result['validation_rows'],result['test_rows'],
            result['user_feature_rows'],result['movie_feature_rows'],
            result['genre_feature_rows'],result['user_genre_rows'],result['global_mean'],run_id))
        result.update({'run_id':run_id,'preparation_id':preparation_id,'train_end':train_end,
                       'validation_end':validation_end,'specification_version':SPEC_VERSION})
        return result


def feature_status(run_id: int|None=None, conn=None) -> dict:
    """Return lineage, row counts and cutoff metadata for the newest feature snapshot."""
    owns_conn = conn is None
    if owns_conn:
        conn = connect()
    try:
        if run_id is None:
            row = conn.execute('''SELECT run_id FROM features.feature_runs
                WHERE status='COMPLETED' ORDER BY run_id DESC LIMIT 1''').fetchone()
            if row is None:
                return {}
            run_id = row[0]
        cols = ('run_id','preparation_id','specification_version','status','train_end',
                'validation_end','train_rows','validation_rows','test_rows','user_feature_rows',
                'movie_feature_rows','genre_feature_rows','user_genre_rows','global_mean')
        row = conn.execute('''SELECT run_id,preparation_id,specification_version,status,train_end,
                validation_end,train_rows,validation_rows,test_rows,user_feature_rows,
                movie_feature_rows,genre_feature_rows,user_genre_rows,global_mean
                FROM features.feature_runs WHERE run_id=%s''',(run_id,)).fetchone()
        if row is None:
            return {}
        record = dict(zip(cols,row))
        for k in ('train_end','validation_end'):
            record[k+'_utc'] = datetime.fromtimestamp(record[k],timezone.utc).isoformat()
        if record['global_mean'] is not None:
            record['global_mean'] = float(record['global_mean'])
        return record
    finally:
        if owns_conn:
            conn.close()


def write_feature_report(report: dict) -> Path:
    """Export auditable Stage 3 feature statistics and lineage information."""
    path = PROJECT_ROOT / 'artifacts' / 'stage3'
    path.mkdir(parents=True,exist_ok=True)
    destination = path / f"feature_run_{report['run_id']}_2025AG05135.json"
    destination.write_text(json.dumps({'roll_number':'2025AG05135',**report},indent=2,default=str),encoding='utf-8')
    return destination
