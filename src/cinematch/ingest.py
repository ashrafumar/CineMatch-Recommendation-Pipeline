"""Atomic per-file PostgreSQL COPY ingestion. CineMatch | Roll Number: 2025AG05135.

A failed COPY rolls back the complete file transaction, and reruns replace that
file's staging table atomically. This favors correctness and completion over speed.
"""
import logging
from pathlib import Path
from .db import connect
from .verify import EXPECTED_COUNTS, EXPECTED_MD5, HEADERS, verify_file

LOG = logging.getLogger(__name__)

# Explicit allowlist: table names are never taken directly from CLI input.
SOURCES = {
    "movies": ("movies.csv", "staging.movies", "movie_id, title, genres"),
    "links": ("links.csv", "staging.links", "movie_id, imdb_id, tmdb_id"),
    "ratings": ("ratings.csv", "staging.ratings", "user_id, movie_id, rating, rating_timestamp"),
}


def _mark_running(filename: str, file_path: Path, checksum: str, expected: int) -> None:
    """Record that source ingestion began for an identified CSV batch."""
    with connect() as conn:
        conn.execute("""
            INSERT INTO staging.ingestion_batches
                (source_name, source_path, source_md5, expected_records, loaded_records, status,
                 started_at, finished_at, error_message)
            VALUES (%s,%s,%s,%s,NULL,'RUNNING',NOW(),NULL,NULL)
            ON CONFLICT (source_name) DO UPDATE SET
                source_path=EXCLUDED.source_path, source_md5=EXCLUDED.source_md5,
                expected_records=EXCLUDED.expected_records, loaded_records=NULL,
                status='RUNNING', started_at=NOW(), finished_at=NULL, error_message=NULL
        """, (filename, str(file_path), checksum, expected))
        conn.execute("INSERT INTO staging.pipeline_events(event_type,source_name,details) VALUES ('STARTED',%s,%s)",
                     (filename, f"source={file_path}"))


def _mark_failure(filename: str, error: Exception) -> None:
    """Persist ingestion failure details for diagnosis and safe retry."""
    with connect() as conn:
        conn.execute("""UPDATE staging.ingestion_batches SET status='FAILED',
            error_message=%s, finished_at=NOW() WHERE source_name=%s""",
            (str(error)[:4000], filename))
        conn.execute("INSERT INTO staging.pipeline_events(event_type,source_name,details) VALUES ('FAILED',%s,%s)",
                     (filename, str(error)[:4000]))


def ingest_one(kind: str, directory: Path, force: bool = False, chunk_size: int = 8 * 1024 * 1024) -> int:
    """Verify and bulk-COPY one MovieLens source into PostgreSQL staging."""
    filename, table, columns = SOURCES[kind]
    file_path = directory / filename
    expected = EXPECTED_COUNTS[filename]
    LOG.info("Verifying %s against pinned official MD5 and header", file_path)
    checksum = verify_file(file_path, EXPECTED_MD5[filename], HEADERS[filename])

    # Do not silently trust a completed ledger without checking the actual table count.
    with connect() as conn:
        row = conn.execute("SELECT status, source_md5, loaded_records FROM staging.ingestion_batches WHERE source_name=%s",
                           (filename,)).fetchone()
        if row and row[0] == "COMPLETED" and row[1] == checksum and row[2] == expected and not force:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count == expected:
                LOG.info("Skipping %s (already completed and reconciled: %s rows)", filename, count)
                return count
            LOG.warning("Ledger and table differ for %s; loading again", filename)

    _mark_running(filename, file_path, checksum, expected)
    try:
        # All changes to the target table and COMPLETED status commit together.
        # No table lock survives rollback; partial COPY cannot become a completed batch.
        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {table}")
                LOG.info("COPY started: %s -> %s", filename, table)
                with cursor.copy(f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, HEADER true)") as copy:
                    with file_path.open("rb") as stream:
                        while block := stream.read(chunk_size):
                            copy.write(block)
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                loaded = cursor.fetchone()[0]
                if loaded != expected:
                    raise ValueError(f"Count mismatch {filename}: expected {expected:,}, loaded {loaded:,}")
                cursor.execute("""
                    UPDATE staging.ingestion_batches
                       SET status='COMPLETED', loaded_records=%s,
                           finished_at=NOW(), error_message=NULL
                     WHERE source_name=%s
                """, (loaded, filename))
                cursor.execute("INSERT INTO staging.pipeline_events(event_type,source_name,details) VALUES ('COMPLETED',%s,%s)",
                               (filename, f"loaded={loaded}; md5={checksum}"))
        LOG.info("COPY completed: %s: %s rows", filename, f"{loaded:,}")
        return loaded
    except Exception as exc:
        _mark_failure(filename, exc)
        LOG.exception("Ingestion failed for %s (rolled back)", filename)
        raise


def status() -> list[tuple]:
    """Return recorded staging ingestion batches and their statuses."""
    with connect() as conn:
        return conn.execute("""SELECT source_name, status, expected_records, loaded_records,
                                      started_at, finished_at, error_message
                               FROM staging.ingestion_batches ORDER BY source_name""").fetchall()


def reconcile() -> dict[str, tuple[int, int, bool]]:
    """Compare staged row counts and ingestion outcomes to official source baselines."""
    result = {}
    with connect() as conn:
        for _, (filename, table, _) in SOURCES.items():
            actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            expected = EXPECTED_COUNTS[filename]
            result[filename] = (expected, actual, actual == expected)
    return result
