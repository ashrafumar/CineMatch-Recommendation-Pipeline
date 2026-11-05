-- CineMatch DMML Assignment 2 | Roll Number: 2025AG05135
-- Stage 1: Source-aligned staging tables. No NLP/tag ingestion.
CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.movies (
    movie_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    genres TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.links (
    movie_id INTEGER NOT NULL,
    imdb_id TEXT,
    tmdb_id TEXT
);

CREATE TABLE IF NOT EXISTS staging.ratings (
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating NUMERIC(2,1) NOT NULL,
    rating_timestamp BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.ingestion_batches (
    source_name TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_md5 CHAR(32) NOT NULL,
    expected_records BIGINT NOT NULL,
    loaded_records BIGINT,
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS staging.pipeline_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type TEXT NOT NULL,
    source_name TEXT,
    details TEXT NOT NULL
);
