-- CineMatch DMML Assignment 2 | Roll Number: 2025AG05135
-- Stage 2: quality profiling and prepared canonical tables (Stage 1 unchanged).
CREATE SCHEMA IF NOT EXISTS quality;
CREATE SCHEMA IF NOT EXISTS core;

CREATE TABLE IF NOT EXISTS quality.profile_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('RUNNING','PASSED','FAILED')),
    source_fingerprint TEXT NOT NULL,
    summary JSONB,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS quality.profile_metrics (
    run_id BIGINT NOT NULL REFERENCES quality.profile_runs(run_id),
    metric_name TEXT NOT NULL,
    metric_value NUMERIC NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('BLOCKER','INFO')),
    passed BOOLEAN NOT NULL,
    description TEXT NOT NULL,
    PRIMARY KEY (run_id, metric_name)
);
CREATE TABLE IF NOT EXISTS quality.preparation_runs (
    preparation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    profile_run_id BIGINT NOT NULL REFERENCES quality.profile_runs(run_id),
    source_fingerprint TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('RUNNING','COMPLETED','FAILED')),
    movie_rows BIGINT,
    link_rows BIGINT,
    rating_rows BIGINT,
    error_message TEXT
);

-- Metadata is stored without NLP processing; title is display-only.
CREATE TABLE IF NOT EXISTS core.movies (
    movie_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT[] NOT NULL,
    has_genres BOOLEAN NOT NULL
);
CREATE TABLE IF NOT EXISTS core.links (
    movie_id INTEGER PRIMARY KEY REFERENCES core.movies(movie_id),
    imdb_id TEXT,
    tmdb_id TEXT
);
-- Preserve all 32M source interactions when the quality contract passes.
CREATE TABLE IF NOT EXISTS core.ratings (
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL REFERENCES core.movies(movie_id),
    rating NUMERIC(2,1) NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    rating_timestamp BIGINT NOT NULL CHECK (rating_timestamp > 0),
    PRIMARY KEY (user_id, movie_id)
);
CREATE INDEX IF NOT EXISTS core_ratings_movie_id_idx ON core.ratings(movie_id);
CREATE INDEX IF NOT EXISTS core_ratings_user_time_idx ON core.ratings(user_id, rating_timestamp);
