-- CineMatch DMML Assignment 2 | Roll Number: 2025AG05135
-- Stage 3: versioned temporal splits and feature snapshots. No changes to core/staging.
CREATE SCHEMA IF NOT EXISTS features;
CREATE TABLE IF NOT EXISTS features.feature_runs (
 run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 preparation_id BIGINT NOT NULL REFERENCES quality.preparation_runs(preparation_id),
 specification_version TEXT NOT NULL,
 started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 completed_at TIMESTAMPTZ,
 status TEXT NOT NULL CHECK (status IN ('RUNNING','COMPLETED','FAILED')),
 train_end BIGINT NOT NULL,
 validation_end BIGINT NOT NULL,
 train_rows BIGINT,
 validation_rows BIGINT,
 test_rows BIGINT,
 user_feature_rows BIGINT,
 movie_feature_rows BIGINT,
 genre_feature_rows BIGINT,
 user_genre_rows BIGINT,
 global_mean NUMERIC,
 error_message TEXT,
 UNIQUE(preparation_id,specification_version,train_end,validation_end),
 CHECK (train_end < validation_end)
);
CREATE TABLE IF NOT EXISTS features.genre_catalog (
 run_id BIGINT NOT NULL REFERENCES features.feature_runs(run_id),
 genre TEXT NOT NULL,
 PRIMARY KEY (run_id,genre)
);
CREATE TABLE IF NOT EXISTS features.movie_genres (
 run_id BIGINT NOT NULL REFERENCES features.feature_runs(run_id),
 movie_id INTEGER NOT NULL REFERENCES core.movies(movie_id),
 genre TEXT NOT NULL,
 PRIMARY KEY (run_id,movie_id,genre),
 FOREIGN KEY(run_id,genre) REFERENCES features.genre_catalog(run_id,genre)
);
CREATE INDEX IF NOT EXISTS movie_genres_run_genre_idx ON features.movie_genres(run_id,genre);
CREATE TABLE IF NOT EXISTS features.user_features (
 run_id BIGINT NOT NULL REFERENCES features.feature_runs(run_id),
 user_id INTEGER NOT NULL,
 rating_count BIGINT NOT NULL,
 mean_rating DOUBLE PRECISION NOT NULL,
 positive_count BIGINT NOT NULL,
 last_rating_timestamp BIGINT NOT NULL,
 PRIMARY KEY (run_id,user_id)
);
CREATE TABLE IF NOT EXISTS features.movie_features (
 run_id BIGINT NOT NULL REFERENCES features.feature_runs(run_id),
 movie_id INTEGER NOT NULL REFERENCES core.movies(movie_id),
 rating_count BIGINT NOT NULL,
 mean_rating DOUBLE PRECISION,
 positive_count BIGINT NOT NULL,
 bayesian_rating DOUBLE PRECISION,
 has_genres BOOLEAN NOT NULL,
 PRIMARY KEY (run_id,movie_id)
);
CREATE TABLE IF NOT EXISTS features.user_genre_preferences (
 run_id BIGINT NOT NULL REFERENCES features.feature_runs(run_id),
 user_id INTEGER NOT NULL,
 genre TEXT NOT NULL,
 positive_weight DOUBLE PRECISION NOT NULL,
 rating_weight DOUBLE PRECISION NOT NULL,
 PRIMARY KEY (run_id,user_id,genre),
 FOREIGN KEY (run_id,user_id) REFERENCES features.user_features(run_id,user_id),
 FOREIGN KEY (run_id,genre) REFERENCES features.genre_catalog(run_id,genre)
);
CREATE INDEX IF NOT EXISTS user_genre_preferences_genre_idx ON features.user_genre_preferences(run_id,genre);
-- Virtual split membership: full 32M retained in core without a second 32M-row copy.
-- Use a specific run_id in downstream queries for reproducible boundaries.
CREATE OR REPLACE VIEW features.ratings_split AS
SELECT f.run_id,r.user_id,r.movie_id,r.rating,r.rating_timestamp,
       CASE WHEN r.rating_timestamp<=f.train_end THEN 'train'
            WHEN r.rating_timestamp<=f.validation_end THEN 'validation'
            ELSE 'test' END AS split
FROM core.ratings r CROSS JOIN features.feature_runs f
WHERE f.status='COMPLETED';
