-- CineMatch DMML Assignment 2 | Roll No. 2025AG05135
-- Stage 4: model registry and auditable recommendations.
CREATE SCHEMA IF NOT EXISTS ml;
CREATE TABLE IF NOT EXISTS ml.model_runs (
 model_run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 feature_run_id BIGINT NOT NULL REFERENCES features.feature_runs(run_id),
 specification_version TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('RUNNING','COMPLETED','FAILED')),
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 completed_at TIMESTAMPTZ,
 algorithm TEXT NOT NULL,
 parameters JSONB NOT NULL,
 artifact_path TEXT,
 train_interactions BIGINT,
 train_positive_interactions BIGINT,
 error_message TEXT,
 UNIQUE(feature_run_id,specification_version,algorithm)
);
CREATE TABLE IF NOT EXISTS ml.recommendations (
 model_run_id BIGINT NOT NULL REFERENCES ml.model_runs(model_run_id),
 user_id INTEGER NOT NULL,
 movie_id INTEGER NOT NULL REFERENCES core.movies(movie_id),
 rank INTEGER NOT NULL CHECK(rank>=1),
 score DOUBLE PRECISION NOT NULL,
 strategy TEXT NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(model_run_id,user_id,rank),
 UNIQUE(model_run_id,user_id,movie_id)
);
CREATE INDEX IF NOT EXISTS recommendations_user_idx ON ml.recommendations(user_id,model_run_id);
