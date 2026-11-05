-- CineMatch Assignment 2 | Roll Number watermark: 2025AG05135
-- Stage 5: Evaluation, model selection and lineage. No changes to prior tables.
CREATE SCHEMA IF NOT EXISTS ml;
CREATE TABLE IF NOT EXISTS ml.evaluation_runs (
  evaluation_run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  feature_run_id BIGINT NOT NULL REFERENCES features.feature_runs(run_id),
  svd_model_run_id BIGINT NOT NULL REFERENCES ml.model_runs(model_run_id),
  ranker_model_run_id BIGINT NOT NULL REFERENCES ml.model_runs(model_run_id),
  selection_id BIGINT,
  evaluation_split TEXT NOT NULL CHECK(evaluation_split IN ('validation','test')),
  specification_version TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('RUNNING','COMPLETED','FAILED')),
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  configuration JSONB NOT NULL,
  users_evaluated BIGINT,
  relevant_interactions BIGINT,
  error_message TEXT
);
CREATE TABLE IF NOT EXISTS ml.evaluation_metrics (
  evaluation_run_id BIGINT NOT NULL REFERENCES ml.evaluation_runs(evaluation_run_id),
  strategy TEXT NOT NULL,
  segment TEXT NOT NULL CHECK (segment IN ('all','known_user','cold_user')),
  evaluated_users BIGINT NOT NULL,
  relevant_interactions BIGINT NOT NULL,
  recall_at_k DOUBLE PRECISION,
  ndcg_at_k DOUBLE PRECISION,
  hit_rate_at_k DOUBLE PRECISION,
  catalog_coverage DOUBLE PRECISION,
  distinct_recommended_movies BIGINT NOT NULL,
  PRIMARY KEY(evaluation_run_id,strategy,segment)
);
CREATE TABLE IF NOT EXISTS ml.model_selections (
  selection_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  feature_run_id BIGINT NOT NULL REFERENCES features.feature_runs(run_id),
  validation_evaluation_run_id BIGINT NOT NULL REFERENCES ml.evaluation_runs(evaluation_run_id),
  selected_strategy TEXT NOT NULL,
  selected_weights JSONB,
  validation_ndcg_at_k DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  specification_version TEXT NOT NULL,
  selection_basis TEXT NOT NULL CHECK (selection_basis IN ('full_validation','partial_validation'))
);
ALTER TABLE ml.evaluation_runs ADD COLUMN IF NOT EXISTS selection_id BIGINT;
CREATE INDEX IF NOT EXISTS evaluation_runs_feature_idx ON ml.evaluation_runs(feature_run_id,evaluation_split,started_at);
