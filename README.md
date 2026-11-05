# CineMatch — End-to-End Data Management Pipeline for Movie Recommendations

**DM4ML Assignment 2 · Roll number: 2025AG05135**

CineMatch uses the public **MovieLens 32M** dataset to ingest and validate
users (identified by user IDs), movies, and rating interactions; construct
leakage-controlled features; train collaborative and learned hybrid models;
compare recommendation strategies; and orchestrate a reproducible workflow.

## Architecture

```text
MovieLens 32M CSVs
   | pinned checksums and source-count reconciliation
   v
PostgreSQL staging (COPY)  --> quality.profile_runs / quality.profile_metrics
   | validation and canonical preparation
   v
PostgreSQL core (movies, links, ratings)
   | time split and train-only aggregation
   v
features (versioned snapshots) --> sparse SVD model trained from scratch
   |                                 |
   +-- early validation labels -----+--> learned SGD logistic hybrid ranker
   |                                         |
   +-- late validation ranking and weight selection
   |                                         |
   +-- final held-out test evaluation --> CSV / Markdown / JSON / charts
   v
`dags/workflow.py` / `cinematch run-all` --> orchestration JSON manifest
```

The **original five SQL schema scripts are preserved byte-for-byte** in `sql/`.
Precision@K and Diversity@K are exported to reports without modifying the
fixed database metric table; see `docs/evaluation_methodology.md`.

## Repository map

- `data/{raw,processed,features,errors}`: raw inputs and local-only outputs; large data excluded from Git.
- `src/{ingestion,validation,preparation,transformation,features,training,evaluation}`: named assignment stage interfaces.
- `src/cinematch`: tested algorithms and existing CLI; facade modules delegate here to avoid duplicate implementation.
- `sql/`: unmodified PostgreSQL schemas 001–005.
- `dags/workflow.py`: one-command synchronous Python workflow.
- `configs/`: environment template; `models/`: local trained artifacts destination (actual artifacts in `artifacts/stage4`).
- `reports/evidence`: historical execution evidence, **not** fresh-run outputs.
- `docs/`: architecture, feature catalog, methodology, developer guide and audit notes.
- `tests/`: regression and metric unit tests.

## Prerequisites

- Python **3.11+**, PostgreSQL, and a database you can administer.
- Download **MovieLens 32M** from GroupLens (https://grouplens.org/datasets/movielens/32m/).
  Extract the `ml-32m` folder beneath `data/raw/`. Do not commit the dataset.
- The pipeline checks pinned source-file MD5s, including `tags.csv` for
  integrity, but intentionally does **not** ingest or model the text tags.
- A machine with enough RAM and free disk for 32M PostgreSQL ratings and a
  sparse user–item SVD; training is computationally intensive.

## 1. Configure and install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
# Edit DATABASE_URL and MOVIELENS_DIR in .env
pytest -q
```

In PostgreSQL, create the target database before using `cinematch init-db`:

```sql
CREATE DATABASE cinematch_dmml;
```

> **From-scratch proof:** Use a *new empty PostgreSQL database* and empty
> local `artifacts/`. Do not reuse the archived model weights; this package
> does not ship them. The training commands compute model parameters anew.

## 2. Run each assignment stage (recommended for grading evidence)

```bash
cinematch verify
cinematch init-db
cinematch ingest --kind all
cinematch reconcile
cinematch init-stage2
cinematch profile
cinematch prepare
cinematch init-stage3
cinematch build-features
cinematch stage3-status
cinematch init-stage4
cinematch train-models --rank 32
cinematch train-ranker --svd-model-run-id 1
cinematch stage4-status
cinematch init-stage5
cinematch evaluate --split validation --max-users 50
cinematch tune-hybrid
cinematch stage5-status
# Replace 1 with the full-validation selection ID shown above:
cinematch final-test --selection-id 1
```

**IDs are examples for a fresh database**: read actual model IDs and
selection IDs from command output. Never select a model using test labels.
The smoke run is exploratory only; it must not be used for final selection.

## 3. One-command orchestration from scratch

```bash
cinematch run-all
# Optional only after a valid full-validation selection is established:
cinematch run-all --final-test
```

`run-all` creates/reuses data and trains **both** models in dependency order.
Existing ingestion, preparation, feature snapshots, and models can be
reused; full validation is intentionally rerun. The generated manifest is
written to `artifacts/stage5/pipeline_manifest_2025AG05135.json`.
Note: `run-all --final-test` also runs validation; it does not simply
reprint prior scores. For production-scale evaluation, omit `--max-users`.

## Evaluation and model comparison

Popularity baseline; collaborative SVD; categorical genre affinity;
fixed hybrid; learned hybrid ranker; plus late-validation-selected blends.
Evaluate Precision@10, Recall@10, NDCG@10, coverage, genre Diversity@10,
and supplementary hit rate. These are ranking models, not explicit
numeric rating regressors, so RMSE/MAE are not applicable. See
`docs/evaluation_methodology.md` for caveats and database persistence.

## Evidence and final submission

`reports/evidence/` contains historical baseline evaluation/manifest
records only. New local execution produces reports in `artifacts/stage5/`.
Capture a screenshot of a successful `run-all` terminal completion,
include the `data validation` report from `artifacts/stage2/`, and attach
the final **2–3-page PDF report** with a repository link. The GitHub URL
is intentionally left for the student to provide after publishing.

## License and attribution

MovieLens 32M source: GroupLens Research, University of Minnesota.
Observe the source dataset license and attribution terms. No MovieLens
CSVs or user rating records are redistributed in this repository.
The project code is provided for the coursework submission; see the
`docs/developer_guide.md` for implementation assumptions.
