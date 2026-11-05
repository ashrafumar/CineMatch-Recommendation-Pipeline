# Developer guide — CineMatch 2025AG05135

## General conventions

- Python 3.11, UTF-8 source, PEP 8 naming, explicit data-integrity checks.
- Every substantial stage has a module docstring describing its contract.
- Use direct PostgreSQL query placeholders rather than interpolating IDs.
- Follow a single direction of dependencies: verify / ingest -> profile /
  prepare -> features -> models -> evaluation -> orchestration.
- Handle large interaction sets using PostgreSQL server-side cursors and
  sparse matrices, not a 32M-row Pandas dense matrix.
- Source-of-truth pipeline logic remains in `cinematch`; assignment-named
  packages expose the same functions without diverging implementations.

## Local development

```bash
python -m pip install -e '.[dev]'
python -m compileall -q src tests dags
pytest -q
cinematch --help
```

Keep secrets in `.env`, never Git. Create a *new* PostgreSQL database to
demonstrate first-run ingestion and model training. A partially initialized
database should be diagnosed with `cinematch status`, `stage2-status`,
`stage3-status`, `stage4-status`, and `stage5-status` before retrying.

## File placement

- Stage algorithms: `src/cinematch/*.py`
- Assignment-based public stage interfaces: `src/<stage>/pipeline.py`
- Dependency workflow: `dags/workflow.py` + `src/cinematch/orchestrate.py`
- Immutable legacy DDL: `sql/001_stage1.sql` to `sql/005_stage5.sql`
- Structured experiment output: `artifacts/` (Git-ignored)
- Submission evidence: `reports/evidence/` (sanitized historical baseline)

## Performance observations

- Stage 2 currently profiles twice when `run-all` invokes `profile` and
  then `prepare`. This favors defensive validation but costs a second
  32M-row scan. A future fingerprint-keyed cache could safely avoid it.
- Full validation evaluation deliberately recomputes on `run-all` even
  with unchanged model artifacts; do not describe evaluation as idempotent.
- The Stage 3 `features.ratings_split` view cross-joins feature runs; always
  filter by a specific run ID for bounded query plans.
- A shared large-dataset replay is expensive. Avoid running `run-all`
  just to regenerate the final Markdown summary.

## Experiment tracking

Model algorithms, hyperparameters, training counts, artifact paths,
statuses, and version fields live in PostgreSQL (`ml.model_runs`). Stage 5
stores evaluation runs and selections in `ml` schema, and exports reports.

## Submission checklist

- [ ] Tests pass on the student's configured machine.
- [ ] Fresh model training and full validation/test finish.
- [ ] Full validation and final test are both labeled, with run IDs.
- [ ] New Precision@10 and Diversity@10 appear in exported reports.
- [ ] Screenshot of successful orchestration from the student's terminal.
- [ ] Data validation evidence is included under `reports/`.
- [ ] 2–3-page PDF, citing public dataset, output metrics, and repo URL.
