# Offline evaluation methodology — 2025AG05135

## Partition integrity

- Global chronological 80/10/10 train/validation/test split.
- Stage 3 user and movie behavioral features use **training data only**.
- Learned SGD logistic ranker trains on early validation (70% of validation
  *time span*, not necessarily 70% of validation rows).
- All ranking candidates and fixed/blend weight selection use **late validation**.
- Final test stays held out until selection is frozen.
- Previously rated train items are excluded from Top-K predictions.

## Metrics at K=10

- **Precision@10**: observed relevant hits / 10 (macro-averaged per user).
- **Recall@10**: relevant hits / number of observed positive held-out items.
- **NDCG@10**: discounted rank-weighted hits / ideal discounted hits.
- **Coverage**: count of unique recommended movies / total catalog size.
- **Diversity@10**: macro mean intra-list **Jaccard distance between
  nonempty movie-genre sets**, averaged over informative item pairs.
  Lists without at least two movies with known genres score 0.0.
- **Hit rate@10**: fraction of users with >=1 observed relevant hit.
- Overall and known-user/cold-user cohorts are reported separately.

**Important limitation**: MovieLens contains observed ratings, not
impression logs. Unrated items are unknown, not confirmed irrelevant.
Precision@K uses the standard held-out-hit numerator and fixed K
denominator but must not be advertised as exposure-verified precision.
Baseline popularity and cold-user scoring are often indistinguishable.

## SQL schema retention

The five assignment SQL files are **unchanged**. Thus, the pre-existing
`ml.evaluation_metrics` table keeps Recall/NDCG/hit/coverage fields.
New Precision@K and Diversity@K values are recorded in versioned JSON,
CSV and Markdown artifacts only. If SQL persistence becomes a requirement,
an additive migration (not a rewrite of the five original files) is needed.

## Metric version warning

Old evidence under `reports/evidence/` predates the two added metrics.
Newly generated results must be rerun after installation; do not copy
past scores into a new-results table or imply the new metrics were already
calculated.
