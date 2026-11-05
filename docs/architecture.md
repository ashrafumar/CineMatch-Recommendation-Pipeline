# Pipeline architecture — 2025AG05135

This implementation uses PostgreSQL for transactional/audited ingestion,
referential constraints, versioned features, and experiment tracking.
The `src/cinematch` package is the single implementation source; stage
interface modules in `src/<assignment-stage>/pipeline.py` preserve the
assignment's required navigable directory layout without code duplication.

## Flow

```mermaid
flowchart TD
  A[MovieLens 32M CSV] --> B[Source checksum / header verification]
  B --> C[Staging COPY / audit]
  C --> D[Quality profile / reject on blockers]
  D --> E[Canonical preparation]
  E --> F[Temporal train-validation-test split]
  F --> G[Train-only feature snapshots]
  G --> H[Sparse SVD model training]
  G --> I[Popularity and genre feature baselines]
  H --> J[Early validation: learned hybrid training]
  H --> K[Late validation: rank strategies]
  I --> K
  J --> K
  K --> L[Select winning strategy]
  L --> M[Held-out final test]
  M --> N[CSV / JSON / Markdown reports]
  O[Python DAG / orchestration manifest] -.-> B
  O -.-> M
```

No tag-based NLP model is used. Raw interaction truth is limited to the
proper partition for each training or evaluation step.
