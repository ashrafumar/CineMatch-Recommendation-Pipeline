# Feature catalog — 2025AG05135

Each behavioral feature uses only `rating_timestamp <= train_end`.
The catalog is attached to `features.feature_runs.run_id`; it is reusable
for both training and inference. No validation/test labels leak into it.

| Family | Feature | Source | Type | Meaning / safeguards |
|---|---|---|---|---|
| User | `rating_count` | `core.ratings` | BIGINT | User's historical activity |
| User | `mean_rating` | `core.ratings` | DOUBLE | Mean train-period rating |
| User | `positive_count` | `core.ratings` | BIGINT | Train ratings >=4 |
| User | `last_rating_timestamp` | `core.ratings` | BIGINT | Latest train interaction |
| Movie | `rating_count` | `core.ratings` | BIGINT | Item popularity count |
| Movie | `mean_rating` | `core.ratings` | DOUBLE | Train-period mean |
| Movie | `positive_count` | `core.ratings` | BIGINT | Train positive count |
| Movie | `bayesian_rating` | ratings + global mean | DOUBLE | Shrunk item popularity score |
| Movie | `has_genres` | `core.movies` | BOOLEAN | Missing genre indicator |
| Metadata | `movie_genres` | `core.movies.genres` | BINARY | Movie categorical membership |
| User × genre | `positive_weight` | ratings joined to genres | DOUBLE | Preferred categories from train positives |
| User × genre | `rating_weight` | ratings joined to genres | DOUBLE | Historical genre affinity |
| Model | SVD collaborative dot product | train matrix | FLOAT | User/movie latent similarity |
| Model | normalized popularity | train movie features | FLOAT | Ranker input |
| Model | genre affinity | user-genre + movie-genre | FLOAT | Ranker input |
| Model | known-user flag | train user IDs | FLOAT | Cold-start indicator |
| Model | has-genre-preference flag | train preferences | FLOAT | Missing signal indicator |

The full 32M-row sparse user–movie matrix is constructed in memory from
**train-partition positive ratings**, not copied to PostgreSQL as a dense
matrix. Feature tables are versioned and linked by feature_run_id.
