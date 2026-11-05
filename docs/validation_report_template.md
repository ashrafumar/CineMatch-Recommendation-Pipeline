# Data validation evidence — 2025AG05135

After a fresh end-to-end run, copy the generated profile JSON from
`artifacts/stage2/` to `reports/evidence/` and summarize:

| Check | Expected |
|---|---|
| movies rows | 87,585 |
| links rows | 87,585 |
| ratings rows | 32,000,204 |
| duplicate user–movie pairs | 0 |
| invalid ratings / timestamps | 0 |
| orphan ratings | 0 |
| blank movie titles | 0 |
| movies without genres | 7,080 (informational) |

Invalid records should be logged or segregated. With the verified official
MovieLens snapshot, blocking invalid-record counts were zero. On malformed
input, source checksum/header verification blocks ingestion; the quality
profile records failed blocker checks. The current code does *not* emit a
row-level rejected-record CSV for arbitrary malformed datasets; document
this limitation and do not claim otherwise.
