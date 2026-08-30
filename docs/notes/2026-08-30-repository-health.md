# Repository health notes — 2026-08-30

- Reproduced the failing workflow locally: feature construction returned zero rows and linear regression rejected an empty training matrix.
- Replaced the insufficient demo with deterministic generated data and documented its provenance and limits.
- Added explicit temporal-split validation, tests, dependency ranges, least-privilege workflow permissions, caching, and pipeline execution in CI.
- Regenerated reports from the repaired pipeline and updated README metrics to match the observed run.
- Next improvement: replace the single holdout with rolling-origin evaluation and derive horizon metrics from actual forecasts rather than an illustrative table.
