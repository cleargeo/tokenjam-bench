# Analytics Dashboard (Phase 4)

Phase 3 made every run queryable; Phase 4 makes it **visible**. The dashboard
gains analytics views backed by the historical DB (read-only), and the same data
is available from the CLI and JSON APIs.

```
results/history.duckdb (P3)
   ├─ leaderboard()  provider_matrix()  version_summary()  regressions()  trend()
   │
   ├─ Dashboard (offline SPA, hash router)   /api/leaderboard · /api/providers
   │     Overview · Leaderboards · Providers      /api/version-summary
   │     Versions · Regressions · Trends · Reports /api/regressions · /api/trend
   │
   └─ CLI   tjbench history leaderboard|providers|regressions|versions|trend  (+ --json)
```

Reads open the DB **read-only**, so the dashboard and CLI coexist with a
concurrent recording write.

## Dashboard views (`python3 run.py serve --open`)

| View | Shows |
|---|---|
| **Overview** | tiles, cross-version regression banner, accuracy/cost trend chart, latest runs |
| **Leaderboards** | models ranked by latest pass-rate on a chosen benchmark |
| **Providers** | per-model matrix: runs, benchmarks, avg accuracy, avg cost |
| **Versions** | per-TokenJam-version: runs, mean Δaccuracy / Δcost, regression count |
| **Regressions** | timeline of runs flagged `significant_regression` / `regression_suspected` |
| **Trends** | a config's accuracy + cost across TokenJam versions (table + chart) |
| **Reports** | every run with a link to its full HTML report |

Navigation is hash-based (`#/leaderboards`, …); the active view auto-refreshes
every 4s. Fully offline — inline CSS/JS, no external HTTP, stdlib server only.

## CLI

```bash
python3 run.py history leaderboard humaneval        # models ranked
python3 run.py history providers --json             # per-model matrix
python3 run.py history versions                     # tokenjam versions seen
python3 run.py history regressions                  # regression timeline
python3 run.py history trend humaneval --json       # accuracy/cost across versions
```

## JSON APIs

`GET /api/leaderboard?benchmark=X` · `GET /api/providers` ·
`GET /api/version-summary` · `GET /api/regressions` · `GET /api/configs` ·
`GET /api/trend?benchmark=X&original=…&candidate=…`

Each returns `{rows: [...]}` (plus `benchmark` where relevant), ready for TokenJam
Lens or any external consumer to render. TokenJam never executes benchmarks — it
would only consume these results.

## Honesty

Leaderboard pass-rates and trend points carry the same evidence-based verdicts
(Wilson + McNemar) recorded per run. Nothing is labelled "safe"; a model's place
on a leaderboard is its measured pass-rate on that suite, not a quality claim.
