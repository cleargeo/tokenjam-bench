# Historical Benchmark Database (P3)

Every proof writes an isolated JSON/HTML artifact. The history database is the
**queryable index over all of them** — one DuckDB store that powers trends,
leaderboards, regression history, and the analytics dashboard.

```
any proof (run / agent / replay / scenario / ci_benchmark)
   → results/tjbench_*.json        (artifact, unchanged)
   → record() into results/history.duckdb   (idempotent, deterministic run_id)
        │
        ▼  query layer
   list_runs · versions · trend · configs
        ▲
   CLI: tjbench history …          Dashboard: /api/history + auto-ingest on serve
```

DuckDB comes in via the `tokenjam` dependency — we never add it. The schema is
append-only (`MIGRATIONS` in `history.py`), mirroring TokenJam's own migration
pattern; new columns are added as new migrations, never by editing old ones.

## What's stored

One row per run in `benchmark_runs`: `run_id`, `created_at`, `benchmark`,
`original_model`, `candidate_model`, `tokenjam_version`, `n_tasks`,
pass counts + rates, `accuracy_delta_pp`, `cost_delta_pct`, costs,
Wilson CI (`wilson_low/high`), paired-delta CI, McNemar (`b`/`c`/`p`),
`significant`, `verdict`, `samples_per_task`, `mock`, `priced_with_defaults`,
`output_token_inflation`, `deepeval_score` (the judge pass-rate for
`judged`/`replay`), and the `json_path` / `html_path`.

`record()` is **idempotent**: the `run_id` is a hash of
`(benchmark, original, candidate, tokenjam_version, created_at)`, so re-ingesting
the same results never duplicates.

## Recording

Recording is automatic and best-effort (it never breaks a run):

- `tjbench run|agent|replay …` records into `results/history.duckdb`
- `ci_benchmark.py` records every CI run
- `tjbench serve` ingests `results/` on startup

Backfill from existing artifacts at any time:

```bash
python3 run.py history ingest                 # scans results/ → DB
python3 run.py history ingest --dir other/ --db other/history.duckdb
```

## Querying

```bash
python3 run.py history list                   # recent runs (table)
python3 run.py history list --benchmark humaneval --json
python3 run.py history versions               # tokenjam versions seen (ascending)
python3 run.py history trend humaneval        # accuracy + cost across versions
python3 run.py history trend replay --json
```

The dashboard exposes the same data at `GET /api/history`
(`{available, count, versions, configs}`) and shows "runs in history" /
"tokenjam versions" tiles. The richer history pages (leaderboards, regression
timeline) are Phase 4, built on this query layer.

## Notes

- The DB lives at `results/history.duckdb` (gitignored). It's an *index* — the
  JSON/HTML artifacts remain the source of truth, so deleting it is harmless
  (`history ingest` rebuilds it).
- Reads (dashboard, `history list`) open the DB **read-only**, so they coexist
  with a concurrent recording write.
