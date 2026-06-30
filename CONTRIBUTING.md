# Contributing to TokenJam Bench

Thanks for helping. TokenJam Bench is honesty-branded: its credibility is its
evidence. The bar for a change is that every number it shows traces to a real,
reproducible measurement.

## Setup

```bash
pip install -e ".[dev]"
pytest          # full offline suite — no keys, no spend
ruff check .    # lint
```

Supported Python: 3.10, 3.11, 3.12. CI runs the suite on all three.

The `pip install -e ".[dev]"` above is the **contributor** install — editable
installs correctly stay `pip`. The **end-user** install is
`pipx install tokenjam-bench` (pipx isolates the `tjb` CLI and sidesteps the
PEP-668 "externally-managed-environment" error on Homebrew Python and Debian
12+/Ubuntu 24+). When you touch user-facing docs, keep `pipx` primary — don't
reintroduce a bare `pip install tokenjam-bench` as the headline install.

Run the app to see your change:

```bash
tjb run         # zero-flag offline proof
tjb serve       # dashboard over the bundled real evidence
```

## The honesty rules (enforced in CI)

The honesty guard [`tests/test_honesty_guard.py`](tests/test_honesty_guard.py)
fails CI if a change reintroduces a dishonest surface. Before you open a PR:

- No headline/dashboard number may come from a placeholder-priced run
  (`priced_with_defaults=true`). Re-run with real rates, or keep legacy runs
  under `docs/evidence/archive/` (non-headline).
- No banned overclaim strings in README, docs, or the dashboard: no
  "quality preserved", no "safe to replace", no single `confidence = NN%`
  scalar, no ROI extrapolation ("at 10x", "annual savings"). The honest forms —
  Wilson CI, McNemar p-value, and the three hedged verdicts
  (`no_significant_regression` / `significant_regression` /
  `insufficient_evidence`) — are what to use instead.
- Accuracy is the pass-rate on a named suite. It is never a general quality
  claim.

## How it's built

- **Offline-first.** Tests, lint, and the default `tjb run` work with no
  provider keys and no network. Live providers are opt-in via a key in the env.
- **Flat layout.** Top-level modules and subpackages live under `tjbench/`.
- **TokenJam is a published dependency**, consumed like any external user — never
  vendored. Every artifact is stamped with the exact `tokenjam_version`.

See [docs/development.md](docs/development.md) for adding a benchmark or model
client, and [docs/architecture.md](docs/architecture.md) for the data flow.

## Pull requests

1. Branch off `main`.
2. Keep the change focused; update docs when behavior changes.
3. Make sure `ruff check .` and `pytest` pass locally (the honesty guard runs
   inside `pytest`).
4. Fill out the PR template. CI must be green before review.

## Reporting issues

Use the issue templates. For a wrong or surprising number, include the artifact
JSON (or its filename under `docs/evidence/` or `results/`) so it's reproducible.
