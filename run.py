#!/usr/bin/env python3
"""Run tjbench from source — the reliable launcher.

    python3 run.py serve --open
    python3 run.py run --benchmark humaneval --original deepseek:deepseek-reasoner \
        --candidate deepseek:deepseek-chat --limit 20 --html

Why this exists: the project uses a flat module layout, so the installed
`tjbench` console script (entry point `cli:cli`) can be shadowed by a same-named
PyPI package (e.g. there's a `cli` package) and fail with an ImportError.
Running here puts the repo directory first on sys.path, so the local modules
always win — no install, no collisions.
"""
from cli import cli

if __name__ == "__main__":
    cli()
