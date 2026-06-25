#!/usr/bin/env python3
"""Run tjbench from source.

    python3 run.py serve --open
    python3 run.py run --benchmark humaneval --original deepseek:deepseek-reasoner \
        --candidate deepseek:deepseek-chat --limit 20 --html

A thin launcher so you can run from a checkout without installing. The code
lives in the `tjbench` package; the installed `tjbench` console script
(`tjbench.cli:cli`) works the same once installed.
"""
from tjbench.cli import cli

if __name__ == "__main__":
    cli()
