.PHONY: install update-tokenjam test lint bench-smoke version serve dashboard

# Use the python where the deps (tokenjam, click, rich) are installed.
PY ?= python3

# Install the bench (editable) + dev tooling.
install:
	pip install -e ".[dev]"

# THE daily-pull command: upgrade to the latest published TokenJam and show the
# version every subsequent proof will be stamped with.
update-tokenjam:
	pip install -U tokenjam
	@$(PY) -c "import importlib.metadata as m; print('tokenjam now at', m.version('tokenjam'))"

# All commands go through run.py so they work without relying on the installed
# console script (flat layout → `cli` collides with the `cli` PyPI package).
version:
	$(PY) run.py version

# Live proof dashboard (offline, auto-refreshing) at http://127.0.0.1:7392/
serve dashboard:
	$(PY) run.py serve --open

# Offline end-to-end smoke (no keys, no spend).
bench-smoke:
	$(PY) run.py run --benchmark samples --original anthropic:claude-opus-4-7 --mock

test:
	pytest -q

lint:
	ruff check .

# Continuous-benchmark set: offline always; live if a provider key is exported.
ci-bench:
	$(PY) -m tjbench.ci_benchmark
