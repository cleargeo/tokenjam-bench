.PHONY: install update-tokenjam test lint bench-smoke version

# Install the bench (editable) + dev tooling.
install:
	pip install -e ".[dev]"

# THE daily-pull command: upgrade to the latest published TokenJam and show the
# version every subsequent proof will be stamped with. This is the only step
# needed to test a new TokenJam release.
update-tokenjam:
	pip install -U tokenjam
	@python -c "import importlib.metadata as m; print('tokenjam now at', m.version('tokenjam'))"

version:
	tjbench version

# Offline end-to-end smoke (no keys, no spend): proves the pipeline runs and
# produces a stamped artifact.
bench-smoke:
	tjbench run --benchmark samples --original anthropic:claude-opus-4-7 --mock

test:
	pytest -q

lint:
	ruff check .
