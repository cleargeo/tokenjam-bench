"""Replay telemetry — load real TokenJam history for replay validation.

Replay consumes EXPORTED TokenJam telemetry and never modifies TokenJam. Each
historical LLM call becomes a `ReplayTurn`: the reconstructed prompt, the model's
original output (used as the equivalence reference), and the original token
counts (for the cost baseline).

Two read-only sources, dispatched by file extension:
  - `.jsonl` / `.json` — a portable export (one LLM call per line). This is the
    decoupled, recommended format; produce it however you like from TokenJam.
  - `.duckdb` — read TokenJam's span store directly, READ-ONLY, consuming it like
    any external user. Uses TokenJam's own public semconv constants.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReplayTurn:
    session_id: str
    prompt: str
    original_output: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


# Portable JSONL field aliases (be liberal in what we accept).
_PROMPT_KEYS = ("prompt", "input", "prompt_content")
_OUTPUT_KEYS = ("output", "original_output", "completion", "completion_content", "response")


def _first(d: dict, keys, default=""):
    for k in keys:
        if d.get(k):
            return d[k]
    return default


def _load_jsonl(path: Path) -> list[ReplayTurn]:
    turns: list[ReplayTurn] = []
    text = path.read_text(encoding="utf-8")
    # Accept JSONL (one object per line) or a top-level JSON array.
    records: list[dict] = []
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        prompt = str(_first(r, _PROMPT_KEYS))
        output = str(_first(r, _OUTPUT_KEYS))
        if not prompt or not output:
            continue  # can't replay/judge a turn missing either side
        turns.append(ReplayTurn(
            session_id=str(r.get("session_id") or r.get("trace_id") or f"turn-{i}"),
            prompt=prompt, original_output=output,
            provider=str(r.get("provider") or "unknown"),
            model=str(r.get("model") or "unknown"),
            input_tokens=int(r.get("input_tokens") or 0),
            output_tokens=int(r.get("output_tokens") or 0),
        ))
    return turns


def _load_duckdb(path: Path) -> list[ReplayTurn]:
    """Read TokenJam's span store READ-ONLY (never writes/modifies it)."""
    import duckdb

    try:
        from tokenjam.otel.semconv import GenAIAttributes
        prompt_key = GenAIAttributes.PROMPT_CONTENT
        completion_key = GenAIAttributes.COMPLETION_CONTENT
    except Exception:  # pragma: no cover - tolerate semconv changes
        prompt_key, completion_key = "gen_ai.prompt.content", "gen_ai.completion.content"

    conn = duckdb.connect(str(path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT session_id, provider, model, input_tokens, output_tokens, attributes "
            "FROM spans WHERE model IS NOT NULL ORDER BY start_time"
        ).fetchall()
    finally:
        conn.close()

    turns: list[ReplayTurn] = []
    for session_id, provider, model, in_tok, out_tok, attrs in rows:
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs)
            except json.JSONDecodeError:
                attrs = {}
        if not isinstance(attrs, dict):
            continue
        prompt = attrs.get(prompt_key)
        output = attrs.get(completion_key)
        if not prompt or not output:
            continue  # content capture was off for this span — can't replay it
        if not isinstance(prompt, str):
            prompt = json.dumps(prompt)
        if not isinstance(output, str):
            output = json.dumps(output)
        turns.append(ReplayTurn(
            session_id=str(session_id or "unknown"),
            prompt=prompt, original_output=output,
            provider=str(provider or "unknown"), model=str(model or "unknown"),
            input_tokens=int(in_tok or 0), output_tokens=int(out_tok or 0),
        ))
    return turns


def load_telemetry(path: str | Path) -> list[ReplayTurn]:
    """Load replay turns from a telemetry export (.jsonl/.json) or a TokenJam
    DuckDB store (.duckdb, read-only)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"telemetry not found: {p}")
    if p.suffix == ".duckdb":
        return _load_duckdb(p)
    return _load_jsonl(p)


def dominant_model(turns: list[ReplayTurn]) -> tuple[str, str]:
    """The most common (provider, model) across turns — the replay's 'original'."""
    counts: dict[tuple[str, str], int] = {}
    for t in turns:
        counts[(t.provider, t.model)] = counts.get((t.provider, t.model), 0) + 1
    if not counts:
        return ("unknown", "unknown")
    return max(counts.items(), key=lambda kv: kv[1])[0]
