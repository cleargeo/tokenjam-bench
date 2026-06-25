"""Replay validation — loaders + the replay proof, fully offline (DI'd)."""
from __future__ import annotations

import json

import pytest

from judge import MockJudge
from models import Completion
from replay import dominant_model, load_telemetry
from replay_pipeline import run_replay_proof


class _FixedCandidate:
    """Stub candidate model: always returns the same text (so a crafted
    original_output matches or diverges deterministically)."""
    provider, model = "anthropic", "claude-haiku-4-5"

    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, prompt, max_tokens=1024, temperature=0.0):
        return Completion(text=self._text, input_tokens=5, output_tokens=5)


_MATCH = "the answer is forty two point zero exactly"
_DIVERGE = "zzz qqq www totally unrelated tokens here"


def _telemetry(tmp_path, n_match, n_diverge, model="claude-opus-4-7"):
    rows = []
    for i in range(n_match):
        rows.append({"session_id": f"m{i}", "prompt": f"q{i}", "output": _MATCH,
                     "provider": "anthropic", "model": model,
                     "input_tokens": 1000, "output_tokens": 200})
    for i in range(n_diverge):
        rows.append({"session_id": f"d{i}", "prompt": f"q{i}", "output": _DIVERGE,
                     "provider": "anthropic", "model": model,
                     "input_tokens": 1000, "output_tokens": 200})
    p = tmp_path / "telemetry.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


# -- loaders --

def test_load_jsonl_skips_turns_missing_a_side(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        json.dumps({"prompt": "a", "output": "b", "model": "m"}) + "\n"
        + json.dumps({"prompt": "a"}) + "\n"            # no output → skipped
        + json.dumps({"output": "b"}) + "\n"            # no prompt → skipped
    )
    turns = load_telemetry(p)
    assert len(turns) == 1 and turns[0].prompt == "a" and turns[0].original_output == "b"


def test_load_json_array_form(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([
        {"prompt": "a", "output": "b", "provider": "openai", "model": "gpt-4o"},
    ]))
    turns = load_telemetry(p)
    assert len(turns) == 1 and turns[0].provider == "openai"


def test_dominant_model_picks_the_majority(tmp_path):
    p = _telemetry(tmp_path, 3, 1)
    turns = load_telemetry(p)
    assert dominant_model(turns) == ("anthropic", "claude-opus-4-7")


# -- replay proof --

def test_replay_equivalent_candidate_no_significant_divergence(tmp_path):
    p = _telemetry(tmp_path, n_match=12, n_diverge=0)
    result = run_replay_proof(
        telemetry_path=str(p), mock=True, judge=MockJudge(threshold=0.5),
        candidate_client=_FixedCandidate(_MATCH))
    assert result.benchmark == "replay"
    assert result.candidate_model == "anthropic:claude-haiku-4-5"   # TokenJam's downgrade
    assert result.recommended_by == "tokenjam.DOWNGRADE_CANDIDATES (replay)"
    assert result.n_tasks == 12
    assert result.candidate_pass == 12                  # all equivalent to original
    assert result.stats.verdict == "no_significant_regression"
    assert result.cost_delta_pct < 0                    # candidate cheaper than historical


def test_replay_detects_significant_divergence(tmp_path):
    # 6 equivalent + 8 divergent → 8 discordant pairs → McNemar significant.
    p = _telemetry(tmp_path, n_match=6, n_diverge=8)
    result = run_replay_proof(
        telemetry_path=str(p), mock=True, judge=MockJudge(threshold=0.5),
        candidate_client=_FixedCandidate(_MATCH))
    assert result.candidate_pass == 6
    assert result.stats.mcnemar_b == 8 and result.stats.mcnemar_c == 0
    assert result.stats.mcnemar_p_value < 0.05
    assert result.stats.verdict == "significant_regression"


def test_replay_requires_candidate_when_no_downgrade(tmp_path):
    p = _telemetry(tmp_path, 2, 0, model="claude-haiku-4-5")   # no downgrade for haiku
    with pytest.raises(ValueError) as exc:
        run_replay_proof(telemetry_path=str(p), mock=True, judge=MockJudge(),
                         candidate_client=_FixedCandidate(_MATCH))
    assert "no downgrade candidate" in str(exc.value)


def test_replay_empty_telemetry_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError) as exc:
        run_replay_proof(telemetry_path=str(p), candidate_spec="anthropic:claude-haiku-4-5",
                         mock=True, judge=MockJudge(), candidate_client=_FixedCandidate(_MATCH))
    assert "No replayable turns" in str(exc.value)
