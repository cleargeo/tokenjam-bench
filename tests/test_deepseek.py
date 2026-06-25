"""DeepSeek provider + judge backend — offline-verifiable parts (no live key).

Also guards that no API key is ever committed to the repo.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from judge import get_judge
from models.openai_compatible import (
    PROVIDERS,
    OpenAICompatibleClient,
    is_openai_compatible,
)
from models.registry import get_client


def test_deepseek_is_a_registered_openai_compatible_provider():
    assert "deepseek" in PROVIDERS
    prov = PROVIDERS["deepseek"]
    assert prov.base_url == "https://api.deepseek.com"
    assert prov.api_key_env == "DEEPSEEK_API_KEY"      # env var, not a literal key
    assert is_openai_compatible("deepseek") and is_openai_compatible("openai")


def test_registry_resolves_deepseek_through_the_shared_abstraction():
    client = get_client("deepseek:deepseek-chat")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "deepseek" and client.model == "deepseek-chat"


def test_missing_key_raises_without_leaking_anything(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = OpenAICompatibleClient("deepseek-chat", "deepseek")
    with pytest.raises(RuntimeError) as exc:
        client.complete("hi")
    msg = str(exc.value)
    assert "DEEPSEEK_API_KEY" in msg          # names the env var
    assert "sk-" not in msg                     # never echoes a key


def test_key_read_from_env_not_stored_on_instance(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-not-be-stored")
    client = OpenAICompatibleClient("deepseek-chat", "deepseek")
    # The key must not live on the client object anywhere.
    assert "sk-should-not-be-stored" not in repr(vars(client))


def test_deepseek_judge_backend_constructs_lazily():
    j = get_judge("deepseek", metric="correctness")
    assert j.name == "deepeval"               # DeepEval framework...
    assert j.provider == "deepseek"           # ...backed by DeepSeek
    assert j.model == "deepseek-chat"


def test_no_api_key_is_committed_in_the_repo():
    root = Path(__file__).resolve().parent.parent
    skip = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "results", ".venv"}
    key_re = re.compile(r"sk-[A-Za-z0-9]{16,}")
    offenders = []
    for p in root.rglob("*"):
        if not p.is_file() or any(part in skip for part in p.parts):
            continue
        if p.suffix not in {".py", ".md", ".toml", ".txt", ".json", ".cfg", ".sh"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # allow the placeholder used in tests
        for m in key_re.findall(text):
            if m != "sk-should-not-be-stored":
                offenders.append(f"{p.name}: {m[:6]}...")
    assert not offenders, f"possible committed secrets: {offenders}"
