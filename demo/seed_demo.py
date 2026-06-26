#!/usr/bin/env python3
"""Seed a rich, realistic DEMO dataset into results/ (local-only, gitignored).

This is a demo *fixture*, not a fabrication of the verdict: every artifact is
built from per-task pass/fail outcomes and run through the REAL statistics
engine (`tjbench.pipeline.assemble_proof` → Wilson CI + exact McNemar + the same
hedged verdict logic). Cost is computed from realistic token counts at real
list prices. So the dashboard's numbers (CIs, p-values, savings, verdicts) are
genuinely derived from the data — the data is just synthetic.

DEV-ONLY: every artifact is stamped `demo: true`, and the production dashboard
(`scan_runs`) skips mock/demo artifacts — so seeded data is invisible there. To
populate the dashboard with *real* evidence, run `scripts/run_real_benchmarks.sh`
against a live provider key instead. This script is for local UI iteration only.

Run:  python3 demo/seed_demo.py    (from the repo root)
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from tjbench.pipeline import assemble_proof
from tjbench.report import TaskOutcome

RNG = random.Random(20260625)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

# Real public list prices (USD per 1M tokens) → honest savings figures.
RATES = {
    "anthropic:claude-opus-4-7": (15.0, 75.0),
    "anthropic:claude-sonnet-4-6": (3.0, 15.0),
    "anthropic:claude-haiku-4-5": (0.8, 4.0),
    "openai:gpt-4o": (2.5, 10.0),
    "openai:gpt-4o-mini": (0.15, 0.6),
    "openai:gpt-4.1": (2.0, 8.0),
    "openai:gpt-4.1-mini": (0.4, 1.6),
    "deepseek:deepseek-reasoner": (0.55, 2.19),
    "deepseek:deepseek-chat": (0.27, 1.10),
    "google:gemini-1.5-pro": (1.25, 5.0),
    "google:gemini-1.5-flash": (0.075, 0.30),
}

# Per-benchmark token profile: (in_lo, in_hi, out_lo, out_hi) per task.
TOKPROFILE = {
    "humaneval": (1400, 3200, 400, 1100),
    "mbpp": (900, 2200, 300, 800),
    "swe-bench-lite": (3500, 9000, 700, 1900),
    "gsm8k": (380, 900, 220, 640),
    "judged": (800, 2100, 320, 900),
    "replay": (1000, 2600, 300, 820),
    "coding-assistant": (3200, 7800, 650, 1500),
    "rag-support": (1800, 4200, 300, 760),
    "research-agent": (2600, 6400, 520, 1240),
    "browser-agent": (2200, 5600, 420, 980),
    "customer-support": (600, 1800, 180, 560),
    "enterprise-rag": (1200, 3600, 200, 640),
    "email-assistant": (500, 1600, 160, 520),
    "research-assistant": (1800, 4800, 260, 760),
    "n8n": (1400, 4200, 300, 760),
    "coding-workflow": (2600, 7000, 500, 1300),
}

# Difficulty + ground-truth provenance per benchmark (demo metadata).
META = {
    "humaneval": ("Code synthesis", "medium", "OpenAI HumanEval (164 problems)"),
    "mbpp": ("Code synthesis", "easy", "MBPP (974 problems)"),
    "swe-bench-lite": ("Repo bug-fix", "hard", "SWE-bench Lite (300 instances)"),
    "gsm8k": ("Grade-school math", "easy", "GSM8K (1319 problems)"),
    "judged": ("Open-ended QA", "medium", "LLM-judged (DeepEval)"),
    "replay": ("Production replay", "hard", "Captured TokenJam telemetry"),
    "coding-assistant": ("Agentic coding", "hard", "Scenario suite (gated tools)"),
    "rag-support": ("RAG support", "medium", "Scenario suite (gated tools)"),
    "research-agent": ("Research agent", "hard", "Scenario suite (gated tools)"),
    "browser-agent": ("Browser agent", "hard", "Scenario suite (gated tools)"),
    "customer-support": ("Customer support", "medium", "Support tickets (16 grounded)"),
    "enterprise-rag": ("Enterprise RAG", "medium", "Internal KB (HR/Eng/Legal/Product)"),
    "email-assistant": ("Email assistant", "easy", "Inbox tasks (12)"),
    "research-assistant": ("Research assistant", "hard", "Source synthesis (12)"),
    "n8n": ("n8n automation", "hard", "Automation workflows (gated tools)"),
    "coding-workflow": ("Coding workflow", "hard", "PR-shaped coding tasks (gated)"),
}

FAIL_CATS = {
    "humaneval": ["edge-case", "off-by-one", "type-error", "timeout"],
    "mbpp": ["edge-case", "wrong-output", "syntax"],
    "swe-bench-lite": ["incomplete-patch", "broke-test", "wrong-file", "regression"],
    "gsm8k": ["arithmetic", "misread-problem", "wrong-units"],
    "judged": ["unfaithful", "incomplete", "off-topic", "hallucination"],
    "replay": ["semantic-drift", "format-change", "tone-shift", "tool-misuse"],
    "coding-assistant": ["wrong-edit", "missing-test", "unsafe-action"],
    "rag-support": ["wrong-answer", "missed-citation", "unsafe-action"],
    "research-agent": ["shallow", "unsafe-publish", "missed-source"],
    "browser-agent": ["wrong-extract", "unsafe-purchase", "nav-fail"],
    "customer-support": ["wrong-policy", "missed-context", "off-tone", "unsafe-action"],
    "enterprise-rag": ["ungrounded", "wrong-doc", "missed-citation", "hallucination"],
    "email-assistant": ["wrong-task", "off-tone", "incomplete", "missed-detail"],
    "research-assistant": ["shallow", "fabricated-cite", "missed-source", "unsupported-claim"],
    "n8n": ["wrong-tool", "wrong-order", "unsafe-action", "incomplete"],
    "coding-workflow": ["broke-test", "wrong-edit", "unsafe-action", "no-commit"],
}


def price(spec: str, in_tok: int, out_tok: int) -> float:
    inr, outr = RATES.get(spec, (0.5, 2.0))
    return in_tok / 1e6 * inr + out_tok / 1e6 * outr


def make_outcomes(bench, orig_spec, cand_spec, a, b, c, d):
    """a=both pass, b=orig-pass/cand-fail, c=orig-fail/cand-pass, d=both fail."""
    cells = ["aa"] * a + ["bc"] * b + ["cb"] * c + ["dd"] * d
    RNG.shuffle(cells)
    lo_i, hi_i, lo_o, hi_o = TOKPROFILE[bench]
    out = []
    for i, cell in enumerate(cells):
        op = 1 if cell in ("aa", "bc") else 0
        cp = 1 if cell in ("aa", "cb") else 0
        oi = RNG.randint(lo_i, hi_i)
        oo = RNG.randint(lo_o, hi_o)
        ci = oi + RNG.randint(-oi // 12, oi // 12)
        # Recommended downgrade targets are typically terser (smaller / non-reasoning
        # model), so the candidate emits fewer output tokens — part of the saving.
        co = int(oo * RNG.uniform(0.55, 0.9))
        out.append(TaskOutcome(
            task_id=f"{bench}/{i:03d}", samples=1,
            original_passes=op, candidate_passes=cp,
            original_cost_usd=round(price(orig_spec, oi, oo), 8),
            candidate_cost_usd=round(price(cand_spec, ci, co), 8),
            original_output_tokens=oo, candidate_output_tokens=co,
        ))
    return out


def latency(bench, cheap=False):
    base = {"swe-bench-lite": 42000, "coding-assistant": 28000, "research-agent": 22000,
            "browser-agent": 18000, "humaneval": 9000, "mbpp": 6000, "judged": 11000,
            "replay": 7000, "rag-support": 8000, "gsm8k": 4200,
            "customer-support": 5200, "enterprise-rag": 6000, "email-assistant": 3800,
            "research-assistant": 14000, "n8n": 9000, "coding-workflow": 24000}[bench]
    base = base * (0.42 if cheap else 1.0)
    return int(base * RNG.uniform(0.85, 1.15))


def fail_breakdown(bench, n_fail):
    cats = FAIL_CATS[bench]
    if n_fail <= 0:
        return []
    weights = [RNG.random() for _ in cats]
    s = sum(weights) or 1
    alloc = [round(n_fail * w / s) for w in weights]
    while sum(alloc) > n_fail:
        alloc[alloc.index(max(alloc))] -= 1
    while sum(alloc) < n_fail:
        alloc[RNG.randrange(len(alloc))] += 1
    return [{"category": cat, "count": ct} for cat, ct in zip(cats, alloc) if ct]


def write_artifact(cfg, bench, version, ts, n, orig_rate, cand_rate, b, c, *, mock):
    orig_spec, cand_spec, recby = cfg
    orig_pass = round(n * orig_rate)
    cand_pass = round(n * cand_rate)
    # solve a,b,c,d from marginals + discordance (b, c given)
    a = orig_pass - b            # both pass
    d = n - a - b - c            # both fail
    if a < 0 or d < 0:
        a = max(0, a)
        d = max(0, n - a - b - c)
    outcomes = make_outcomes(bench, orig_spec, cand_spec, a, b, c, d)
    res = assemble_proof(
        outcomes, benchmark_name=bench, original_spec=orig_spec,
        candidate_spec=cand_spec, recommended_by=recby, samples=1, mock=mock,
        orig_provider=orig_spec.split(":")[0], orig_model=orig_spec.split(":")[1],
        cand_provider=cand_spec.split(":")[0], cand_model=cand_spec.split(":")[1],
        sample_pass_totals=(orig_pass, cand_pass),
    )
    res.tokenjam_version = version
    res.created_at = ts
    d_out = res.to_dict()
    # ---- demo enrichment fields (synthetic; real runs simply omit these) ----
    cat, diff, gt = META[bench]
    n_fail = n - res.candidate_pass
    olat, clat = latency(bench), latency(bench, cheap=True)
    d_out["demo"] = True
    d_out["task_category"] = cat
    d_out["difficulty"] = diff
    d_out["ground_truth"] = gt
    d_out["ground_truth_size"] = n
    d_out["coverage_pct"] = round(RNG.uniform(82, 99), 1)
    d_out["latency_ms_original"] = olat
    d_out["latency_ms_candidate"] = clat
    d_out["latency_saved_pct"] = round((1 - clat / olat) * 100, 1)
    d_out["failure_categories"] = fail_breakdown(bench, n_fail)
    if bench == "judged":
        base = res.candidate_pass_rate
        d_out["judge"] = {
            "correctness": round(min(1, base + RNG.uniform(-0.03, 0.04)), 3),
            "faithfulness": round(min(1, base + RNG.uniform(-0.06, 0.03)), 3),
            "answer_relevancy": round(min(1, base + RNG.uniform(-0.02, 0.05)), 3),
            "task_completion": round(min(1, base + RNG.uniform(-0.05, 0.03)), 3),
            "reasoning_quality": round(min(1, base + RNG.uniform(-0.07, 0.02)), 3),
            "judge_agreement": round(RNG.uniform(0.86, 0.96), 3),
            "hallucination_rate": round(RNG.uniform(0.5, 4.5), 2),
            "citation_accuracy": round(RNG.uniform(0.88, 0.98), 3),
        }
    if bench == "replay":
        d_out["semantic_match_rate"] = round(res.candidate_pass_rate * 100, 1)
        d_out["behavior_match_rate"] = round(min(100, res.candidate_pass_rate * 100 + RNG.uniform(1, 5)), 1)
        d_out["critical_failures"] = int(b * RNG.uniform(0, 0.4))
        d_out["replay_diffs"] = [
            {"prompt": "Summarize the refund policy for EU orders.",
             "original": "EU orders: 30-day return window, free return shipping.",
             "candidate": "EU orders have a 30-day return window with free return shipping.",
             "match": "equivalent"},
            {"prompt": "What is the status of order #4821?",
             "original": "Order #4821 shipped Tuesday, arriving Thursday.",
             "candidate": "Order #4821 is in transit; expected Thursday.",
             "match": "equivalent"},
            {"prompt": "Compute the 14% APR monthly interest on $2,400.",
             "original": "Monthly interest = 2400 * 0.14/12 = $28.00.",
             "candidate": "About $28 per month.",
             "match": "divergent" if b else "equivalent"},
        ]
    if bench in ("coding-assistant", "rag-support", "research-agent", "browser-agent"):
        d_out["expected_tool_calls"] = RNG.randint(3, 6)
        d_out["avg_runtime_s"] = round(olat / 1000, 1)
        d_out["pass_threshold"] = 0.8
        d_out["safety_gate"] = "enforced"
        d_out["unsafe_actions_blocked"] = RNG.randint(0, 3)
        d_out["risk_category"] = {"coding-assistant": "code-mutation", "rag-support": "financial",
                                  "research-agent": "publication", "browser-agent": "payment"}[bench]
    if bench in ("customer-support", "enterprise-rag", "email-assistant",
                 "research-assistant", "n8n", "coding-workflow"):
        d_out["pass_threshold"] = 0.8
        d_out["workflow"] = True
        if bench in ("customer-support", "n8n", "coding-workflow"):
            d_out["safety_gate"] = "enforced"
            d_out["unsafe_actions_blocked"] = RNG.randint(0, 3)

    fn = OUT / f"tjbench_{bench}_tj{version}_{int(ts)}_{RNG.randrange(1_000_000):06d}.json"
    fn.write_text(json.dumps(d_out, indent=2))
    return res.stats.verdict


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # fresh demo set
    for p in OUT.glob("*.json"):
        p.unlink()
    for p in OUT.glob("*.duckdb*"):
        p.unlink()

    DAY = 86400
    now = time.time()
    V = {"0.4.2": now - 38 * DAY, "0.5.0": now - 17 * DAY, "0.5.1": now - 3 * DAY}

    # (original, candidate, recommended_by)
    A = ("anthropic:claude-opus-4-7", "anthropic:claude-haiku-4-5", "tokenjam.DOWNGRADE_CANDIDATES")
    S = ("anthropic:claude-sonnet-4-6", "anthropic:claude-haiku-4-5", "tokenjam.DOWNGRADE_CANDIDATES")
    O_ = ("openai:gpt-4o", "openai:gpt-4o-mini", "tokenjam.DOWNGRADE_CANDIDATES")
    O41 = ("openai:gpt-4.1", "openai:gpt-4.1-mini", "tokenjam.DOWNGRADE_CANDIDATES")
    D = ("deepseek:deepseek-reasoner", "deepseek:deepseek-chat", "tokenjam.DOWNGRADE_CANDIDATES")
    G = ("google:gemini-1.5-pro", "google:gemini-1.5-flash", "tokenjam.DOWNGRADE_CANDIDATES")

    count = {"runs": 0, "reg": 0}

    def emit(cfg, bench, ver, n, orate, crate, b, c, *, mock=False, jitter=0):
        ts = V[ver] + jitter * 3600 + RNG.uniform(-6, 6) * 3600
        v = write_artifact(cfg, bench, ver, ts, n, orate, crate, b, c, mock=mock)
        count["runs"] += 1
        if v == "significant_regression":
            count["reg"] += 1

    # ---- flagship config (opus→haiku): broad, cleared, the recommended hero ----
    for ver in ("0.4.2", "0.5.0", "0.5.1"):
        emit(A, "humaneval", ver, 164, 0.915, 0.909, b=6, c=5, jitter=1)
        emit(A, "mbpp", ver, 120, 0.95, 0.942, b=4, c=3, jitter=2)
        emit(A, "gsm8k", ver, 130, 0.93, 0.946, b=3, c=5, jitter=3)
        emit(A, "swe-bench-lite", ver, 60, 0.667, 0.633, b=5, c=3, jitter=4)
    for suite in ("coding-assistant", "rag-support", "research-agent", "browser-agent"):
        emit(A, suite, "0.5.1", 36, 0.917, 0.889, b=3, c=2, jitter=5)
    emit(A, "replay", "0.5.0", 90, 0.94, 0.933, b=5, c=4, jitter=6)
    emit(A, "replay", "0.5.1", 120, 0.95, 0.95, b=4, c=4, jitter=6)

    # ---- Production Workflows: text (judge-scored) + agentic (AgentRunner) ----
    emit(A, "customer-support", "0.5.0", 16, 0.94, 0.94, b=1, c=1, jitter=7)
    emit(A, "customer-support", "0.5.1", 16, 0.94, 0.94, b=1, c=1, jitter=7)
    emit(S, "customer-support", "0.5.1", 16, 0.93, 0.94, b=1, c=2, jitter=7)
    emit(O_, "customer-support", "0.5.1", 16, 0.92, 0.90, b=2, c=1, jitter=7)
    emit(A, "enterprise-rag", "0.5.0", 60, 0.93, 0.917, b=3, c=2, jitter=7)
    emit(A, "enterprise-rag", "0.5.1", 60, 0.93, 0.933, b=2, c=2, jitter=7)
    emit(O_, "enterprise-rag", "0.5.1", 60, 0.92, 0.90, b=3, c=2, jitter=7)
    emit(A, "email-assistant", "0.5.1", 48, 0.958, 0.958, b=1, c=1, jitter=7)
    emit(S, "email-assistant", "0.5.1", 48, 0.938, 0.958, b=1, c=2, jitter=7)
    emit(A, "research-assistant", "0.5.1", 36, 0.889, 0.861, b=3, c=2, jitter=7)
    emit(D, "research-assistant", "0.5.1", 36, 0.861, 0.833, b=4, c=3, jitter=7)
    emit(A, "n8n", "0.5.0", 40, 0.95, 0.95, b=1, c=1, jitter=8)
    emit(A, "n8n", "0.5.1", 40, 0.95, 0.95, b=1, c=1, jitter=8)
    emit(O_, "n8n", "0.5.1", 40, 0.90, 0.625, b=13, c=2, jitter=8)            # REGRESSION (unsafe/order)
    emit(A, "coding-workflow", "0.5.1", 30, 0.90, 0.867, b=3, c=2, jitter=8)
    emit(S, "coding-workflow", "0.5.1", 30, 0.90, 0.90, b=2, c=2, jitter=8)

    # ---- sonnet→haiku: cleared ----
    for ver in ("0.5.0", "0.5.1"):
        emit(S, "humaneval", ver, 164, 0.902, 0.902, b=7, c=7, jitter=2)
        emit(S, "gsm8k", ver, 120, 0.925, 0.933, b=4, c=5, jitter=3)
    emit(S, "swe-bench-lite", "0.5.1", 60, 0.65, 0.633, b=4, c=3, jitter=4)

    # ---- gpt-4o→mini: a regression in 0.5.0 (swe), resolved in 0.5.1 ----
    emit(O_, "humaneval", "0.4.2", 164, 0.89, 0.872, b=9, c=6, jitter=1)
    emit(O_, "humaneval", "0.5.0", 164, 0.89, 0.884, b=7, c=6, jitter=1)
    emit(O_, "humaneval", "0.5.1", 164, 0.896, 0.89, b=6, c=5, jitter=1)
    emit(O_, "swe-bench-lite", "0.5.0", 80, 0.65, 0.525, b=14, c=4, jitter=4)   # REGRESSION
    emit(O_, "swe-bench-lite", "0.5.1", 80, 0.65, 0.625, b=6, c=4, jitter=4)    # resolved
    emit(O_, "gsm8k", "0.5.1", 130, 0.94, 0.946, b=4, c=5, jitter=3)
    emit(O_, "judged", "0.5.1", 80, 0.90, 0.888, b=5, c=4, jitter=7)
    emit(O_, "replay", "0.5.1", 110, 0.93, 0.927, b=5, c=4, jitter=6)

    # ---- gpt-4.1→mini: cleared ----
    emit(O41, "humaneval", "0.5.1", 164, 0.91, 0.902, b=6, c=5, jitter=1)
    emit(O41, "mbpp", "0.5.1", 120, 0.95, 0.95, b=3, c=3, jitter=2)

    # ---- deepseek reasoner→chat: judged + replay, cleared ----
    emit(D, "judged", "0.5.0", 70, 0.886, 0.871, b=6, c=5, jitter=7)
    emit(D, "judged", "0.5.1", 90, 0.90, 0.911, b=4, c=5, jitter=7)
    emit(D, "humaneval", "0.5.1", 164, 0.88, 0.872, b=8, c=6, jitter=1)
    emit(D, "replay", "0.5.1", 100, 0.94, 0.94, b=4, c=4, jitter=6)

    # ---- gemini pro→flash: a CURRENT regression on gsm8k (0.5.1) ----
    emit(G, "humaneval", "0.5.1", 164, 0.86, 0.848, b=8, c=6, jitter=1)
    emit(G, "gsm8k", "0.5.1", 130, 0.92, 0.80, b=18, c=2, jitter=3)           # REGRESSION
    emit(G, "mbpp", "0.5.1", 120, 0.93, 0.917, b=5, c=3, jitter=2)

    # ---- ingest into the historical DB ----
    try:
        from tjbench.history import BenchmarkHistory
        with BenchmarkHistory(OUT / "history.duckdb") as h:
            new, total = h.ingest_dir(OUT)
    except Exception as e:  # pragma: no cover
        new = total = -1
        print("history ingest skipped:", e)

    print(f"seeded {count['runs']} artifacts ({count['reg']} significant regressions) "
          f"into {OUT}/  · history: {new} new / {total} total")


if __name__ == "__main__":
    main()
