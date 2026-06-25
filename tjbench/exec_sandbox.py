"""Guarded execution of model-generated code for benchmark scoring.

Executable benchmarks (HumanEval-style) work by running the model's code against
a hidden test suite — pass/fail is the objective ground truth. That necessarily
means executing untrusted, model-generated code. This runs it in a separate
Python subprocess with a hard wall-clock timeout, in a throwaway temp directory.

SECURITY: this executes generated code. It is intended ONLY for running known,
trusted benchmark suites (HumanEval / GSM8K / your own task sets) on a machine
you control — the same pattern HumanEval's official harness uses. Do not point
it at adversarial inputs. A future hardening pass can add an OS-level sandbox
(seccomp / container); the subprocess + timeout boundary is the v1.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class ExecResult:
    passed: bool
    detail: str


def run_python(program: str, timeout_s: float = 10.0) -> ExecResult:
    """Run `program` in a fresh subprocess. Pass = exit code 0 within timeout.

    The program is expected to raise (non-zero exit) on a failed assertion, so
    benchmark test harnesses that `assert candidate(...) == expected` map
    directly onto pass/fail.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, f"timeout after {timeout_s}s")
        except Exception as exc:  # pragma: no cover - defensive
            return ExecResult(False, f"exec error: {exc}")

    if proc.returncode == 0:
        return ExecResult(True, "ok")
    err = (proc.stderr or proc.stdout or "").strip()
    # Keep the last line of the traceback — enough to triage without flooding.
    last = err.splitlines()[-1] if err else f"exit code {proc.returncode}"
    return ExecResult(False, last[:200])
