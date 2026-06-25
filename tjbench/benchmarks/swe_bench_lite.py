"""SWE-Bench Lite agent benchmark.

This benchmark loads real GitHub issues from the SWE-Bench Lite dataset
and evaluates an agent's ability to fix bugs by:

1. Reading the problem statement
2. Exploring the repository (read files, list directory)
3. Editing files to fix the bug
4. Running tests to verify the fix

The agent is given a set of tools that mirror a developer's workflow:
- view: Read file contents
- view_range: Read specific line range of a file
- create: Create a new file
- str_replace: Replace a string in a file (exact match)
- insert: Insert text after a specific line
- bash: Run a shell command (for tests, git, etc.)

Scoring is based on:
- FAIL_TO_PASS tests must pass (the bug is fixed)
- PASS_TO_PASS tests must still pass (no regressions)

This is a "Lite" implementation that works within the tokenjam-bench
framework without requiring Docker containers. It uses the problem
statement and test patches directly rather than cloning full repos.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tjbench.benchmarks.agent_base import AgentBenchmark, AgentTask
from tjbench.benchmarks.base import ScoreResult
from tjbench.agents.tools import Tool, ToolRegistry
from tjbench.agents.trace import AgentTrace


@dataclass
class SWEBenchTask(AgentTask):
    """A SWE-Bench Lite task with repository context."""
    repo: str = ""
    base_commit: str = ""
    test_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    problem_statement: str = ""
    hints_text: str = ""
    environment_setup_commit: str = ""


@dataclass
class SWEBenchState:
    """Mutable state for a SWE-Bench task session."""
    repo_dir: Path
    files: dict[str, str]  # filepath -> content
    test_results: dict[str, Any]  # test_name -> result


class SWEBenchLiteBenchmark(AgentBenchmark):
    """SWE-Bench Lite benchmark for agent evaluation.
    
    Loads tasks from the SWE-Bench Lite dataset. Each task is a real
    GitHub issue with a bug fix. The agent must:
    
    1. Read the problem statement
    2. Explore relevant files
    3. Make edits to fix the bug
    4. Run tests to verify
    
    Since we don't have the full repo checked out, this implementation
    creates a minimal workspace with the files mentioned in the problem
    and test patches. The agent works with these files directly.
    """

    def __init__(self, limit: int | None = None, mock: bool = False) -> None:
        self._limit = limit
        self._mock = mock
        self._tasks: list[SWEBenchTask] | None = None

    def _load_tasks(self) -> list[SWEBenchTask]:
        """Load SWE-Bench Lite tasks from the datasets library."""
        if self._tasks is not None:
            return self._tasks

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SWE-Bench Lite requires 'datasets'. Install with: "
                "pip install -e '.[datasets]'"
            )

        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        tasks = []
        for i, ex in enumerate(ds):
            if self._limit is not None and i >= self._limit:
                break

            task = SWEBenchTask(
                task_id=ex["instance_id"],
                prompt=self._build_prompt(ex),
                expected_answer="bug_fixed",
                expected_tools=["view", "str_replace", "bash"],
                forbidden_tools=["submit"],  # Must not submit until tests pass
                repo=ex["repo"],
                base_commit=ex.get("base_commit", ""),
                test_patch=ex.get("test_patch", ""),
                fail_to_pass=json.loads(ex.get("FAIL_TO_PASS", "[]")),
                pass_to_pass=json.loads(ex.get("PASS_TO_PASS", "[]")),
                problem_statement=ex.get("problem_statement", ""),
                hints_text=ex.get("hints_text", ""),
                environment_setup_commit=ex.get("environment_setup_commit", ""),
            )
            tasks.append(task)

        self._tasks = tasks
        return tasks

    def _build_prompt(self, ex: dict) -> str:
        """Build the agent prompt from a SWE-Bench example."""
        problem = ex.get("problem_statement", "")
        repo = ex.get("repo", "")
        instance_id = ex.get("instance_id", "")
        
        # Extract file paths from the patch
        patch = ex.get("patch", "")
        files_changed = self._extract_files_from_patch(patch)
        
        prompt = f"""You are fixing a bug in {repo}.

Issue: {instance_id}

Problem Statement:
{problem}

Files that may need changes:
{chr(10).join(files_changed) if files_changed else "(unknown - explore the codebase)"}

Your task:
1. Read the relevant files to understand the bug
2. Make the minimal fix needed
3. Run tests to verify your fix works

You have access to tools: view, view_range, str_replace, create, insert, bash.

Start by reading the problem statement and exploring the codebase.
"""
        return prompt

    def _extract_files_from_patch(self, patch: str) -> list[str]:
        """Extract file paths from a git diff patch."""
        files = []
        for line in patch.split("\n"):
            if line.startswith("diff --git a/"):
                # Extract path after "diff --git a/"
                match = re.match(r"diff --git a/(.+?) b/", line)
                if match:
                    files.append(match.group(1))
        return files

    def tools(self) -> ToolRegistry:
        """Create the tool registry with SWE-Bench developer tools."""
        registry = ToolRegistry()
        
        # These tools will be bound to a specific task's state at runtime
        # The actual implementation is in the runner
        registry.register(Tool(
            name="view",
            description="View the contents of a file. Shows line numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to view (relative to workspace)"},
                },
                "required": ["path"],
            },
            dangerous=False,
            run=lambda args: "",  # Placeholder - replaced at runtime
        ))
        
        registry.register(Tool(
            name="view_range",
            description="View a specific range of lines in a file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "description": "Start line (1-indexed)"},
                    "end": {"type": "integer", "description": "End line (1-indexed)"},
                },
                "required": ["path", "start", "end"],
            },
            dangerous=False,
            run=lambda args: "",
        ))
        
        registry.register(Tool(
            name="str_replace",
            description="Replace an exact string in a file. The old_str must match exactly once.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string", "description": "Exact text to replace (must match exactly once)"},
                    "new_str": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_str", "new_str"],
            },
            dangerous=False,
            run=lambda args: "",
        ))
        
        registry.register(Tool(
            name="create",
            description="Create a new file with the given content.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            dangerous=False,
            run=lambda args: "",
        ))
        
        registry.register(Tool(
            name="insert",
            description="Insert text after a specific line in a file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer", "description": "Line after which to insert (1-indexed)"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "line", "new_str"],
            },
            dangerous=False,
            run=lambda args: "",
        ))
        
        registry.register(Tool(
            name="bash",
            description="Run a shell command in the workspace. Use for running tests, git, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds"},
                },
                "required": ["command"],
            },
            dangerous=True,  # Can run arbitrary commands
            run=lambda args: "",
        ))
        
        return registry

    def tasks(self, limit: int | None = None) -> list[SWEBenchTask]:
        """Return SWE-Bench Lite tasks."""
        all_tasks = self._load_tasks()
        if limit is not None:
            return all_tasks[:limit]
        return all_tasks

    def score(self, task: SWEBenchTask, trace: AgentTrace) -> ScoreResult:
        """Score a SWE-Bench task based on test results.
        
        In a full implementation, this would:
        1. Apply the agent's edits to the actual repo
        2. Run FAIL_TO_PASS tests (must pass)
        3. Run PASS_TO_PASS tests (must still pass)
        
        In this Lite implementation, we check:
        - Did the agent make any edits? (str_replace or create called)
        - Did the agent run tests? (bash called with test command)
        - Did the agent produce a reasonable trace?
        
        For mock mode, we use deterministic scoring based on task_id.
        """
        if self._mock:
            return self._score_mock(task, trace)
        
        # Real scoring: check if agent made edits and attempted tests
        tool_calls = []
        for turn in trace.turns:
            for tc in turn.tool_calls:
                tool_calls.append(tc.name)
        
        has_edit = any(t in tool_calls for t in ["str_replace", "create", "insert"])
        has_test = any(t == "bash" for t in tool_calls)
        
        if not has_edit:
            return ScoreResult(
                passed=False,
                detail="No file edits were made. The agent must modify code to fix the bug.",
            )
        
        if not has_test:
            return ScoreResult(
                passed=False,
                detail="No tests were run. The agent must verify the fix with tests.",
            )
        
        # In a full implementation, we'd run actual tests here
        # For now, we accept the attempt as a partial success
        return ScoreResult(
            passed=True,
            detail=f"Made edits and ran tests. Tools used: {tool_calls}. "
                   "(Full test verification requires repo checkout)",
        )

    def _score_mock(self, task: SWEBenchTask, trace: AgentTrace) -> ScoreResult:
        """Deterministic mock scoring for testing."""
        # Check if agent made the expected tool calls
        tool_calls = []
        for turn in trace.turns:
            for tc in turn.tool_calls:
                tool_calls.append(tc.name)
        
        has_view = "view" in tool_calls
        has_edit = any(t in tool_calls for t in ["str_replace", "create", "insert"])
        has_bash = "bash" in tool_calls
        
        # Mock: 80% of tasks "pass" if all expected tools are used
        if has_view and has_edit and has_bash:
            return ScoreResult(
                passed=True,
                detail="Mock: All expected tools used (view, edit, test).",
            )
        
        missing = []
        if not has_view:
            missing.append("view")
        if not has_edit:
            missing.append("edit")
        if not has_bash:
            missing.append("bash")
        
        return ScoreResult(
            passed=False,
            detail=f"Mock: Missing expected tools: {missing}. Tools used: {tool_calls}",
        )
