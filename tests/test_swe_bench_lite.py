"""Tests for SWE-Bench Lite integration (offline, no keys, no dataset download)."""

from __future__ import annotations


from tjbench.benchmarks.swe_bench_lite import SWEBenchLiteBenchmark, SWEBenchTask
from tjbench.agents.trace import AgentTrace, TurnRecord, ToolCallRecord


class TestSWEBenchLiteBenchmark:
    """Test SWE-Bench Lite benchmark without downloading the dataset."""

    def test_mock_scoring_with_all_tools(self):
        """Agent that uses view, str_replace, and bash should pass in mock mode."""
        benchmark = SWEBenchLiteBenchmark(mock=True)
        
        task = SWEBenchTask(
            task_id="test__repo-12345",
            prompt="Fix the bug",
            expected_answer="bug_fixed",
            expected_tools=["view", "str_replace", "bash"],
            forbidden_tools=["submit"],
        )
        
        trace = AgentTrace(
            task_id="test__repo-12345",
            turns=[
                TurnRecord(
                    index=0,
                    assistant_text="Let me look at the file.",
                    tool_calls=[
                        ToolCallRecord(name="view", arguments={"path": "foo.py"}, result="", is_error=False),
                    ],
                    input_tokens=100,
                    output_tokens=50,
                ),
                TurnRecord(
                    index=1,
                    assistant_text="Now I'll fix it.",
                    tool_calls=[
                        ToolCallRecord(name="str_replace", arguments={"path": "foo.py", "old_str": "x", "new_str": "y"}, result="", is_error=False),
                    ],
                    input_tokens=100,
                    output_tokens=50,
                ),
                TurnRecord(
                    index=2,
                    assistant_text="Let me run the tests.",
                    tool_calls=[
                        ToolCallRecord(name="bash", arguments={"command": "pytest"}, result="", is_error=False),
                    ],
                    input_tokens=100,
                    output_tokens=50,
                ),
            ],
            final_text="Fixed!",
            stopped_reason="completed",
        )
        
        result = benchmark.score(task, trace)
        assert result.passed is True
        assert "All expected tools used" in result.detail

    def test_mock_scoring_missing_view(self):
        """Agent that doesn't view files should fail in mock mode."""
        benchmark = SWEBenchLiteBenchmark(mock=True)
        
        task = SWEBenchTask(
            task_id="test__repo-12345",
            prompt="Fix the bug",
            expected_answer="bug_fixed",
            expected_tools=["view", "str_replace", "bash"],
            forbidden_tools=["submit"],
        )
        
        trace = AgentTrace(
            task_id="test__repo-12345",
            turns=[
                TurnRecord(
                    index=0,
                    assistant_text="I'll fix it directly.",
                    tool_calls=[
                        ToolCallRecord(name="str_replace", arguments={"path": "foo.py", "old_str": "x", "new_str": "y"}, result="", is_error=False),
                    ],
                    input_tokens=100,
                    output_tokens=50,
                ),
                TurnRecord(
                    index=1,
                    assistant_text="Run tests.",
                    tool_calls=[
                        ToolCallRecord(name="bash", arguments={"command": "pytest"}, result="", is_error=False),
                    ],
                    input_tokens=100,
                    output_tokens=50,
                ),
            ],
            final_text="Fixed!",
            stopped_reason="completed",
        )
        
        result = benchmark.score(task, trace)
        assert result.passed is False
        assert "Missing expected tools" in result.detail
        assert "view" in result.detail

    def test_mock_scoring_missing_edit(self):
        """Agent that doesn't edit files should fail in mock mode."""
        benchmark = SWEBenchLiteBenchmark(mock=True)
        
        task = SWEBenchTask(
            task_id="test__repo-12345",
            prompt="Fix the bug",
            expected_answer="bug_fixed",
            expected_tools=["view", "str_replace", "bash"],
            forbidden_tools=["submit"],
        )
        
        trace = AgentTrace(
            task_id="test__repo-12345",
            turns=[
                TurnRecord(
                    index=0,
                    assistant_text="Let me look.",
                    tool_calls=[
                        ToolCallRecord(name="view", arguments={"path": "foo.py"}, result="", is_error=False),
                    ],
                    input_tokens=100,
                    output_tokens=50,
                ),
                TurnRecord(
                    index=1,
                    assistant_text="Run tests.",
                    tool_calls=[
                        ToolCallRecord(name="bash", arguments={"command": "pytest"}, result="", is_error=False),
                    ],
                    input_tokens=100,
                    output_tokens=50,
                ),
            ],
            final_text="I see the bug but won't fix it.",
            stopped_reason="completed",
        )
        
        result = benchmark.score(task, trace)
        assert result.passed is False
        assert "edit" in result.detail

    def test_mock_scoring_missing_bash(self):
        """Agent that doesn't run tests should fail in mock mode."""
        benchmark = SWEBenchLiteBenchmark(mock=True)
        
        task = SWEBenchTask(
            task_id="test__repo-12345",
            prompt="Fix the bug",
            expected_answer="bug_fixed",
            expected_tools=["view", "str_replace", "bash"],
            forbidden_tools=["submit"],
        )
        
        trace = AgentTrace(
            task_id="test__repo-12345",
            turns=[
                TurnRecord(
                    index=0,
                    assistant_text="Let me look.",
                    tool_calls=[
                        ToolCallRecord(name="view", arguments={"path": "foo.py"}, result="", is_error=False),
                    ],
                    input_tokens=100,
                    output_tokens=50,
                ),
                TurnRecord(
                    index=1,
                    assistant_text="I'll fix it.",
                    tool_calls=[
                        ToolCallRecord(name="str_replace", arguments={"path": "foo.py", "old_str": "x", "new_str": "y"}, result="", is_error=False),
                    ],
                    input_tokens=100,
                    output_tokens=50,
                ),
            ],
            final_text="Fixed!",
            stopped_reason="completed",
        )
        
        result = benchmark.score(task, trace)
        assert result.passed is False
        assert "bash" in result.detail

    def test_tools_registry_has_swe_bench_tools(self):
        """The benchmark should provide SWE-Bench developer tools."""
        benchmark = SWEBenchLiteBenchmark(mock=True)
        registry = benchmark.tools()
        specs = registry.specs()
        names = [s["name"] for s in specs]
        
        assert "view" in names
        assert "view_range" in names
        assert "str_replace" in names
        assert "create" in names
        assert "insert" in names
        assert "bash" in names

    def test_extract_files_from_patch(self):
        """Patch parsing should extract file paths correctly."""
        benchmark = SWEBenchLiteBenchmark(mock=True)
        
        patch = """diff --git a/foo/bar.py b/foo/bar.py
--- a/foo/bar.py
+++ b/foo/bar.py
@@ -1,3 +1,3 @@
 def hello():
-    return "world"
+    return "hello world"

diff --git a/baz/qux.py b/baz/qux.py
--- a/baz/qux.py
+++ b/baz/qux.py
@@ -1,2 +1,2 @@
 x = 1
-y = 2
+z = 3
"""
        
        files = benchmark._extract_files_from_patch(patch)
        assert "foo/bar.py" in files
        assert "baz/qux.py" in files
        assert len(files) == 2

    def test_build_prompt_includes_problem(self):
        """Prompt should include the problem statement and repo info."""
        benchmark = SWEBenchLiteBenchmark(mock=True)
        
        ex = {
            "repo": "test/repo",
            "instance_id": "test__repo-12345",
            "problem_statement": "The foo function returns wrong value.",
            "patch": "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n",
        }
        
        prompt = benchmark._build_prompt(ex)
        assert "test/repo" in prompt
        assert "test__repo-12345" in prompt
        assert "The foo function returns wrong value." in prompt
        assert "foo.py" in prompt


class TestSWEBenchToolSet:
    """Test the SWE-Bench tool implementations."""

    def test_view_file(self, tmp_path):
        """view should read a file with line numbers."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        (tmp_path / "test.py").write_text("line1\nline2\nline3")
        
        result = toolset.view({"path": "test.py"})
        assert result.is_error is False
        assert "line1" in result.output
        assert "1 | line1" in result.output  # Line numbers

    def test_view_nonexistent_file(self, tmp_path):
        """view should error for missing files."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        result = toolset.view({"path": "missing.py"})
        assert result.is_error is True
        assert "does not exist" in result.output

    def test_view_range(self, tmp_path):
        """view_range should show only specified lines."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        (tmp_path / "test.py").write_text("line1\nline2\nline3\nline4\nline5")
        
        result = toolset.view_range({"path": "test.py", "start": 2, "end": 4})
        assert result.is_error is False
        assert "2 | line2" in result.output
        assert "3 | line3" in result.output
        assert "4 | line4" in result.output
        assert "1 | line1" not in result.output
        assert "5 | line5" not in result.output

    def test_str_replace_exact_match(self, tmp_path):
        """str_replace should replace exact string once."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        (tmp_path / "test.py").write_text("def hello():\n    return 'world'")
        
        result = toolset.str_replace({
            "path": "test.py",
            "old_str": "    return 'world'",
            "new_str": "    return 'hello world'",
        })
        assert result.is_error is False
        assert "Successfully replaced" in result.output
        
        content = (tmp_path / "test.py").read_text()
        assert "hello world" in content
        assert "return 'world'" not in content  # Old text gone

    def test_str_replace_no_match(self, tmp_path):
        """str_replace should error if old_str not found."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        (tmp_path / "test.py").write_text("def hello():\n    return 'world'")
        
        result = toolset.str_replace({
            "path": "test.py",
            "old_str": "    return 'missing'",
            "new_str": "    return 'new'",
        })
        assert result.is_error is True
        assert "Could not find" in result.output

    def test_str_replace_multiple_matches(self, tmp_path):
        """str_replace should error if old_str appears multiple times."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        (tmp_path / "test.py").write_text("x = 1\nx = 1\nx = 1")
        
        result = toolset.str_replace({
            "path": "test.py",
            "old_str": "x = 1",
            "new_str": "y = 2",
        })
        assert result.is_error is True
        assert "3 occurrences" in result.output

    def test_create_file(self, tmp_path):
        """create should create a new file."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        result = toolset.create({
            "path": "new_file.py",
            "content": "def new():\n    pass",
        })
        assert result.is_error is False
        assert "Created file" in result.output
        assert (tmp_path / "new_file.py").exists()
        assert "def new():" in (tmp_path / "new_file.py").read_text()

    def test_create_existing_file(self, tmp_path):
        """create should error if file already exists."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        (tmp_path / "existing.py").write_text("x = 1")
        
        result = toolset.create({
            "path": "existing.py",
            "content": "y = 2",
        })
        assert result.is_error is True
        assert "already exists" in result.output

    def test_insert_after_line(self, tmp_path):
        """insert should add text after specified line."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        (tmp_path / "test.py").write_text("line1\nline2\nline3")
        
        result = toolset.insert({
            "path": "test.py",
            "line": 1,
            "new_str": "inserted",
        })
        assert result.is_error is False
        
        content = (tmp_path / "test.py").read_text()
        lines = content.split("\n")
        assert lines[1] == "inserted"
        assert lines[0] == "line1"
        assert lines[2] == "line2"

    def test_bash_command(self, tmp_path):
        """bash should run shell commands in workspace."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        result = toolset.bash({"command": "echo hello", "timeout": 5})
        assert result.is_error is False
        assert "hello" in result.output

    def test_bash_timeout(self, tmp_path):
        """bash should timeout long-running commands."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        result = toolset.bash({"command": "sleep 10", "timeout": 1})
        assert result.is_error is True
        assert "timed out" in result.output

    def test_path_traversal_blocked(self, tmp_path):
        """Path traversal outside workspace should be blocked."""
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(tmp_path)
        result = toolset.view({"path": "../../../etc/passwd"})
        assert result.is_error is True
        assert "escapes workspace" in result.output

    def test_tool_specs(self):
        """get_tool_specs should return all tool definitions."""
        from pathlib import Path
        from tjbench.agents.swe_bench_tools import SWEBenchToolSet
        
        toolset = SWEBenchToolSet(Path("/tmp"))
        specs = toolset.get_tool_specs()
        names = [s["name"] for s in specs]
        
        assert "view" in names
        assert "view_range" in names
        assert "str_replace" in names
        assert "create" in names
        assert "insert" in names
        assert "bash" in names
        assert len(specs) == 6
