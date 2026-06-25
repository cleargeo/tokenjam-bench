"""SWE-Bench tool implementations that operate on a real workspace.

These tools are designed to be bound to a specific task's file workspace
at runtime. They provide the core developer operations needed for
SWE-Bench: reading files, editing files, and running commands.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tjbench.agents.tools import ToolResult


class SWEBenchToolSet:
    """Collection of tools for SWE-Bench agent evaluation.
    
    Each tool operates on a specific workspace directory. The toolset
    is instantiated per-task and bound to the task's workspace.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._files: dict[str, str] = {}  # Cache of file contents

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the workspace."""
        # Prevent directory traversal outside workspace
        resolved = (self.workspace / path).resolve()
        if not str(resolved).startswith(str(self.workspace.resolve())):
            raise ValueError(f"Path {path} escapes workspace")
        return resolved

    def _read_file(self, path: Path) -> str:
        """Read a file, caching the result."""
        str_path = str(path)
        if str_path not in self._files:
            if path.exists():
                self._files[str_path] = path.read_text(encoding="utf-8")
            else:
                self._files[str_path] = ""
        return self._files[str_path]

    def _write_file(self, path: Path, content: str) -> None:
        """Write a file and update cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._files[str(path)] = content

    # --- Tool implementations ---

    def view(self, args: dict[str, Any]) -> ToolResult:
        """View the contents of a file."""
        try:
            path = self._resolve_path(args["path"])
            if not path.exists():
                return ToolResult(
                    output=f"Error: File '{args['path']}' does not exist.",
                    is_error=True,
                )
            content = self._read_file(path)
            # Add line numbers for readability
            lines = content.split("\n")
            numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
            return ToolResult(output=f"File: {args['path']}\n{numbered}")
        except Exception as e:
            return ToolResult(output=f"Error: {e}", is_error=True)

    def view_range(self, args: dict[str, Any]) -> ToolResult:
        """View a specific range of lines in a file."""
        try:
            path = self._resolve_path(args["path"])
            start = args["start"]
            end = args["end"]
            
            if not path.exists():
                return ToolResult(
                    output=f"Error: File '{args['path']}' does not exist.",
                    is_error=True,
                )
            
            content = self._read_file(path)
            lines = content.split("\n")
            
            # Clamp to valid range
            start = max(1, start)
            end = min(len(lines), end)
            
            selected = lines[start - 1:end]
            numbered = "\n".join(f"{i+start:4d} | {line}" for i, line in enumerate(selected))
            return ToolResult(
                output=f"File: {args['path']} (lines {start}-{end})\n{numbered}"
            )
        except Exception as e:
            return ToolResult(output=f"Error: {e}", is_error=True)

    def str_replace(self, args: dict[str, Any]) -> ToolResult:
        """Replace an exact string in a file."""
        try:
            path = self._resolve_path(args["path"])
            old_str = args["old_str"]
            new_str = args["new_str"]
            
            if not path.exists():
                return ToolResult(
                    output=f"Error: File '{args['path']}' does not exist.",
                    is_error=True,
                )
            
            content = self._read_file(path)
            
            if old_str not in content:
                return ToolResult(
                    output=f"Error: Could not find the exact string in {args['path']}. "
                           "Make sure the old_str matches exactly (including whitespace).",
                    is_error=True,
                )
            
            # Count occurrences
            count = content.count(old_str)
            if count > 1:
                return ToolResult(
                    output=f"Error: Found {count} occurrences of the string. "
                           "Please use a more specific old_str that matches exactly once.",
                    is_error=True,
                )
            
            new_content = content.replace(old_str, new_str, 1)
            self._write_file(path, new_content)
            
            return ToolResult(
                output=f"Successfully replaced in {args['path']}. "
                       f"Changed {len(old_str)} chars to {len(new_str)} chars."
            )
        except Exception as e:
            return ToolResult(output=f"Error: {e}", is_error=True)

    def create(self, args: dict[str, Any]) -> ToolResult:
        """Create a new file with the given content."""
        try:
            path = self._resolve_path(args["path"])
            content = args["content"]
            
            if path.exists():
                return ToolResult(
                    output=f"Error: File '{args['path']}' already exists. Use str_replace to modify it.",
                    is_error=True,
                )
            
            self._write_file(path, content)
            return ToolResult(output=f"Created file: {args['path']}")
        except Exception as e:
            return ToolResult(output=f"Error: {e}", is_error=True)

    def insert(self, args: dict[str, Any]) -> ToolResult:
        """Insert text after a specific line."""
        try:
            path = self._resolve_path(args["path"])
            line = args["line"]
            new_str = args["new_str"]
            
            if not path.exists():
                return ToolResult(
                    output=f"Error: File '{args['path']}' does not exist.",
                    is_error=True,
                )
            
            content = self._read_file(path)
            lines = content.split("\n")
            
            if line < 0 or line > len(lines):
                return ToolResult(
                    output=f"Error: Line {line} is out of range (file has {len(lines)} lines).",
                    is_error=True,
                )
            
            lines.insert(line, new_str)
            self._write_file(path, "\n".join(lines))
            
            return ToolResult(output=f"Inserted after line {line} in {args['path']}")
        except Exception as e:
            return ToolResult(output=f"Error: {e}", is_error=True)

    def bash(self, args: dict[str, Any]) -> ToolResult:
        """Run a shell command in the workspace."""
        try:
            command = args["command"]
            timeout = args.get("timeout", 30)
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            output = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            
            return ToolResult(
                output=output,
                is_error=result.returncode != 0,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"Error: Command timed out after {timeout}s.",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(output=f"Error: {e}", is_error=True)

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Return tool specifications for the agent."""
        return [
            {
                "name": "view",
                "description": "View the contents of a file. Shows line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file (relative to workspace)"},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "view_range",
                "description": "View a specific range of lines in a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start": {"type": "integer", "description": "Start line (1-indexed)"},
                        "end": {"type": "integer", "description": "End line (1-indexed)"},
                    },
                    "required": ["path", "start", "end"],
                },
            },
            {
                "name": "str_replace",
                "description": "Replace an exact string in a file. The old_str must match exactly once.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_str": {"type": "string", "description": "Exact text to replace (must match exactly once)"},
                        "new_str": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["path", "old_str", "new_str"],
                },
            },
            {
                "name": "create",
                "description": "Create a new file with the given content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "insert",
                "description": "Insert text after a specific line in a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "line": {"type": "integer", "description": "Line after which to insert (1-indexed)"},
                        "new_str": {"type": "string"},
                    },
                    "required": ["path", "line", "new_str"],
                },
            },
            {
                "name": "bash",
                "description": "Run a shell command in the workspace. Use for running tests, git, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds"},
                    },
                    "required": ["command"],
                },
            },
        ]
