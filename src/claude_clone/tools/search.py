"""Search tools: Grep and Glob."""

import asyncio
import fnmatch
import re
from pathlib import Path
from typing import Any

from claude_clone.tools.base import Tool, ToolResult


class GrepTool(Tool):
    """Search file contents using regex patterns."""

    name = "grep"
    description = (
        "Search for a pattern in files using regular expressions. "
        "Returns matching lines with file paths and line numbers."
    )
    parameters = {
        "pattern": {
            "type": "string",
            "description": "Regular expression pattern to search for",
        },
        "path": {
            "type": "string",
            "description": "Directory or file to search in. Default: current directory",
        },
        "glob": {
            "type": "string",
            "description": "Glob pattern to filter files (e.g., '*.py', '*.js'). Default: all files",
        },
        "case_insensitive": {
            "type": "boolean",
            "description": "Ignore case in search. Default: false",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return. Default: 100",
        },
    }
    required = ["pattern"]

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        case_insensitive: bool = False,
        max_results: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Search for a pattern in files."""
        try:
            search_path = Path(path).expanduser().resolve()

            if not search_path.exists():
                return ToolResult.fail(f"Path not found: {path}")

            # Compile regex
            flags = re.IGNORECASE if case_insensitive else 0
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return ToolResult.fail(f"Invalid regex pattern: {e}")

            results = []
            files_searched = 0
            files_matched = 0

            # Get files to search
            if search_path.is_file():
                files = [search_path]
            else:
                if glob:
                    files = list(search_path.rglob(glob))
                else:
                    files = [f for f in search_path.rglob("*") if f.is_file()]

            # Filter out common non-text directories
            skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}
            files = [
                f for f in files
                if not any(skip in f.parts for skip in skip_dirs)
            ]

            for file_path in files:
                if len(results) >= max_results:
                    break

                if not file_path.is_file():
                    continue

                files_searched += 1

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.splitlines()

                    file_had_match = False
                    for line_num, line in enumerate(lines, 1):
                        if len(results) >= max_results:
                            break
                        if regex.search(line):
                            if not file_had_match:
                                files_matched += 1
                                file_had_match = True
                            rel_path = file_path.relative_to(Path.cwd()) if file_path.is_relative_to(Path.cwd()) else file_path
                            results.append(f"{rel_path}:{line_num}: {line.strip()}")

                except (UnicodeDecodeError, PermissionError):
                    continue

            if not results:
                return ToolResult.ok(
                    f"No matches found for pattern '{pattern}' in {files_searched} files.",
                    files_searched=files_searched,
                )

            output = "\n".join(results)
            if len(results) >= max_results:
                output += f"\n\n[Results truncated at {max_results}. Use a more specific pattern or glob filter.]"

            return ToolResult.ok(
                output,
                matches=len(results),
                files_matched=files_matched,
                files_searched=files_searched,
            )

        except Exception as e:
            return ToolResult.fail(f"Search error: {str(e)}")


class GlobTool(Tool):
    """Find files matching glob patterns."""

    name = "glob"
    description = (
        "Find files matching a glob pattern (e.g., '**/*.py', 'src/**/*.ts'). "
        "Returns list of matching file paths."
    )
    parameters = {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match (e.g., '**/*.py', 'src/*.js')",
        },
        "path": {
            "type": "string",
            "description": "Base directory to search from. Default: current directory",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results. Default: 100",
        },
    }
    required = ["pattern"]

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Find files matching a glob pattern."""
        try:
            base_path = Path(path).expanduser().resolve()

            if not base_path.exists():
                return ToolResult.fail(f"Path not found: {path}")

            if not base_path.is_dir():
                return ToolResult.fail(f"Not a directory: {path}")

            # Find matching files
            matches = list(base_path.glob(pattern))

            # Filter out common skip directories
            skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}
            matches = [
                m for m in matches
                if not any(skip in m.parts for skip in skip_dirs)
            ]

            # Sort by modification time (newest first)
            matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            # Apply limit
            truncated = len(matches) > max_results
            matches = matches[:max_results]

            if not matches:
                return ToolResult.ok(
                    f"No files found matching pattern '{pattern}'",
                    count=0,
                )

            # Format output with relative paths
            lines = []
            for match in matches:
                try:
                    rel_path = match.relative_to(Path.cwd()) if match.is_relative_to(Path.cwd()) else match
                    file_type = "d" if match.is_dir() else "f"
                    lines.append(f"[{file_type}] {rel_path}")
                except Exception:
                    lines.append(f"[?] {match}")

            output = "\n".join(lines)
            if truncated:
                output += f"\n\n[Results truncated. {len(matches)} of many matches shown.]"

            return ToolResult.ok(
                output,
                count=len(matches),
                truncated=truncated,
            )

        except Exception as e:
            return ToolResult.fail(f"Glob error: {str(e)}")
