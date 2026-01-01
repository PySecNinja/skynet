"""File operation tools: Read, Write, Edit."""

from pathlib import Path
from typing import Any

from claude_clone.tools.base import Tool, ToolResult


class ReadFileTool(Tool):
    """Read file contents with line numbers."""

    name = "read_file"
    description = "Read the contents of a file. Returns the file content with line numbers."
    parameters = {
        "file_path": {
            "type": "string",
            "description": "The path to the file to read (absolute or relative to cwd)",
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (0-indexed). Default: 0",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read. Default: 500",
        },
    }
    required = ["file_path"]

    async def execute(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 500,
        **kwargs: Any,
    ) -> ToolResult:
        """Read a file with optional offset and limit."""
        try:
            path = Path(file_path).expanduser().resolve()

            if not path.exists():
                return ToolResult.fail(f"File not found: {file_path}")

            if not path.is_file():
                return ToolResult.fail(f"Not a file: {file_path}")

            # Read file content
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Try reading as binary and showing info
                size = path.stat().st_size
                return ToolResult.fail(
                    f"Cannot read binary file: {file_path} (size: {size} bytes)"
                )

            lines = content.splitlines()
            total_lines = len(lines)

            # Apply offset and limit
            selected_lines = lines[offset : offset + limit]

            # Format with line numbers
            formatted_lines = []
            for i, line in enumerate(selected_lines, start=offset + 1):
                formatted_lines.append(f"{i:6}\t{line}")

            output = "\n".join(formatted_lines)

            # Add info about truncation
            if offset > 0 or offset + limit < total_lines:
                output = (
                    f"[Showing lines {offset + 1}-{min(offset + limit, total_lines)} "
                    f"of {total_lines}]\n\n{output}"
                )

            return ToolResult.ok(output, total_lines=total_lines, path=str(path))

        except PermissionError:
            return ToolResult.fail(f"Permission denied: {file_path}")
        except Exception as e:
            return ToolResult.fail(f"Error reading file: {str(e)}")


class WriteFileTool(Tool):
    """Write content to a file."""

    name = "write_file"
    description = "Write content to a file, creating it if it doesn't exist. Will overwrite existing content."
    parameters = {
        "file_path": {
            "type": "string",
            "description": "The path to the file to write (absolute or relative to cwd)",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file",
        },
    }
    required = ["file_path", "content"]

    async def execute(
        self,
        file_path: str,
        content: str,
        **kwargs: Any,
    ) -> ToolResult:
        """Write content to a file."""
        try:
            path = Path(file_path).expanduser().resolve()

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists for reporting
            existed = path.exists()

            # Write the file
            path.write_text(content, encoding="utf-8")

            if existed:
                return ToolResult.ok(
                    f"File updated: {file_path} ({len(content)} bytes)",
                    path=str(path),
                    created=False,
                )
            else:
                return ToolResult.ok(
                    f"File created: {file_path} ({len(content)} bytes)",
                    path=str(path),
                    created=True,
                )

        except PermissionError:
            return ToolResult.fail(f"Permission denied: {file_path}")
        except Exception as e:
            return ToolResult.fail(f"Error writing file: {str(e)}")


class EditFileTool(Tool):
    """Edit a file by replacing specific content."""

    name = "edit_file"
    description = (
        "Edit a file by replacing specific content. The old_content must match exactly "
        "(including whitespace). Use this for precise edits rather than rewriting entire files."
    )
    parameters = {
        "file_path": {
            "type": "string",
            "description": "The path to the file to edit",
        },
        "old_content": {
            "type": "string",
            "description": "The exact content to find and replace",
        },
        "new_content": {
            "type": "string",
            "description": "The content to replace it with",
        },
    }
    required = ["file_path", "old_content", "new_content"]

    async def execute(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        **kwargs: Any,
    ) -> ToolResult:
        """Edit a file by replacing old_content with new_content."""
        try:
            path = Path(file_path).expanduser().resolve()

            if not path.exists():
                return ToolResult.fail(f"File not found: {file_path}")

            if not path.is_file():
                return ToolResult.fail(f"Not a file: {file_path}")

            # Read current content
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ToolResult.fail(f"Cannot edit binary file: {file_path}")

            # Check if old_content exists
            if old_content not in content:
                # Try to find similar content for helpful error
                lines = content.splitlines()
                first_line = old_content.splitlines()[0] if old_content else ""

                similar = []
                for i, line in enumerate(lines, 1):
                    if first_line and first_line.strip() in line:
                        similar.append(f"  Line {i}: {line[:80]}")

                hint = ""
                if similar:
                    hint = "\n\nSimilar lines found:\n" + "\n".join(similar[:3])

                return ToolResult.fail(
                    f"Content not found in file. Make sure old_content matches exactly, "
                    f"including whitespace and indentation.{hint}"
                )

            # Count occurrences
            count = content.count(old_content)
            if count > 1:
                return ToolResult.fail(
                    f"Found {count} occurrences of the content. "
                    "Please provide more context to make the match unique."
                )

            # Perform replacement
            new_file_content = content.replace(old_content, new_content, 1)

            # Write back
            path.write_text(new_file_content, encoding="utf-8")

            # Show what changed
            old_lines = len(old_content.splitlines())
            new_lines = len(new_content.splitlines())
            diff_info = f"Replaced {old_lines} line(s) with {new_lines} line(s)"

            return ToolResult.ok(
                f"File edited: {file_path}\n{diff_info}",
                path=str(path),
            )

        except PermissionError:
            return ToolResult.fail(f"Permission denied: {file_path}")
        except Exception as e:
            return ToolResult.fail(f"Error editing file: {str(e)}")
