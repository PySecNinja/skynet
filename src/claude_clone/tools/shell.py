"""Shell command execution tool."""

import asyncio
import os
import re
import shlex
from typing import Any

from claude_clone.tools.base import Tool, ToolResult


# Dangerous commands that should never be executed
BLOCKED_COMMANDS = [
    r"rm\s+-rf\s+/\s*$",
    r"rm\s+-rf\s+/\*",
    r"rm\s+-rf\s+~\s*$",
    r"mkfs\.",
    r"dd\s+if=.*of=/dev/[sh]d",
    r">\s*/dev/sd",
    r"chmod\s+-R\s+777\s+/\s*$",
    r"chown\s+-R.*:\s*/\s*$",
    r"curl.*\|\s*sh",
    r"wget.*\|\s*sh",
    r"curl.*\|\s*bash",
    r"wget.*\|\s*bash",
]

# Commands that require extra caution
DANGEROUS_COMMANDS = [
    r"^rm\s",
    r"^mv\s.*\s/",
    r"^chmod\s",
    r"^chown\s",
    r"git\s+push",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-fd",
    r"npm\s+publish",
    r"pip\s+install(?!\s+-r)",
    r"sudo\s",
]


class BashTool(Tool):
    """Execute shell commands."""

    name = "bash"
    description = (
        "Execute a bash command in the shell. Use for running tests, builds, "
        "git operations, and other shell commands. Returns stdout and stderr."
    )
    parameters = {
        "command": {
            "type": "string",
            "description": "The shell command to execute",
        },
        "working_directory": {
            "type": "string",
            "description": "Directory to run the command in. Default: current directory",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds. Default: 120",
        },
    }
    required = ["command"]

    def _is_blocked(self, command: str) -> bool:
        """Check if command matches blocked patterns."""
        for pattern in BLOCKED_COMMANDS:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        return False

    def _is_dangerous(self, command: str) -> bool:
        """Check if command requires extra caution."""
        for pattern in DANGEROUS_COMMANDS:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        return False

    async def execute(
        self,
        command: str,
        working_directory: str | None = None,
        timeout: int = 120,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a shell command."""
        # Check for blocked commands
        if self._is_blocked(command):
            return ToolResult.fail(
                f"Command blocked for safety: {command}\n"
                "This command pattern is considered too dangerous to execute."
            )

        # Resolve working directory
        cwd = working_directory or os.getcwd()

        try:
            # Create subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env={**os.environ, "TERM": "dumb"},  # Disable terminal colors
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult.fail(
                    f"Command timed out after {timeout} seconds: {command}"
                )

            # Decode output
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            # Build output
            output_parts = []

            if stdout_text:
                output_parts.append(stdout_text)

            if stderr_text:
                if stdout_text:
                    output_parts.append(f"\n[stderr]\n{stderr_text}")
                else:
                    output_parts.append(stderr_text)

            output = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate very long output
            max_output = 50000
            if len(output) > max_output:
                output = output[:max_output] + f"\n\n[Output truncated at {max_output} characters]"

            if process.returncode == 0:
                return ToolResult.ok(
                    output,
                    exit_code=process.returncode,
                    command=command,
                )
            else:
                return ToolResult.ok(
                    f"[Exit code: {process.returncode}]\n\n{output}",
                    exit_code=process.returncode,
                    command=command,
                )

        except FileNotFoundError:
            return ToolResult.fail(f"Working directory not found: {cwd}")
        except PermissionError:
            return ToolResult.fail(f"Permission denied executing command")
        except Exception as e:
            return ToolResult.fail(f"Command execution error: {str(e)}")
