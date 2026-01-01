"""Git operation tools."""

import asyncio
import os
from typing import Any

from claude_clone.tools.base import Tool, ToolResult


async def run_git_command(args: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Run a git command and return (exit_code, stdout, stderr)."""
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd or os.getcwd(),
    )
    stdout, stderr = await process.communicate()
    return (
        process.returncode or 0,
        stdout.decode("utf-8", errors="replace").strip(),
        stderr.decode("utf-8", errors="replace").strip(),
    )


class GitStatusTool(Tool):
    """Show git repository status."""

    name = "git_status"
    description = "Show the working tree status. Shows staged, unstaged, and untracked files."
    parameters = {
        "path": {
            "type": "string",
            "description": "Path to the git repository. Default: current directory",
        },
    }
    required = []

    async def execute(self, path: str | None = None, **kwargs: Any) -> ToolResult:
        """Get git status."""
        try:
            exit_code, stdout, stderr = await run_git_command(
                ["status", "--porcelain=v1", "-b"],
                cwd=path,
            )

            if exit_code != 0:
                if "not a git repository" in stderr.lower():
                    return ToolResult.fail("Not a git repository")
                return ToolResult.fail(f"Git error: {stderr}")

            if not stdout:
                return ToolResult.ok("Working tree clean, nothing to commit")

            # Also get a more readable status
            _, readable, _ = await run_git_command(["status", "-s", "-b"], cwd=path)

            return ToolResult.ok(readable or stdout)

        except FileNotFoundError:
            return ToolResult.fail("Git is not installed")
        except Exception as e:
            return ToolResult.fail(f"Error: {str(e)}")


class GitDiffTool(Tool):
    """Show changes between commits, working tree, etc."""

    name = "git_diff"
    description = (
        "Show changes between commits, commit and working tree, staged and unstaged changes. "
        "Use --staged to see staged changes."
    )
    parameters = {
        "target": {
            "type": "string",
            "description": "What to diff (e.g., 'HEAD', '--staged', 'main..feature', a file path)",
        },
        "path": {
            "type": "string",
            "description": "Repository path. Default: current directory",
        },
    }
    required = []

    async def execute(
        self,
        target: str | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Get git diff."""
        try:
            args = ["diff", "--stat"]
            if target:
                args.extend(target.split())

            exit_code, stdout, stderr = await run_git_command(args, cwd=path)

            if exit_code != 0:
                return ToolResult.fail(f"Git error: {stderr}")

            if not stdout:
                return ToolResult.ok("No changes")

            # Get full diff (limited)
            full_args = ["diff"]
            if target:
                full_args.extend(target.split())

            _, full_diff, _ = await run_git_command(full_args, cwd=path)

            # Truncate if too long
            max_len = 10000
            if len(full_diff) > max_len:
                full_diff = full_diff[:max_len] + "\n\n[Diff truncated...]"

            return ToolResult.ok(f"{stdout}\n\n{full_diff}")

        except FileNotFoundError:
            return ToolResult.fail("Git is not installed")
        except Exception as e:
            return ToolResult.fail(f"Error: {str(e)}")


class GitCommitTool(Tool):
    """Create a new commit."""

    name = "git_commit"
    description = "Create a new commit with staged changes. Stage files first with git add."
    parameters = {
        "message": {
            "type": "string",
            "description": "The commit message",
        },
        "path": {
            "type": "string",
            "description": "Repository path. Default: current directory",
        },
    }
    required = ["message"]

    async def execute(
        self,
        message: str,
        path: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Create a git commit."""
        try:
            # Check if there are staged changes
            exit_code, stdout, stderr = await run_git_command(
                ["diff", "--cached", "--quiet"],
                cwd=path,
            )

            if exit_code == 0:
                return ToolResult.fail(
                    "No staged changes to commit. Use 'git add' to stage files first."
                )

            # Create the commit
            exit_code, stdout, stderr = await run_git_command(
                ["commit", "-m", message],
                cwd=path,
            )

            if exit_code != 0:
                return ToolResult.fail(f"Commit failed: {stderr or stdout}")

            return ToolResult.ok(f"Commit created:\n{stdout}")

        except FileNotFoundError:
            return ToolResult.fail("Git is not installed")
        except Exception as e:
            return ToolResult.fail(f"Error: {str(e)}")


class GitLogTool(Tool):
    """Show commit history."""

    name = "git_log"
    description = "Show recent commit history."
    parameters = {
        "count": {
            "type": "integer",
            "description": "Number of commits to show. Default: 10",
        },
        "oneline": {
            "type": "boolean",
            "description": "Show condensed one-line format. Default: true",
        },
        "path": {
            "type": "string",
            "description": "Repository path. Default: current directory",
        },
    }
    required = []

    async def execute(
        self,
        count: int = 10,
        oneline: bool = True,
        path: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Get git log."""
        try:
            args = ["log", f"-{count}"]
            if oneline:
                args.append("--oneline")
            else:
                args.extend(["--format=%h %s (%an, %ar)"])

            exit_code, stdout, stderr = await run_git_command(args, cwd=path)

            if exit_code != 0:
                if "does not have any commits" in stderr:
                    return ToolResult.ok("No commits yet")
                return ToolResult.fail(f"Git error: {stderr}")

            return ToolResult.ok(stdout or "No commits")

        except FileNotFoundError:
            return ToolResult.fail("Git is not installed")
        except Exception as e:
            return ToolResult.fail(f"Error: {str(e)}")


class GitBranchTool(Tool):
    """List, create, or switch branches."""

    name = "git_branch"
    description = "List branches, create a new branch, or switch to a branch."
    parameters = {
        "action": {
            "type": "string",
            "description": "Action: 'list' (default), 'create', 'switch', 'delete'",
        },
        "name": {
            "type": "string",
            "description": "Branch name for create/switch/delete actions",
        },
        "path": {
            "type": "string",
            "description": "Repository path. Default: current directory",
        },
    }
    required = []

    async def execute(
        self,
        action: str = "list",
        name: str | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Manage git branches."""
        try:
            if action == "list":
                exit_code, stdout, stderr = await run_git_command(
                    ["branch", "-a", "-v"],
                    cwd=path,
                )
                if exit_code != 0:
                    return ToolResult.fail(f"Git error: {stderr}")
                return ToolResult.ok(stdout or "No branches")

            elif action == "create":
                if not name:
                    return ToolResult.fail("Branch name required for create action")
                exit_code, stdout, stderr = await run_git_command(
                    ["branch", name],
                    cwd=path,
                )
                if exit_code != 0:
                    return ToolResult.fail(f"Failed to create branch: {stderr}")
                return ToolResult.ok(f"Branch '{name}' created")

            elif action == "switch":
                if not name:
                    return ToolResult.fail("Branch name required for switch action")
                exit_code, stdout, stderr = await run_git_command(
                    ["checkout", name],
                    cwd=path,
                )
                if exit_code != 0:
                    return ToolResult.fail(f"Failed to switch branch: {stderr}")
                return ToolResult.ok(f"Switched to branch '{name}'")

            elif action == "delete":
                if not name:
                    return ToolResult.fail("Branch name required for delete action")
                exit_code, stdout, stderr = await run_git_command(
                    ["branch", "-d", name],
                    cwd=path,
                )
                if exit_code != 0:
                    return ToolResult.fail(f"Failed to delete branch: {stderr}")
                return ToolResult.ok(f"Branch '{name}' deleted")

            else:
                return ToolResult.fail(f"Unknown action: {action}")

        except FileNotFoundError:
            return ToolResult.fail("Git is not installed")
        except Exception as e:
            return ToolResult.fail(f"Error: {str(e)}")
