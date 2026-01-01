"""Tool registry for managing available tools."""

from typing import Any

from claude_clone.tools.base import Tool, ToolResult


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas for LLM API."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool with given arguments."""
        tool = self.get(name)
        if not tool:
            return ToolResult.fail(f"Unknown tool: {name}")

        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return ToolResult.fail(f"Tool execution failed: {str(e)}")

    def register_default_tools(self) -> None:
        """Register all default tools."""
        from claude_clone.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
        from claude_clone.tools.git import (
            GitBranchTool,
            GitCommitTool,
            GitDiffTool,
            GitLogTool,
            GitStatusTool,
        )
        from claude_clone.tools.search import GlobTool, GrepTool
        from claude_clone.tools.shell import BashTool
        from claude_clone.tools.plan import CreatePlanTool
        from claude_clone.tools.todo import TodoWriteTool
        from claude_clone.tools.web import WebFetchTool, WebSearchTool

        # File operations
        self.register(ReadFileTool())
        self.register(WriteFileTool())
        self.register(EditFileTool())

        # Search tools
        self.register(GrepTool())
        self.register(GlobTool())

        # Shell
        self.register(BashTool())

        # Git
        self.register(GitStatusTool())
        self.register(GitDiffTool())
        self.register(GitCommitTool())
        self.register(GitLogTool())
        self.register(GitBranchTool())

        # Web
        self.register(WebSearchTool())
        self.register(WebFetchTool())

        # Task tracking
        self.register(TodoWriteTool())

        # Planning
        self.register(CreatePlanTool())
