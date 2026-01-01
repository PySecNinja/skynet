"""Task tracking tool for managing work items."""

from dataclasses import dataclass, field
from typing import Any, Literal

from claude_clone.tools.base import Tool, ToolResult


@dataclass
class TodoItem:
    """A single todo item."""

    content: str  # Imperative form: "Run tests"
    status: Literal["pending", "in_progress", "completed"]
    active_form: str  # Present continuous: "Running tests"


class TodoManager:
    """Singleton manager for todo items across the session.

    This allows the LLM to track progress on multi-step tasks
    and provides visibility to the user.
    """

    _instance: "TodoManager | None" = None

    def __new__(cls) -> "TodoManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.todos: list[TodoItem] = []
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def update(self, todos: list[TodoItem]) -> None:
        """Replace the entire todo list."""
        self.todos = todos

    def get_todos(self) -> list[TodoItem]:
        """Get all todos."""
        return self.todos

    def get_active(self) -> TodoItem | None:
        """Get the currently in-progress item (should only be one)."""
        for todo in self.todos:
            if todo.status == "in_progress":
                return todo
        return None

    def get_pending_count(self) -> int:
        """Get count of pending items."""
        return sum(1 for t in self.todos if t.status == "pending")

    def get_completed_count(self) -> int:
        """Get count of completed items."""
        return sum(1 for t in self.todos if t.status == "completed")

    def format_display(self) -> str:
        """Format todos for display."""
        if not self.todos:
            return ""

        lines = []
        for todo in self.todos:
            icon = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[todo.status]
            lines.append(f"{icon} {todo.content}")
        return "\n".join(lines)


class TodoWriteTool(Tool):
    """Tool for tracking tasks and progress.

    The LLM uses this to create and update a task list,
    providing visibility into multi-step work.
    """

    name = "todo_write"
    description = """Update the task list for the current work. Use this to:
- Track multi-step tasks (3+ steps)
- Show progress to the user
- Organize complex work

Each todo has:
- content: What to do (imperative form, e.g., "Run tests")
- status: pending, in_progress, or completed
- active_form: Present continuous form shown during execution (e.g., "Running tests")

Guidelines:
- Only have ONE task as in_progress at a time
- Mark tasks completed immediately when done
- Use for complex tasks, not simple one-step actions"""

    parameters = {
        "todos": {
            "type": "array",
            "description": "The complete updated todo list",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Task description in imperative form",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "Current status of the task",
                    },
                    "active_form": {
                        "type": "string",
                        "description": "Task in present continuous form for status display",
                    },
                },
                "required": ["content", "status", "active_form"],
            },
        },
    }
    required = ["todos"]

    async def execute(self, todos: list[dict[str, Any]], **kwargs: Any) -> ToolResult:
        """Update the todo list."""
        try:
            manager = TodoManager()
            items = [
                TodoItem(
                    content=t["content"],
                    status=t["status"],
                    active_form=t["active_form"],
                )
                for t in todos
            ]
            manager.update(items)

            # Format output for conversation history
            output = manager.format_display()

            # Add summary
            pending = manager.get_pending_count()
            completed = manager.get_completed_count()
            active = manager.get_active()

            summary = f"\n\n{completed} completed, {pending} pending"
            if active:
                summary += f" | Currently: {active.active_form}"

            return ToolResult.ok(output + summary)

        except KeyError as e:
            return ToolResult.fail(f"Missing required field: {e}")
        except Exception as e:
            return ToolResult.fail(f"Failed to update todos: {e}")
