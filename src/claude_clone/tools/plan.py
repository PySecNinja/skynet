"""Planning tools for plan mode workflow."""

from typing import Any

from claude_clone.core.plan import PlanManager
from claude_clone.tools.base import Tool, ToolResult


class CreatePlanTool(Tool):
    """Tool for creating an execution plan.

    Used in plan mode to outline steps before implementation.
    The plan must be approved by the user before execution begins.
    """

    name = "create_plan"
    description = """Create a structured execution plan before implementation.

Use this in plan mode to outline the steps needed to accomplish a goal.
The user will review and approve the plan before you begin execution.

Each step should have:
- description: Clear description of what will be done
- files_affected: List of files that will be created or modified (optional)

Guidelines:
- Be specific about what each step accomplishes
- List files that will be affected for transparency
- Break complex tasks into manageable steps
- Consider dependencies between steps"""

    parameters = {
        "goal": {
            "type": "string",
            "description": "The overall goal of the plan - what we're trying to accomplish",
        },
        "steps": {
            "type": "array",
            "description": "List of steps to accomplish the goal",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Clear description of this step",
                    },
                    "files_affected": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files that will be created or modified in this step",
                    },
                },
                "required": ["description"],
            },
        },
    }
    required = ["goal", "steps"]

    async def execute(
        self, goal: str, steps: list[dict[str, Any]], **kwargs: Any
    ) -> ToolResult:
        """Create an execution plan.

        Args:
            goal: The overall goal.
            steps: List of step dictionaries.

        Returns:
            ToolResult with the plan markdown.
        """
        try:
            manager = PlanManager()

            # Validate steps
            if not steps:
                return ToolResult.fail("Plan must have at least one step")

            for i, step in enumerate(steps):
                if not step.get("description"):
                    return ToolResult.fail(f"Step {i + 1} is missing a description")

            # Create the plan
            plan = manager.create_plan(goal, steps)

            output = (
                f"{plan.to_markdown()}\n\n"
                "---\n"
                "Plan created and awaiting approval.\n"
                "User commands:\n"
                "  /approve - Approve and begin execution\n"
                "  /reject  - Reject and discard the plan"
            )

            return ToolResult.ok(output)

        except Exception as e:
            return ToolResult.fail(f"Failed to create plan: {e}")
