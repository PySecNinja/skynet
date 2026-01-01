"""Plan mode management for planning-first workflow."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class PlanStatus(Enum):
    """Status of a plan."""

    DRAFTING = "drafting"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    REJECTED = "rejected"


@dataclass
class PlanStep:
    """A single step in a plan."""

    description: str
    files_affected: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, skipped

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "files_affected": self.files_affected,
            "status": self.status,
        }


@dataclass
class Plan:
    """An execution plan."""

    goal: str
    steps: list[PlanStep]
    status: PlanStatus = PlanStatus.DRAFTING
    created_at: datetime = field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        """Render plan as markdown."""
        lines = [
            f"# Plan: {self.goal}",
            "",
            f"Status: {self.status.value}",
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Steps",
            "",
        ]

        for i, step in enumerate(self.steps, 1):
            status_icon = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
                "skipped": "[-]",
            }.get(step.status, "[ ]")

            lines.append(f"{i}. {status_icon} {step.description}")

            if step.files_affected:
                for f in step.files_affected:
                    lines.append(f"   - `{f}`")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }


class PlanManager:
    """Manages plan mode state.

    When plan mode is active, the agent should only use read-only tools
    to explore and understand the codebase before creating a plan.
    """

    _instance: "PlanManager | None" = None

    def __new__(cls) -> "PlanManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Initialize instance state."""
        self.plan_mode_active: bool = False
        self.current_plan: Plan | None = None
        self.plan_dir = Path.home() / ".claude-clone" / "plans"
        self.plan_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def start_plan_mode(self) -> None:
        """Enter plan mode."""
        self.plan_mode_active = True
        self.current_plan = None

    def end_plan_mode(self) -> None:
        """Exit plan mode."""
        self.plan_mode_active = False

    def is_active(self) -> bool:
        """Check if plan mode is active."""
        return self.plan_mode_active

    def create_plan(self, goal: str, steps: list[dict[str, Any]]) -> Plan:
        """Create a new plan.

        Args:
            goal: The overall goal of the plan.
            steps: List of step dictionaries with 'description' and optional 'files_affected'.

        Returns:
            The created Plan.
        """
        plan_steps = [
            PlanStep(
                description=s.get("description", ""),
                files_affected=s.get("files_affected", []),
            )
            for s in steps
        ]

        self.current_plan = Plan(
            goal=goal,
            steps=plan_steps,
            status=PlanStatus.PENDING_APPROVAL,
        )

        return self.current_plan

    def approve_plan(self) -> bool:
        """Mark current plan as approved.

        Returns:
            True if plan was approved, False if no plan exists.
        """
        if self.current_plan:
            self.current_plan.status = PlanStatus.APPROVED
            self.plan_mode_active = False  # Exit plan mode on approval
            return True
        return False

    def reject_plan(self) -> bool:
        """Reject and discard current plan.

        Returns:
            True if plan was rejected, False if no plan exists.
        """
        if self.current_plan:
            self.current_plan.status = PlanStatus.REJECTED
            self.current_plan = None
            self.plan_mode_active = False
            return True
        return False

    def get_plan(self) -> Plan | None:
        """Get the current plan."""
        return self.current_plan

    def has_pending_plan(self) -> bool:
        """Check if there's a plan pending approval."""
        return (
            self.current_plan is not None
            and self.current_plan.status == PlanStatus.PENDING_APPROVAL
        )

    def mark_step_in_progress(self, step_index: int) -> bool:
        """Mark a step as in progress.

        Args:
            step_index: Zero-based index of the step.

        Returns:
            True if successful.
        """
        if self.current_plan and 0 <= step_index < len(self.current_plan.steps):
            self.current_plan.steps[step_index].status = "in_progress"
            return True
        return False

    def mark_step_completed(self, step_index: int) -> bool:
        """Mark a step as completed.

        Args:
            step_index: Zero-based index of the step.

        Returns:
            True if successful.
        """
        if self.current_plan and 0 <= step_index < len(self.current_plan.steps):
            self.current_plan.steps[step_index].status = "completed"
            return True
        return False

    def save_plan(self, plan_id: str | None = None) -> Path:
        """Save plan to disk.

        Args:
            plan_id: Optional ID for the plan file. Auto-generated if not provided.

        Returns:
            Path to the saved plan file.

        Raises:
            ValueError: If no active plan exists.
        """
        if not self.current_plan:
            raise ValueError("No active plan to save")

        if not plan_id:
            plan_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        path = self.plan_dir / f"{plan_id}.md"
        path.write_text(self.current_plan.to_markdown())
        return path

    # Read-only tools that are allowed in plan mode
    READONLY_TOOLS = {
        "read_file",
        "grep",
        "glob",
        "git_status",
        "git_diff",
        "git_log",
        "git_branch",
        "web_search",
        "web_fetch",
        "todo_write",  # Allow task tracking in plan mode
        "create_plan",  # The plan creation tool itself
    }

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in plan mode.

        Args:
            tool_name: Name of the tool.

        Returns:
            True if tool is allowed (read-only or plan mode not active).
        """
        if not self.plan_mode_active:
            return True
        return tool_name in self.READONLY_TOOLS
