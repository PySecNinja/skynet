"""Tests for PlanManager, Plan, PlanStep, and PlanStatus."""

import pytest
from datetime import datetime

from claude_clone.core.plan import PlanManager, Plan, PlanStep, PlanStatus


class TestPlanStatus:
    """Tests for PlanStatus enum."""

    def test_drafting_status_exists(self):
        """Verify DRAFTING status exists."""
        assert PlanStatus.DRAFTING.value == "drafting"

    def test_pending_approval_status_exists(self):
        """Verify PENDING_APPROVAL status exists."""
        assert PlanStatus.PENDING_APPROVAL.value == "pending_approval"

    def test_approved_status_exists(self):
        """Verify APPROVED status exists."""
        assert PlanStatus.APPROVED.value == "approved"

    def test_executing_status_exists(self):
        """Verify EXECUTING status exists."""
        assert PlanStatus.EXECUTING.value == "executing"

    def test_completed_status_exists(self):
        """Verify COMPLETED status exists."""
        assert PlanStatus.COMPLETED.value == "completed"

    def test_rejected_status_exists(self):
        """Verify REJECTED status exists."""
        assert PlanStatus.REJECTED.value == "rejected"

    def test_all_statuses_count(self):
        """Verify there are exactly 6 statuses."""
        assert len(PlanStatus) == 6


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_create_minimal_step(self):
        """Test creating a step with only description."""
        step = PlanStep(description="Test step")
        assert step.description == "Test step"
        assert step.files_affected == []
        assert step.status == "pending"

    def test_create_full_step(self):
        """Test creating a step with all fields."""
        step = PlanStep(
            description="Full step",
            files_affected=["file1.py", "file2.py"],
            status="in_progress",
        )
        assert step.description == "Full step"
        assert step.files_affected == ["file1.py", "file2.py"]
        assert step.status == "in_progress"

    def test_default_status_is_pending(self):
        """Verify default status is 'pending'."""
        step = PlanStep(description="Test")
        assert step.status == "pending"

    def test_to_dict_minimal(self):
        """Test to_dict() with minimal step."""
        step = PlanStep(description="Test step")
        result = step.to_dict()
        assert result == {
            "description": "Test step",
            "files_affected": [],
            "status": "pending",
        }

    def test_to_dict_full(self):
        """Test to_dict() with full step."""
        step = PlanStep(
            description="Full step",
            files_affected=["file.py"],
            status="completed",
        )
        result = step.to_dict()
        assert result == {
            "description": "Full step",
            "files_affected": ["file.py"],
            "status": "completed",
        }


class TestPlan:
    """Tests for Plan dataclass."""

    def test_create_plan(self):
        """Test creating a plan with goal and steps."""
        steps = [PlanStep(description="Step 1")]
        plan = Plan(goal="Test goal", steps=steps)
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "Step 1"

    def test_default_status_is_drafting(self):
        """Verify default status is DRAFTING."""
        plan = Plan(goal="Test", steps=[])
        assert plan.status == PlanStatus.DRAFTING

    def test_created_at_is_set(self):
        """Verify created_at timestamp is set automatically."""
        before = datetime.now()
        plan = Plan(goal="Test", steps=[])
        after = datetime.now()
        assert before <= plan.created_at <= after

    def test_to_markdown_basic(self):
        """Test basic markdown output."""
        steps = [PlanStep(description="First step")]
        plan = Plan(goal="Test goal", steps=steps)
        md = plan.to_markdown()

        assert "# Plan: Test goal" in md
        assert "Status: drafting" in md
        assert "## Steps" in md
        assert "1. [ ] First step" in md

    def test_to_markdown_with_status_icons(self):
        """Test status icons in markdown."""
        steps = [
            PlanStep(description="Pending", status="pending"),
            PlanStep(description="In progress", status="in_progress"),
            PlanStep(description="Completed", status="completed"),
            PlanStep(description="Skipped", status="skipped"),
        ]
        plan = Plan(goal="Test", steps=steps)
        md = plan.to_markdown()

        assert "[ ] Pending" in md
        assert "[>] In progress" in md
        assert "[x] Completed" in md
        assert "[-] Skipped" in md

    def test_to_markdown_with_files_affected(self):
        """Test files are listed under steps in markdown."""
        steps = [
            PlanStep(description="Create files", files_affected=["src/main.py", "src/utils.py"])
        ]
        plan = Plan(goal="Test", steps=steps)
        md = plan.to_markdown()

        assert "- `src/main.py`" in md
        assert "- `src/utils.py`" in md

    def test_to_dict(self):
        """Test to_dict() produces correct dictionary."""
        steps = [PlanStep(description="Step 1")]
        plan = Plan(goal="Test goal", steps=steps, status=PlanStatus.APPROVED)
        result = plan.to_dict()

        assert result["goal"] == "Test goal"
        assert result["status"] == "approved"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["description"] == "Step 1"
        assert "created_at" in result


class TestPlanManagerSingleton:
    """Tests for PlanManager singleton pattern."""

    def test_singleton_pattern(self):
        """Verify PlanManager returns same instance."""
        manager1 = PlanManager()
        manager2 = PlanManager()
        assert manager1 is manager2

    def test_reset_creates_new_instance(self):
        """Verify reset() allows creating a fresh instance."""
        manager1 = PlanManager()
        manager1.plan_mode_active = True  # Modify state
        PlanManager.reset()
        manager2 = PlanManager()

        assert manager1 is not manager2
        assert manager2.plan_mode_active is False

    def test_initial_state(self):
        """Verify initial state: plan_mode_active=False, current_plan=None."""
        manager = PlanManager()
        assert manager.plan_mode_active is False
        assert manager.current_plan is None


class TestPlanManagerPlanMode:
    """Tests for PlanManager plan mode operations."""

    def test_start_plan_mode(self, plan_manager):
        """Test entering plan mode sets plan_mode_active=True."""
        plan_manager.start_plan_mode()
        assert plan_manager.plan_mode_active is True

    def test_end_plan_mode(self, plan_manager):
        """Test exiting plan mode sets plan_mode_active=False."""
        plan_manager.start_plan_mode()
        plan_manager.end_plan_mode()
        assert plan_manager.plan_mode_active is False

    def test_is_active_returns_correct_state(self, plan_manager):
        """Test is_active() reflects plan_mode_active state."""
        assert plan_manager.is_active() is False
        plan_manager.start_plan_mode()
        assert plan_manager.is_active() is True
        plan_manager.end_plan_mode()
        assert plan_manager.is_active() is False

    def test_start_plan_mode_clears_current_plan(self, plan_manager, sample_steps):
        """Verify starting plan mode clears any existing plan."""
        plan_manager.create_plan("Old plan", sample_steps)
        assert plan_manager.current_plan is not None

        plan_manager.start_plan_mode()
        assert plan_manager.current_plan is None


class TestPlanManagerPlanLifecycle:
    """Tests for PlanManager plan creation, approval, and rejection."""

    def test_create_plan_returns_plan(self, plan_manager, sample_steps):
        """Test create_plan() returns a Plan object."""
        plan = plan_manager.create_plan("Implement feature X", sample_steps)

        assert isinstance(plan, Plan)
        assert plan.goal == "Implement feature X"
        assert len(plan.steps) == len(sample_steps)

    def test_create_plan_sets_pending_approval(self, plan_manager, sample_steps):
        """Test created plan has PENDING_APPROVAL status."""
        plan = plan_manager.create_plan("Test goal", sample_steps)
        assert plan.status == PlanStatus.PENDING_APPROVAL

    def test_create_plan_stores_current_plan(self, plan_manager, sample_steps):
        """Test plan is stored in current_plan."""
        plan = plan_manager.create_plan("Test goal", sample_steps)
        assert plan_manager.current_plan is plan

    def test_create_plan_with_empty_files_affected(self, plan_manager):
        """Test steps without files_affected default to empty list."""
        steps = [{"description": "Step without files"}]
        plan = plan_manager.create_plan("Test", steps)
        assert plan.steps[0].files_affected == []

    def test_approve_plan_success(self, plan_manager, sample_steps):
        """Test approve_plan() returns True and sets APPROVED."""
        plan_manager.create_plan("Test goal", sample_steps)

        result = plan_manager.approve_plan()

        assert result is True
        assert plan_manager.current_plan.status == PlanStatus.APPROVED

    def test_approve_plan_exits_plan_mode(self, plan_manager, sample_steps):
        """Test approving plan sets plan_mode_active=False."""
        plan_manager.start_plan_mode()
        plan_manager.create_plan("Test goal", sample_steps)

        plan_manager.approve_plan()

        assert plan_manager.plan_mode_active is False

    def test_approve_plan_no_plan_returns_false(self, plan_manager):
        """Test approve_plan() returns False when no plan exists."""
        result = plan_manager.approve_plan()
        assert result is False

    def test_reject_plan_success(self, plan_manager, sample_steps):
        """Test reject_plan() returns True and sets REJECTED."""
        plan_manager.create_plan("Test goal", sample_steps)

        # Keep reference before rejection clears it
        plan = plan_manager.current_plan

        result = plan_manager.reject_plan()

        assert result is True
        assert plan.status == PlanStatus.REJECTED

    def test_reject_plan_clears_current_plan(self, plan_manager, sample_steps):
        """Test rejecting plan sets current_plan=None."""
        plan_manager.create_plan("Test goal", sample_steps)
        plan_manager.reject_plan()
        assert plan_manager.current_plan is None

    def test_reject_plan_exits_plan_mode(self, plan_manager, sample_steps):
        """Test rejecting plan sets plan_mode_active=False."""
        plan_manager.start_plan_mode()
        plan_manager.create_plan("Test goal", sample_steps)

        plan_manager.reject_plan()

        assert plan_manager.plan_mode_active is False

    def test_reject_plan_no_plan_returns_false(self, plan_manager):
        """Test reject_plan() returns False when no plan exists."""
        result = plan_manager.reject_plan()
        assert result is False


class TestPlanManagerStepStatus:
    """Tests for PlanManager step status operations."""

    def test_mark_step_in_progress(self, plan_manager, sample_steps):
        """Test marking a step as in_progress."""
        plan_manager.create_plan("Test", sample_steps)

        result = plan_manager.mark_step_in_progress(0)

        assert result is True
        assert plan_manager.current_plan.steps[0].status == "in_progress"

    def test_mark_step_completed(self, plan_manager, sample_steps):
        """Test marking a step as completed."""
        plan_manager.create_plan("Test", sample_steps)

        result = plan_manager.mark_step_completed(1)

        assert result is True
        assert plan_manager.current_plan.steps[1].status == "completed"

    def test_mark_step_invalid_index_returns_false(self, plan_manager, sample_steps):
        """Test invalid step index returns False."""
        plan_manager.create_plan("Test", sample_steps)

        result = plan_manager.mark_step_in_progress(100)

        assert result is False

    def test_mark_step_negative_index_returns_false(self, plan_manager, sample_steps):
        """Test negative index returns False."""
        plan_manager.create_plan("Test", sample_steps)

        result = plan_manager.mark_step_completed(-1)

        assert result is False

    def test_mark_step_no_plan_returns_false(self, plan_manager):
        """Test marking step without a plan returns False."""
        assert plan_manager.mark_step_in_progress(0) is False
        assert plan_manager.mark_step_completed(0) is False


class TestPlanManagerUtilities:
    """Tests for PlanManager utility methods."""

    def test_get_plan_returns_current(self, plan_manager, sample_steps):
        """Test get_plan() returns current_plan."""
        plan = plan_manager.create_plan("Test", sample_steps)
        assert plan_manager.get_plan() is plan

    def test_get_plan_returns_none_initially(self, plan_manager):
        """Test get_plan() returns None when no plan exists."""
        assert plan_manager.get_plan() is None

    def test_has_pending_plan_true(self, plan_manager, sample_steps):
        """Test has_pending_plan() when plan is PENDING_APPROVAL."""
        plan_manager.create_plan("Test", sample_steps)
        assert plan_manager.has_pending_plan() is True

    def test_has_pending_plan_false_when_approved(self, plan_manager, sample_steps):
        """Test has_pending_plan() returns False when plan is APPROVED."""
        plan_manager.create_plan("Test", sample_steps)
        plan_manager.approve_plan()
        assert plan_manager.has_pending_plan() is False

    def test_has_pending_plan_false_when_no_plan(self, plan_manager):
        """Test has_pending_plan() returns False when no plan."""
        assert plan_manager.has_pending_plan() is False


class TestPlanManagerToolRestriction:
    """Tests for PlanManager tool restriction in plan mode."""

    def test_is_tool_allowed_when_not_in_plan_mode(self, plan_manager):
        """All tools should be allowed when plan_mode_active=False."""
        assert plan_manager.is_tool_allowed("write_file") is True
        assert plan_manager.is_tool_allowed("bash") is True
        assert plan_manager.is_tool_allowed("edit_file") is True
        assert plan_manager.is_tool_allowed("any_tool") is True

    def test_is_tool_allowed_readonly_tools_in_plan_mode(self, plan_manager):
        """Test read-only tools are allowed in plan mode."""
        plan_manager.start_plan_mode()

        # All readonly tools should be allowed
        readonly_tools = [
            "read_file",
            "grep",
            "glob",
            "git_status",
            "git_diff",
            "git_log",
            "git_branch",
            "web_search",
            "web_fetch",
            "todo_write",
            "create_plan",
        ]

        for tool in readonly_tools:
            assert plan_manager.is_tool_allowed(tool) is True, f"{tool} should be allowed"

    def test_is_tool_blocked_write_tools_in_plan_mode(self, plan_manager):
        """Test write tools are blocked in plan mode."""
        plan_manager.start_plan_mode()

        blocked_tools = ["write_file", "edit_file", "bash", "git_commit"]

        for tool in blocked_tools:
            assert plan_manager.is_tool_allowed(tool) is False, f"{tool} should be blocked"

    def test_readonly_tools_constant(self, plan_manager):
        """Verify READONLY_TOOLS set contains expected tools."""
        expected = {
            "read_file",
            "grep",
            "glob",
            "git_status",
            "git_diff",
            "git_log",
            "git_branch",
            "web_search",
            "web_fetch",
            "todo_write",
            "create_plan",
        }
        assert PlanManager.READONLY_TOOLS == expected


class TestPlanManagerSave:
    """Tests for PlanManager save operations."""

    def test_save_plan_creates_file(self, plan_manager_with_temp_dir, sample_steps):
        """Test save_plan() creates a markdown file."""
        manager = plan_manager_with_temp_dir
        manager.create_plan("Test goal", sample_steps)

        path = manager.save_plan()

        assert path.exists()
        assert path.suffix == ".md"

    def test_save_plan_with_custom_id(self, plan_manager_with_temp_dir, sample_steps):
        """Test save_plan() uses provided plan_id."""
        manager = plan_manager_with_temp_dir
        manager.create_plan("Test goal", sample_steps)

        path = manager.save_plan(plan_id="custom_plan_id")

        assert path.name == "custom_plan_id.md"

    def test_save_plan_generates_id(self, plan_manager_with_temp_dir, sample_steps):
        """Test save_plan() auto-generates timestamp-based ID."""
        manager = plan_manager_with_temp_dir
        manager.create_plan("Test goal", sample_steps)

        path = manager.save_plan()

        # Should be in format YYYYMMDD_HHMMSS.md
        assert len(path.stem) == 15  # 8 digits + underscore + 6 digits

    def test_save_plan_raises_when_no_plan(self, plan_manager_with_temp_dir):
        """Test save_plan() raises ValueError when no plan exists."""
        manager = plan_manager_with_temp_dir

        with pytest.raises(ValueError, match="No active plan"):
            manager.save_plan()

    def test_save_plan_content_matches_markdown(self, plan_manager_with_temp_dir, sample_steps):
        """Verify saved file content matches to_markdown() output."""
        manager = plan_manager_with_temp_dir
        plan = manager.create_plan("Test goal", sample_steps)

        path = manager.save_plan()
        content = path.read_text()

        assert content == plan.to_markdown()
