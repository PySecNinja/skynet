"""Tests for CreatePlanTool."""

import pytest

from claude_clone.tools.plan import CreatePlanTool
from claude_clone.tools.base import ToolResult
from claude_clone.core.plan import PlanManager, PlanStatus


class TestCreatePlanToolMetadata:
    """Tests for CreatePlanTool metadata."""

    def test_tool_name(self):
        """Verify tool name is 'create_plan'."""
        tool = CreatePlanTool()
        assert tool.name == "create_plan"

    def test_tool_description_exists(self):
        """Verify description is set and non-empty."""
        tool = CreatePlanTool()
        assert tool.description
        assert len(tool.description) > 0

    def test_required_parameters(self):
        """Verify 'goal' and 'steps' are required."""
        tool = CreatePlanTool()
        assert "goal" in tool.required
        assert "steps" in tool.required

    def test_parameters_schema_has_goal(self):
        """Verify goal parameter is correctly defined."""
        tool = CreatePlanTool()
        assert "goal" in tool.parameters
        assert tool.parameters["goal"]["type"] == "string"

    def test_parameters_schema_has_steps(self):
        """Verify steps parameter is correctly defined."""
        tool = CreatePlanTool()
        assert "steps" in tool.parameters
        assert tool.parameters["steps"]["type"] == "array"

    def test_to_schema_output(self):
        """Test to_schema() produces valid Ollama function schema."""
        tool = CreatePlanTool()
        schema = tool.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "create_plan"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"
        assert "goal" in schema["function"]["parameters"]["required"]
        assert "steps" in schema["function"]["parameters"]["required"]


class TestCreatePlanToolExecute:
    """Tests for CreatePlanTool execution."""

    @pytest.fixture
    def tool(self):
        """Provide a CreatePlanTool instance."""
        return CreatePlanTool()

    async def test_execute_creates_plan(self, tool):
        """Test successful plan creation."""
        result = await tool.execute(
            goal="Build a REST API",
            steps=[
                {"description": "Design endpoints", "files_affected": ["api.py"]},
                {"description": "Implement handlers"},
            ],
        )

        assert result.success is True

        manager = PlanManager()
        assert manager.current_plan is not None
        assert manager.current_plan.goal == "Build a REST API"

    async def test_execute_returns_toolresult_ok(self, tool):
        """Test execute returns ToolResult with success=True."""
        result = await tool.execute(
            goal="Test goal",
            steps=[{"description": "Step 1"}],
        )

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.error is None

    async def test_execute_output_contains_markdown(self, tool):
        """Test output contains plan markdown."""
        result = await tool.execute(
            goal="Test goal",
            steps=[{"description": "First step"}],
        )

        assert "# Plan: Test goal" in result.output
        assert "## Steps" in result.output
        assert "First step" in result.output

    async def test_execute_output_contains_approval_instructions(self, tool):
        """Test output contains /approve and /reject commands."""
        result = await tool.execute(
            goal="Test goal",
            steps=[{"description": "Step 1"}],
        )

        assert "/approve" in result.output
        assert "/reject" in result.output

    async def test_execute_sets_plan_in_manager(self, tool):
        """Test plan is stored in PlanManager.current_plan."""
        await tool.execute(
            goal="Test goal",
            steps=[{"description": "Step 1"}],
        )

        manager = PlanManager()
        assert manager.current_plan is not None
        assert manager.current_plan.goal == "Test goal"

    async def test_plan_status_is_pending_approval(self, tool):
        """Verify created plan has PENDING_APPROVAL status."""
        await tool.execute(
            goal="Test goal",
            steps=[{"description": "Step 1"}],
        )

        manager = PlanManager()
        assert manager.current_plan.status == PlanStatus.PENDING_APPROVAL

    async def test_multiple_steps_preserved(self, tool):
        """Test all steps are preserved in created plan."""
        await tool.execute(
            goal="Multi-step plan",
            steps=[
                {"description": "Step 1"},
                {"description": "Step 2"},
                {"description": "Step 3"},
            ],
        )

        manager = PlanManager()
        assert len(manager.current_plan.steps) == 3
        assert manager.current_plan.steps[0].description == "Step 1"
        assert manager.current_plan.steps[1].description == "Step 2"
        assert manager.current_plan.steps[2].description == "Step 3"

    async def test_files_affected_preserved(self, tool):
        """Test files_affected lists are preserved."""
        await tool.execute(
            goal="Test goal",
            steps=[
                {
                    "description": "Modify files",
                    "files_affected": ["src/main.py", "src/utils.py"],
                }
            ],
        )

        manager = PlanManager()
        step = manager.current_plan.steps[0]
        assert step.files_affected == ["src/main.py", "src/utils.py"]


class TestCreatePlanToolValidation:
    """Tests for CreatePlanTool input validation."""

    @pytest.fixture
    def tool(self):
        """Provide a CreatePlanTool instance."""
        return CreatePlanTool()

    async def test_execute_empty_steps_fails(self, tool):
        """Test empty steps list returns failure."""
        result = await tool.execute(goal="Test", steps=[])

        assert result.success is False
        assert "at least one step" in result.error.lower()

    async def test_execute_step_missing_description_fails(self, tool):
        """Test step without description returns failure."""
        result = await tool.execute(
            goal="Test",
            steps=[{"files_affected": ["file.py"]}],  # Missing description
        )

        assert result.success is False
        assert "missing" in result.error.lower() and "description" in result.error.lower()

    async def test_execute_step_empty_description_fails(self, tool):
        """Test step with empty description returns failure."""
        result = await tool.execute(
            goal="Test",
            steps=[{"description": ""}],
        )

        assert result.success is False
        assert "description" in result.error.lower()
