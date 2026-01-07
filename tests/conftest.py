"""Shared pytest fixtures for Skynet tests."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons before and after each test to ensure isolation."""
    from claude_clone.core.plan import PlanManager
    from claude_clone.tools.todo import TodoManager

    PlanManager.reset()
    TodoManager.reset()
    yield
    PlanManager.reset()
    TodoManager.reset()


@pytest.fixture
def plan_manager():
    """Provide a fresh PlanManager instance."""
    from claude_clone.core.plan import PlanManager

    return PlanManager()


@pytest.fixture
def sample_steps():
    """Provide sample plan steps for testing."""
    return [
        {"description": "Create test file", "files_affected": ["tests/test_example.py"]},
        {"description": "Write unit tests", "files_affected": ["tests/test_example.py"]},
        {"description": "Run tests and verify", "files_affected": []},
    ]


@pytest.fixture
def plan_manager_with_temp_dir(tmp_path, plan_manager):
    """PlanManager with temporary plan directory."""
    plan_manager.plan_dir = tmp_path / "plans"
    plan_manager.plan_dir.mkdir()
    return plan_manager


@pytest.fixture
def mock_console():
    """Provide a mocked Rich console."""
    return MagicMock()


@pytest.fixture
def mock_event():
    """Provide a mocked prompt_toolkit event."""
    event = MagicMock()
    event.app = MagicMock()
    return event


@pytest.fixture
def chat_console_with_mocks():
    """Create ChatConsole with mocked dependencies."""
    with patch("claude_clone.ui.console.Console") as mock_console_cls:
        with patch("claude_clone.ui.console.FileHistory"):
            from claude_clone.ui.console import ChatConsole

            console = ChatConsole()
            console.console = mock_console_cls.return_value
            return console
