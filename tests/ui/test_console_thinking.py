"""Tests for ChatConsole thinking toggle functionality."""

import pytest
from unittest.mock import MagicMock, patch, call
from rich.panel import Panel
from rich.text import Text


class TestThinkingInitialState:
    """Tests for initial thinking toggle state."""

    def test_show_thinking_default_true(self, chat_console_with_mocks):
        """Verify _show_thinking is True by default."""
        console = chat_console_with_mocks
        assert console._show_thinking is True

    def test_show_thinking_property_returns_state(self, chat_console_with_mocks):
        """Test show_thinking property returns _show_thinking value."""
        console = chat_console_with_mocks

        console._show_thinking = True
        assert console.show_thinking is True

        console._show_thinking = False
        assert console.show_thinking is False


class TestThinkingToggle:
    """Tests for thinking toggle functionality."""

    def test_on_toggle_thinking_toggles_state(self, chat_console_with_mocks, mock_event):
        """Test _on_toggle_thinking() inverts _show_thinking."""
        console = chat_console_with_mocks

        # Initially True
        assert console._show_thinking is True

        # Toggle to False
        console._on_toggle_thinking(mock_event)
        assert console._show_thinking is False

        # Toggle back to True
        console._on_toggle_thinking(mock_event)
        assert console._show_thinking is True

    def test_toggle_from_true_to_false(self, chat_console_with_mocks, mock_event):
        """Test toggling from True to False."""
        console = chat_console_with_mocks
        console._show_thinking = True

        console._on_toggle_thinking(mock_event)

        assert console._show_thinking is False

    def test_toggle_from_false_to_true(self, chat_console_with_mocks, mock_event):
        """Test toggling from False to True."""
        console = chat_console_with_mocks
        console._show_thinking = False

        console._on_toggle_thinking(mock_event)

        assert console._show_thinking is True

    def test_toggle_calls_invalidate(self, chat_console_with_mocks, mock_event):
        """Test toggle calls event.app.invalidate() for UI refresh."""
        console = chat_console_with_mocks

        console._on_toggle_thinking(mock_event)

        mock_event.app.invalidate.assert_called_once()

    def test_toggle_multiple_times_calls_invalidate_each_time(
        self, chat_console_with_mocks, mock_event
    ):
        """Test each toggle calls invalidate."""
        console = chat_console_with_mocks

        console._on_toggle_thinking(mock_event)
        console._on_toggle_thinking(mock_event)
        console._on_toggle_thinking(mock_event)

        assert mock_event.app.invalidate.call_count == 3


class TestPrintThinking:
    """Tests for print_thinking() method."""

    def test_print_thinking_when_enabled(self, chat_console_with_mocks):
        """Test print_thinking() prints when _show_thinking=True."""
        console = chat_console_with_mocks
        console._show_thinking = True

        console.print_thinking("Test reasoning content")

        console.console.print.assert_called_once()

    def test_print_thinking_when_disabled(self, chat_console_with_mocks):
        """Test print_thinking() does nothing when _show_thinking=False."""
        console = chat_console_with_mocks
        console._show_thinking = False

        console.print_thinking("Test reasoning content")

        console.console.print.assert_not_called()

    def test_print_thinking_calls_with_panel(self, chat_console_with_mocks):
        """Test print_thinking() calls console.print with a Panel."""
        console = chat_console_with_mocks
        console._show_thinking = True

        console.print_thinking("Test content")

        # Get the argument passed to print
        call_args = console.console.print.call_args
        printed_object = call_args[0][0]

        assert isinstance(printed_object, Panel)

    def test_print_thinking_empty_content(self, chat_console_with_mocks):
        """Test print_thinking() with empty content still works."""
        console = chat_console_with_mocks
        console._show_thinking = True

        console.print_thinking("")

        console.console.print.assert_called_once()


class TestThinkingStatusBar:
    """Tests for thinking indicator in status bar."""

    def test_status_bar_shows_filled_indicator_when_enabled(self, chat_console_with_mocks):
        """Test status bar shows filled circle when thinking enabled."""
        console = chat_console_with_mocks
        console._show_thinking = True

        status_bar = console._get_status_bar()

        # Convert FormattedText to string representation for checking
        status_text = "".join(part[1] for part in status_bar)
        assert "◉" in status_text
        assert "Think" in status_text

    def test_status_bar_shows_empty_indicator_when_disabled(self, chat_console_with_mocks):
        """Test status bar shows empty circle when thinking disabled."""
        console = chat_console_with_mocks
        console._show_thinking = False

        status_bar = console._get_status_bar()

        # Convert FormattedText to string representation for checking
        status_text = "".join(part[1] for part in status_bar)
        assert "○" in status_text
        assert "Think" in status_text

    def test_get_status_bar_includes_think_section(self, chat_console_with_mocks):
        """Test _get_status_bar() includes thinking indicator."""
        console = chat_console_with_mocks

        status_bar = console._get_status_bar()

        # Convert to string and check for Think label
        status_text = "".join(part[1] for part in status_bar)
        assert "Think" in status_text


class TestThinkingIntegration:
    """Integration tests for thinking functionality."""

    def test_toggle_affects_print_thinking_behavior(self, chat_console_with_mocks, mock_event):
        """Test that toggling affects print_thinking output."""
        console = chat_console_with_mocks

        # Initially enabled
        console.print_thinking("Content 1")
        assert console.console.print.call_count == 1

        # Toggle off
        console._on_toggle_thinking(mock_event)
        console.print_thinking("Content 2")
        assert console.console.print.call_count == 1  # Still 1, not printed

        # Toggle on
        console._on_toggle_thinking(mock_event)
        console.print_thinking("Content 3")
        assert console.console.print.call_count == 2  # Now 2

    def test_status_bar_updates_with_toggle(self, chat_console_with_mocks, mock_event):
        """Test status bar reflects toggle state changes."""
        console = chat_console_with_mocks

        # Check initial state (enabled)
        status1 = console._get_status_bar()
        status1_text = "".join(part[1] for part in status1)
        assert "◉" in status1_text

        # Toggle and check
        console._on_toggle_thinking(mock_event)
        status2 = console._get_status_bar()
        status2_text = "".join(part[1] for part in status2)
        assert "○" in status2_text

        # Toggle back and check
        console._on_toggle_thinking(mock_event)
        status3 = console._get_status_bar()
        status3_text = "".join(part[1] for part in status3)
        assert "◉" in status3_text
