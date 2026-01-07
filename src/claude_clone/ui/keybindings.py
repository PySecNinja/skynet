"""Keyboard bindings for SkyNet CLI."""

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from typing import Callable, Any


class KeyBindingManager:
    """Manages keyboard shortcuts for the CLI.

    Provides a clean interface for registering callbacks for various
    keyboard shortcuts, mimicking Claude Code's interaction patterns.
    """

    def __init__(self):
        self.bindings = KeyBindings()
        self._callbacks: dict[str, Callable] = {}
        self._escape_count = 0
        self._last_escape_time = 0.0
        self._setup_bindings()

    def _setup_bindings(self) -> None:
        """Set up all keyboard bindings."""
        import time

        # Ctrl+R - Searchable history (handled by prompt_toolkit when enabled)
        # We just need to make sure enable_history_search=True in PromptSession

        # Tab - Toggle thinking display OR complete
        @self.bindings.add(Keys.Tab)
        def handle_tab(event) -> None:
            buffer = event.app.current_buffer
            # If there's text and completions available, do completion
            if buffer.text and buffer.complete_state:
                buffer.complete_next()
            elif buffer.text.startswith("/"):
                # Trigger completion for slash commands
                buffer.start_completion()
            elif "toggle_thinking" in self._callbacks:
                # Otherwise toggle thinking display
                self._callbacks["toggle_thinking"](event)

        # Shift+Tab - Cycle permission modes
        @self.bindings.add(Keys.BackTab)
        def handle_shift_tab(event) -> None:
            if "cycle_permission" in self._callbacks:
                self._callbacks["cycle_permission"](event)

        # Escape - Interrupt or double-escape for edit previous
        @self.bindings.add(Keys.Escape)
        def handle_escape(event) -> None:
            current_time = time.time()
            # Check for double-escape (within 500ms)
            if current_time - self._last_escape_time < 0.5:
                self._escape_count += 1
                if self._escape_count >= 2:
                    # Double escape - edit previous prompt
                    if "edit_previous" in self._callbacks:
                        self._callbacks["edit_previous"](event)
                    self._escape_count = 0
            else:
                self._escape_count = 1
                # Single escape - interrupt
                if "interrupt" in self._callbacks:
                    self._callbacks["interrupt"](event)

            self._last_escape_time = current_time

        # Ctrl+O - Expand last collapsed content
        @self.bindings.add("c-o")
        def handle_ctrl_o(event) -> None:
            if "expand_content" in self._callbacks:
                self._callbacks["expand_content"](event)

    def register_callback(self, action: str, callback: Callable) -> None:
        """Register a callback for a keyboard action.

        Args:
            action: One of 'toggle_thinking', 'cycle_permission',
                   'interrupt', 'edit_previous'
            callback: Function to call when the action is triggered
        """
        self._callbacks[action] = callback

    def get_bindings(self) -> KeyBindings:
        """Get the KeyBindings object for prompt_toolkit."""
        return self.bindings
