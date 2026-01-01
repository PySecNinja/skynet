"""Rich console wrapper for terminal output."""

from datetime import datetime
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import FormattedText
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from claude_clone.config import settings, permission_mode_manager
from claude_clone.core.interrupt import interrupt_controller
from claude_clone.ui.keybindings import KeyBindingManager


# Slash commands with descriptions
SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/quit": "Save and exit",
    "/exit": "Save and exit",
    "/q": "Save and exit",
    "/clear": "Clear conversation (starts new session)",
    "/save": "Save current session",
    "/sessions": "List recent sessions",
    "/resume": "Resume a specific session by ID",
    "/models": "List available models",
    "/model": "Switch to a different model",
    "/todos": "Show current task list",
    "/context": "Show token usage",
    "/plan": "Enter/exit plan mode",
    "/approve": "Approve pending plan",
    "/reject": "Reject pending plan",
}


class SlashCommandCompleter(Completer):
    """Autocomplete for slash commands."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only complete if starting with /
        if not text.startswith("/"):
            return

        # Find matching commands
        for cmd, description in SLASH_COMMANDS.items():
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display_meta=description,
                )


class StreamingPanel:
    """Manages a Rich panel that updates during streaming."""

    def __init__(self, console: Console, title: str = "SkyNet"):
        self.console = console
        self.title = title
        self.content = ""
        self.live: Live | None = None

    def _render(self) -> Panel:
        """Render current content as a panel."""
        # Show cursor during streaming
        display_content = self.content + "▌" if self.content else "▌"
        return Panel(
            Text(display_content),
            title=f"[bold red]{self.title}[/bold red]",
            border_style="red",
            padding=(0, 1),
        )

    def start(self) -> None:
        """Begin streaming output."""
        self.content = ""
        self.console.print()  # Newline before panel
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=10,
            transient=True,  # Remove when done so we can render final markdown
        )
        self.live.start()

    def append(self, chunk: str) -> None:
        """Append content and refresh display."""
        self.content += chunk
        if self.live:
            self.live.update(self._render())

    def finish(self) -> Panel:
        """Complete streaming, return final content for markdown rendering."""
        if self.live:
            self.live.stop()
            self.live = None
        return self.content


class StatusSpinner:
    """Displays a spinner with status message."""

    def __init__(self, console: Console):
        self.console = console
        self.live: Live | None = None
        self.message = ""

    def _render(self) -> Text:
        """Render spinner with message."""
        return Text.assemble(
            ("⠋ ", "cyan"),
            (self.message, "dim"),
        )

    def start(self, message: str) -> None:
        """Start spinner with message."""
        self.message = message
        self.live = Live(
            Spinner("dots", text=Text(message, style="dim")),
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self.live.start()

    def update(self, message: str) -> None:
        """Update spinner message."""
        self.message = message
        if self.live:
            self.live.update(Spinner("dots", text=Text(message, style="dim")))

    def stop(self) -> None:
        """Stop the spinner."""
        if self.live:
            self.live.stop()
            self.live = None


class ChatConsole:
    """Handles all terminal output with Rich formatting."""

    def __init__(self):
        self.console = Console()
        self._streaming_panel: StreamingPanel | None = None
        self._status_spinner: StatusSpinner | None = None
        self._completer = SlashCommandCompleter()
        self._prompt_session: PromptSession | None = None

        # Status line data
        self._status_data: dict[str, Any] = {
            "context_used": 0,
            "context_max": settings.num_ctx,
            "model": settings.model,
            "session_start": datetime.now(),
        }

        # Thinking display toggle
        self._show_thinking = True

        # Key binding manager
        self._key_manager = KeyBindingManager()
        self._setup_key_callbacks()

        # History file
        self._history = FileHistory(str(settings.history_file))

    def _setup_key_callbacks(self) -> None:
        """Set up callbacks for keyboard shortcuts."""
        self._key_manager.register_callback("toggle_thinking", self._on_toggle_thinking)
        self._key_manager.register_callback("cycle_permission", self._on_cycle_permission)
        self._key_manager.register_callback("interrupt", self._on_interrupt)
        self._key_manager.register_callback("edit_previous", self._on_edit_previous)

    def _on_toggle_thinking(self, event) -> None:
        """Handle Tab key - toggle thinking display."""
        self._show_thinking = not self._show_thinking
        # Force status bar refresh
        event.app.invalidate()

    def _on_cycle_permission(self, event) -> None:
        """Handle Shift+Tab - cycle permission modes."""
        permission_mode_manager.cycle()
        # Force status bar refresh
        event.app.invalidate()

    def _on_interrupt(self, event) -> None:
        """Handle Escape - signal interrupt."""
        interrupt_controller.signal_interrupt_sync()
        # Force status bar refresh
        event.app.invalidate()

    def _on_edit_previous(self, event) -> None:
        """Handle double-Escape - load previous prompt for editing."""
        buffer = event.app.current_buffer
        history = list(self._history.load_history_strings())
        if history:
            buffer.text = history[-1]
            buffer.cursor_position = len(buffer.text)

    def _get_status_bar(self) -> FormattedText:
        """Generate the bottom status bar content."""
        # Context usage bar
        used = self._status_data.get("context_used", 0)
        max_ctx = self._status_data.get("context_max", settings.num_ctx)
        percent = (used / max_ctx * 100) if max_ctx else 0

        bar_width = 10
        filled = int(bar_width * percent / 100)
        empty = bar_width - filled

        if percent < 50:
            bar_color = "ansigreen"
        elif percent < 75:
            bar_color = "ansiyellow"
        else:
            bar_color = "ansired"

        # Model name
        model = self._status_data.get("model", settings.model)
        # Truncate model name if too long
        if len(model) > 20:
            model = model[:17] + "..."

        # Permission mode
        mode_name, mode_color = permission_mode_manager.get_display_info()
        mode_ansi = {"green": "ansigreen", "yellow": "ansiyellow", "cyan": "ansicyan"}.get(
            mode_color, "ansiwhite"
        )

        # Session duration
        duration = datetime.now() - self._status_data.get("session_start", datetime.now())
        minutes, seconds = divmod(int(duration.total_seconds()), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            time_str = f"{hours}h {minutes}m"
        else:
            time_str = f"{minutes}m {seconds}s"

        # Thinking indicator
        think_indicator = "◉" if self._show_thinking else "○"

        return FormattedText([
            ("", " "),
            (bar_color, "█" * filled),
            ("ansigray", "░" * empty),
            ("", f" {used:,}/{max_ctx:,}"),
            ("", " │ "),
            ("bold", model),
            ("", " │ "),
            (mode_ansi, f"● {mode_name}"),
            ("", " │ "),
            ("ansigray", f"{think_indicator} Think"),
            ("", " │ "),
            ("ansigray", time_str),
            ("", " "),
        ])

    def _get_prompt_session(self) -> PromptSession:
        """Get or create the prompt session."""
        if self._prompt_session is None:
            self._prompt_session = PromptSession(
                completer=self._completer,
                complete_while_typing=True,
                key_bindings=self._key_manager.get_bindings(),
                bottom_toolbar=self._get_status_bar,
                history=self._history,
                enable_history_search=True,  # Ctrl+R support
            )
        return self._prompt_session

    def update_status_bar(self, **kwargs) -> None:
        """Update status bar data.

        Args:
            context_used: Current token usage
            context_max: Maximum context size
            model: Current model name
        """
        self._status_data.update(kwargs)

    def print_thinking(self, content: str) -> None:
        """Print thinking/reasoning process in italic gray style."""
        if not self._show_thinking:
            return

        self.console.print(
            Panel(
                Text(content, style="italic dim"),
                title="[dim]Thinking...[/dim]",
                border_style="dim",
                padding=(0, 1),
            )
        )

    @property
    def show_thinking(self) -> bool:
        """Whether thinking display is enabled."""
        return self._show_thinking

    # === Status Spinner Methods ===

    def start_status(self, message: str) -> None:
        """Start or update the status spinner."""
        if self._status_spinner is None:
            self._status_spinner = StatusSpinner(self.console)
        self._status_spinner.start(message)

    def update_status(self, message: str) -> None:
        """Update the status message without restarting."""
        if self._status_spinner:
            self._status_spinner.update(message)

    def stop_status(self) -> None:
        """Stop the status spinner."""
        if self._status_spinner:
            self._status_spinner.stop()

    # === Streaming Methods ===

    def start_streaming(self) -> None:
        """Start streaming panel for real-time output."""
        self.stop_status()  # Stop spinner when starting to stream
        self._streaming_panel = StreamingPanel(self.console)
        self._streaming_panel.start()

    def stream_chunk(self, chunk: str) -> None:
        """Append a chunk to the streaming output."""
        if self._streaming_panel:
            self._streaming_panel.append(chunk)

    def finish_streaming(self) -> str:
        """Finish streaming and render final content as markdown.

        Returns the final content.
        """
        if self._streaming_panel:
            content = self._streaming_panel.finish()
            self._streaming_panel = None
            # Render final content as proper markdown
            if content:
                try:
                    md = Markdown(content)
                    self.console.print(
                        Panel(
                            md,
                            title="[bold red]SkyNet[/bold red]",
                            border_style="red",
                            padding=(0, 1),
                        )
                    )
                except Exception:
                    self.console.print(
                        Panel(
                            content,
                            title="[bold red]SkyNet[/bold red]",
                            border_style="red",
                            padding=(0, 1),
                        )
                    )
            return content
        return ""

    def print_welcome(self, model: str, session_id: str | None = None) -> None:
        """Print welcome message."""
        session_info = f"[dim]Session: {session_id}[/dim]\n" if session_id else ""
        self.console.print(
            Panel(
                f"[bold red]SkyNet[/bold red] - Local AI Coding Assistant\n"
                f"[dim]Model: {model}[/dim]\n"
                f"{session_info}\n"
                f"[dim]Type your message and press Enter. Use Ctrl+C to exit.[/dim]",
                border_style="red",
            )
        )

    def print_user_message(self, content: str) -> None:
        """Print a user message."""
        self.console.print()
        self.console.print(
            Panel(
                Text(content, style="white"),
                title="[bold blue]You[/bold blue]",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def print_assistant_message(self, content: str) -> None:
        """Print an assistant message with markdown rendering."""
        self.console.print()
        try:
            md = Markdown(content)
            self.console.print(
                Panel(
                    md,
                    title="[bold red]SkyNet[/bold red]",
                    border_style="red",
                    padding=(0, 1),
                )
            )
        except Exception:
            # Fallback to plain text if markdown fails
            self.console.print(
                Panel(
                    content,
                    title="[bold red]SkyNet[/bold red]",
                    border_style="red",
                    padding=(0, 1),
                )
            )

    def print_streaming_start(self) -> None:
        """Indicate streaming has started."""
        self.console.print()
        self.console.print("[dim red]SkyNet:[/dim red]", end=" ")

    def print_streaming_chunk(self, content: str) -> None:
        """Print a streaming chunk."""
        self.console.print(content, end="", markup=False)

    def print_streaming_end(self) -> None:
        """End streaming output."""
        self.console.print()

    def print_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Print a tool invocation."""
        self.console.print()
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in arguments.items())
        self.console.print(
            f"[dim cyan]> Calling tool:[/dim cyan] [bold]{tool_name}[/bold]({args_str})"
        )

    def print_tool_result(self, output: str, success: bool = True) -> None:
        """Print tool execution result."""
        # Truncate long output
        display_output = output[:2000] + "..." if len(output) > 2000 else output

        if success:
            self.console.print(
                Panel(
                    Syntax(display_output, "text", theme="monokai", word_wrap=True),
                    title="[cyan]Tool Result[/cyan]",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
        else:
            self.console.print(
                Panel(
                    Text(display_output, style="red"),
                    title="[red]Tool Error[/red]",
                    border_style="red",
                    padding=(0, 1),
                )
            )

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[dim]{message}[/dim]")

    def print_todos(self) -> None:
        """Print current todo list if any."""
        from claude_clone.tools.todo import TodoManager

        manager = TodoManager()
        todos = manager.get_todos()

        if not todos:
            return

        self.console.print()
        lines = []
        for todo in todos:
            if todo.status == "completed":
                lines.append(f"  [green][x][/green] [dim]{todo.content}[/dim]")
            elif todo.status == "in_progress":
                lines.append(f"  [cyan][>][/cyan] [bold]{todo.content}[/bold]")
            else:
                lines.append(f"  [dim][ ] {todo.content}[/dim]")

        self.console.print(Panel(
            "\n".join(lines),
            title="[bold]Tasks[/bold]",
            border_style="dim",
            padding=(0, 1),
        ))

    def get_active_todo_message(self) -> str | None:
        """Get the active todo message for status display."""
        from claude_clone.tools.todo import TodoManager

        manager = TodoManager()
        active = manager.get_active()
        if active:
            return active.active_form
        return None

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[yellow]Warning:[/yellow] {message}")

    def print_plan_mode_status(self, active: bool) -> None:
        """Print plan mode status."""
        if active:
            self.console.print(
                Panel(
                    "[bold]Plan Mode Active[/bold]\n\n"
                    "I will explore and create a plan for your approval.\n"
                    "Only read-only operations are allowed until the plan is approved.\n\n"
                    "[dim]Commands: /approve, /reject, /plan (to exit without plan)[/dim]",
                    title="[yellow]Planning[/yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )
        else:
            self.console.print("[dim]Exited plan mode.[/dim]")

    def print_plan(self, plan_markdown: str) -> None:
        """Print a plan."""
        self.console.print()
        self.console.print(
            Panel(
                Markdown(plan_markdown),
                title="[bold cyan]Execution Plan[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
        )

    def print_plan_approved(self) -> None:
        """Print plan approved message."""
        self.console.print("[bold green]Plan approved![/bold green] Beginning execution...")

    def print_plan_rejected(self) -> None:
        """Print plan rejected message."""
        self.console.print("[yellow]Plan rejected.[/yellow] You can describe a different approach.")

    def print_context_usage(self, used: int, max_tokens: int, percent: float) -> None:
        """Print context token usage bar.

        Args:
            used: Tokens currently used.
            max_tokens: Maximum context window size.
            percent: Percentage of context used.
        """
        bar_width = 30
        filled = int(bar_width * percent / 100)
        empty = bar_width - filled

        # Color based on usage level
        if percent < 50:
            color = "green"
        elif percent < 75:
            color = "yellow"
        else:
            color = "red"

        bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"

        self.console.print()
        self.console.print(
            Panel(
                f"{bar}\n\n"
                f"[bold]{used:,}[/bold] / {max_tokens:,} tokens ({percent:.1f}%)",
                title="[bold]Context Usage[/bold]",
                border_style="dim",
                padding=(0, 1),
            )
        )

    def confirm(self, message: str) -> bool:
        """Ask for user confirmation."""
        self.console.print(f"[yellow]{message}[/yellow] ", end="")
        response = input("[y/N]: ").strip().lower()
        return response in ("y", "yes")

    def confirm_directory(self, directory: str, tool_name: str) -> bool:
        """Ask for permission to execute tools in a directory.

        This is similar to how Claude Code asks for directory-level permission
        once, rather than asking for each individual tool call.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{tool_name}[/bold] wants to operate in:\n\n"
                f"  [cyan]{directory}[/cyan]\n\n"
                f"[dim]Approving will allow all tool operations in this directory "
                f"and its subdirectories for this session.[/dim]",
                title="[yellow]Permission Required[/yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        self.console.print("[yellow]Allow operations in this directory?[/yellow] ", end="")
        response = input("[y/N]: ").strip().lower()
        return response in ("y", "yes")

    async def get_input_async(self, input_prompt: str = "> ") -> str | None:
        """Get user input with slash command autocomplete (async version).

        Type / to see available commands. Use Tab to autocomplete.

        Returns None on EOF to signal the REPL should exit.
        """
        try:
            session = self._get_prompt_session()
            return await session.prompt_async(input_prompt)
        except EOFError:
            return None  # Signal EOF to caller
        except KeyboardInterrupt:
            # Let the caller handle Ctrl+C
            raise

    def get_input(self, input_prompt: str = "> ") -> str:
        """Get user input with slash command autocomplete (sync version).

        Note: Use get_input_async() when in an async context.
        """
        try:
            session = self._get_prompt_session()
            return session.prompt(input_prompt)
        except EOFError:
            return ""
        except KeyboardInterrupt:
            raise

    def print_sessions(self, sessions: list) -> None:
        """Print a list of sessions."""
        if not sessions:
            self.console.print("[dim]No saved sessions found.[/dim]")
            return

        self.console.print("\n[bold]Recent Sessions:[/bold]")
        for i, session in enumerate(sessions, 1):
            # Format the date nicely
            try:
                from datetime import datetime
                updated = datetime.fromisoformat(session.updated_at)
                date_str = updated.strftime("%Y-%m-%d %H:%M")
            except (ValueError, AttributeError):
                date_str = session.updated_at[:16] if session.updated_at else "unknown"

            self.console.print(
                f"  [cyan]{i}.[/cyan] [bold]{session.id}[/bold] - {session.title}\n"
                f"      [dim]{date_str} | {session.message_count} messages | {session.model}[/dim]"
            )
        self.console.print()

    def print_session_resumed(self, session_id: str, message_count: int) -> None:
        """Print session resumed message."""
        self.console.print(
            f"[green]Resumed session:[/green] {session_id} ({message_count} messages)"
        )

    def print_session_saved(self, session_id: str) -> None:
        """Print session saved message."""
        self.console.print(f"[dim]Session saved: {session_id}[/dim]")

    def print_models(self, models: list[str], current_model: str) -> None:
        """Print list of available models."""
        if not models:
            self.console.print("[dim]No models found.[/dim]")
            return

        self.console.print("\n[bold]Available Models:[/bold]")
        for i, model in enumerate(models, 1):
            if model == current_model:
                self.console.print(f"  [green]{i}. {model} (active)[/green]")
            else:
                self.console.print(f"  [cyan]{i}.[/cyan] {model}")
        self.console.print("\n[dim]Use /model <name> or /model <number> to switch[/dim]")
