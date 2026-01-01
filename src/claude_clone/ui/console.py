"""Rich console wrapper for terminal output."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


class ChatConsole:
    """Handles all terminal output with Rich formatting."""

    def __init__(self):
        self.console = Console()

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

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[yellow]Warning:[/yellow] {message}")

    def confirm(self, message: str) -> bool:
        """Ask for user confirmation."""
        self.console.print(f"[yellow]{message}[/yellow] ", end="")
        response = input("[y/N]: ").strip().lower()
        return response in ("y", "yes")

    def get_input(self, prompt: str = "> ") -> str:
        """Get user input."""
        try:
            return input(prompt)
        except EOFError:
            return ""

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
