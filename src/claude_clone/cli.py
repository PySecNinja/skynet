"""CLI entry point and REPL loop."""

import asyncio
import sys
from pathlib import Path

import click

from claude_clone.config import Settings
from claude_clone.core.agent import Agent
from claude_clone.core.session import SessionManager
from claude_clone.llm.ollama_provider import Message, OllamaProvider
from claude_clone.tools.registry import ToolRegistry
from claude_clone.ui.console import ChatConsole


def get_system_prompt(cwd: Path) -> str:
    """Generate the system prompt for SkyNet."""
    return f"""You are SkyNet, a powerful AI coding assistant running locally. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search through code using grep and glob patterns
- Perform git operations

Current working directory: {cwd}

When using tools:
- Always use absolute paths or paths relative to the current working directory
- Be careful with destructive operations - the user will be asked to confirm
- Explain what you're doing before executing commands
- If a task requires multiple steps, break it down and execute step by step

Be concise but thorough. Focus on solving the user's problem effectively."""


async def run_repl(
    settings: Settings,
    console: ChatConsole,
    resume_session: str | None = None,
) -> None:
    """Run the main REPL loop."""
    cwd = Path.cwd()

    # Initialize session manager
    session_manager = SessionManager(settings.session_dir)
    current_session_id: str | None = None

    # Cache for available models (populated by /models command)
    available_models: list[str] = []

    # Initialize components
    provider = OllamaProvider(
        model=settings.model,
        host=settings.ollama_host,
        settings=settings,
    )

    # Check connection
    if not await provider.check_connection():
        console.print_error(
            f"Cannot connect to Ollama at {settings.ollama_host}. "
            "Make sure Ollama is running."
        )
        sys.exit(1)

    # Initialize tool registry and agent
    registry = ToolRegistry()
    registry.register_default_tools()

    agent = Agent(
        provider=provider,
        registry=registry,
        console=console,
        system_prompt=get_system_prompt(cwd),
        settings=settings,
    )

    # Resume session if requested
    if resume_session:
        if resume_session == "last":
            resume_session = session_manager.get_last_session_id()

        if resume_session:
            result = session_manager.load_session(resume_session)
            if result:
                messages, saved_model = result
                agent.messages = messages
                current_session_id = resume_session

                # Switch model if different
                if saved_model != settings.model:
                    agent.switch_model(saved_model)
                    settings.model = saved_model

                console.print_session_resumed(resume_session, len(messages))
            else:
                console.print_warning(f"Session not found: {resume_session}")

    console.print_welcome(settings.model, current_session_id)

    # Main loop
    while True:
        try:
            user_input = console.get_input("\n> ")

            if not user_input.strip():
                continue

            # Handle special commands
            cmd = user_input.strip().lower()

            if cmd in ("/quit", "/exit", "/q"):
                # Auto-save on exit if we have messages
                if len(agent.messages) > 1:  # More than just system prompt
                    session_id = session_manager.save_session(
                        agent.messages,
                        settings.model,
                        current_session_id,
                    )
                    console.print_session_saved(session_id)
                console.print_info("Goodbye!")
                break

            if cmd == "/help":
                console.print_info(
                    "Commands:\n"
                    "  /quit, /exit, /q - Save and exit\n"
                    "  /clear - Clear conversation (starts new session)\n"
                    "  /save - Save current session\n"
                    "  /sessions - List recent sessions\n"
                    "  /resume <id> - Resume a specific session\n"
                    "  /models - List available models\n"
                    "  /model <name|number> - Switch model\n"
                    "  /help - Show this help"
                )
                continue

            if cmd == "/clear":
                # Save current session before clearing
                if len(agent.messages) > 1:
                    session_manager.save_session(
                        agent.messages,
                        settings.model,
                        current_session_id,
                    )
                agent.clear_history()
                current_session_id = None
                console.print_info("Conversation cleared. Starting new session.")
                continue

            if cmd == "/save":
                session_id = session_manager.save_session(
                    agent.messages,
                    settings.model,
                    current_session_id,
                )
                current_session_id = session_id
                console.print_session_saved(session_id)
                continue

            if cmd == "/sessions":
                sessions = session_manager.list_sessions()
                console.print_sessions(sessions)
                continue

            if cmd.startswith("/resume "):
                session_id = user_input.strip()[8:].strip()
                result = session_manager.load_session(session_id)
                if result:
                    messages, saved_model = result
                    agent.messages = messages
                    current_session_id = session_id
                    if saved_model != settings.model:
                        agent.switch_model(saved_model)
                        settings.model = saved_model
                    console.print_session_resumed(session_id, len(messages))
                else:
                    console.print_error(f"Session not found: {session_id}")
                continue

            if cmd == "/models":
                models = await provider.list_models()
                available_models.clear()
                available_models.extend(models)
                console.print_models(models, settings.model)
                continue

            if cmd.startswith("/model "):
                model_arg = user_input.strip()[7:].strip()

                # Check if it's a number (index into available_models)
                if model_arg.isdigit():
                    idx = int(model_arg) - 1
                    if 0 <= idx < len(available_models):
                        new_model = available_models[idx]
                    else:
                        console.print_error(f"Invalid model number. Use /models to see available models.")
                        continue
                else:
                    new_model = model_arg

                agent.switch_model(new_model)
                settings.model = new_model
                console.print_info(f"Switched to model: {new_model}")
                continue

            # Process the message
            console.print_user_message(user_input)
            await agent.process_message(user_input)

            # Auto-save after each message
            session_id = session_manager.save_session(
                agent.messages,
                settings.model,
                current_session_id,
            )
            current_session_id = session_id

        except KeyboardInterrupt:
            # Auto-save on interrupt
            if len(agent.messages) > 1:
                session_id = session_manager.save_session(
                    agent.messages,
                    settings.model,
                    current_session_id,
                )
                console.print_session_saved(session_id)
            console.print_info("\nGoodbye!")
            break
        except Exception as e:
            console.print_error(str(e))


@click.command()
@click.option(
    "--model", "-m",
    default=None,
    help="Ollama model to use (default: qwen2.5-coder:32b)",
)
@click.option(
    "--host",
    default=None,
    help="Ollama host URL (default: http://localhost:11434)",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    help="Skip confirmation prompts for writes and commands",
)
@click.option(
    "--resume", "-r",
    is_flag=True,
    help="Resume the last session",
)
@click.option(
    "--session", "-s",
    default=None,
    help="Resume a specific session by ID",
)
@click.argument("prompt", required=False)
def main(
    model: str | None,
    host: str | None,
    no_confirm: bool,
    resume: bool,
    session: str | None,
    prompt: str | None,
) -> None:
    """SkyNet - A local AI coding assistant powered by Ollama."""
    # Load settings with CLI overrides
    settings = Settings()

    if model:
        settings.model = model
    if host:
        settings.ollama_host = host
    if no_confirm:
        settings.confirm_writes = False
        settings.confirm_commands = False

    console = ChatConsole()

    if prompt:
        # Single prompt mode - not implemented yet
        console.print_error("Single prompt mode not yet implemented. Use interactive mode.")
        sys.exit(1)

    # Determine which session to resume
    resume_session = None
    if session:
        resume_session = session
    elif resume:
        resume_session = "last"

    # Run the REPL
    try:
        asyncio.run(run_repl(settings, console, resume_session))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
