"""Main agent orchestration loop."""

import os
from pathlib import Path
from typing import Any

from claude_clone.config import Settings, PermissionMode, permission_mode_manager
from claude_clone.core.context import ContextManager
from claude_clone.core.interrupt import interrupt_controller, InterruptType
from claude_clone.core.plan import PlanManager
from claude_clone.llm.ollama_provider import (
    ChatChunk,
    Message,
    OllamaProvider,
    ToolCall,
    extract_json_tool_call,
)
from claude_clone.tools.registry import ToolRegistry
from claude_clone.ui.console import ChatConsole


class Agent:
    """Main agent that orchestrates conversation and tool execution."""

    def __init__(
        self,
        provider: OllamaProvider,
        registry: ToolRegistry,
        console: ChatConsole,
        system_prompt: str,
        settings: Settings,
    ):
        self.provider = provider
        self.registry = registry
        self.console = console
        self.system_prompt = system_prompt
        self.settings = settings
        self.messages: list[Message] = []

        # Track directories that user has approved for tool execution
        self.approved_directories: set[str] = set()

        # Context management for token limits
        self.context_manager = ContextManager(
            provider=provider,
            max_tokens=settings.num_ctx,
        )

        # Add system prompt
        self.messages.append(Message(role="system", content=system_prompt))

    def clear_history(self) -> None:
        """Clear conversation history, keeping system prompt."""
        self.messages = [Message(role="system", content=self.system_prompt)]

    def switch_model(self, model: str) -> None:
        """Switch to a different model."""
        self.provider = OllamaProvider(
            model=model,
            host=self.provider.host,
            settings=self.settings,
        )

    def get_context_usage(self):
        """Get current context token usage."""
        return self.context_manager.get_usage(self.messages)

    async def process_message(self, user_input: str) -> None:
        """Process a user message and generate response with tool execution."""
        # Clear any pending interrupt
        await interrupt_controller.clear()

        # Check if summarization is needed before adding new message
        if self.context_manager.should_summarize(self.messages):
            self.console.print_info("Context limit approaching, summarizing conversation...")
            self.messages = await self.context_manager.summarize_conversation(self.messages)

        # Add user message
        self.messages.append(Message(role="user", content=user_input))

        # Get tool schemas
        tools = self.registry.get_schemas()

        # Agentic loop - keep going until we get a final response
        max_iterations = 10
        iteration = 0
        interrupted = False

        while iteration < max_iterations and not interrupted:
            iteration += 1

            # Get response from LLM with streaming
            response_content = ""
            tool_calls: list[ToolCall] = []
            streaming_started = False
            looks_like_tool_call = False
            repetition_detected = False

            # Show spinner while waiting for first response
            self.console.start_status("Thinking...")

            async for chunk in self.provider.chat(
                messages=self.messages,
                tools=tools,
                stream=True,  # Enable streaming for real-time output
            ):
                # Check for interrupt during streaming
                if await interrupt_controller.check_interrupted() != InterruptType.NONE:
                    interrupted = True
                    self.console.stop_status()
                    if streaming_started:
                        self.console._streaming_panel.finish()
                        self.console._streaming_panel = None
                    self.console.print_warning("Interrupted by user.")
                    break

                if chunk.content:
                    response_content += chunk.content

                    # Check if this looks like a JSON tool call (don't stream it)
                    # Some models output tool calls as JSON in content
                    stripped = response_content.strip()

                    # Detect model control tokens that shouldn't be displayed
                    if '<|im_start|>' in stripped or '<|im_end|>' in stripped:
                        looks_like_tool_call = True
                    # Detect JSON tool calls at the start
                    elif stripped.startswith("{"):
                        if ('"name"' in stripped or
                            '"name' in stripped or
                            'name":' in stripped or
                            '"arguments"' in stripped):
                            looks_like_tool_call = True
                        elif len(stripped) < 80:
                            looks_like_tool_call = True
                    # Detect JSON tool calls appearing mid-content
                    elif '{"name":' in stripped or '{"name" :' in stripped:
                        looks_like_tool_call = True

                    # Detect repetition (model stuck in loop)
                    # Check if the same pattern appears multiple times
                    if len(stripped) > 500:
                        # Look for repeated JSON patterns
                        if stripped.count('{"name":') > 2 or stripped.count('"arguments"') > 2:
                            repetition_detected = True
                            self.console.print_warning(
                                "Model appears to be repeating itself. Stopping generation."
                            )
                            break

                    # Only stream if it doesn't look like a tool call
                    if not looks_like_tool_call:
                        if not streaming_started:
                            self.console.stop_status()
                            self.console.start_streaming()
                            streaming_started = True
                        self.console.stream_chunk(chunk.content)

                if chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)

            # If interrupted during streaming, break out of the main loop
            if interrupted:
                break

            # If repetition detected, clean up and try to extract what we can
            if repetition_detected:
                self.console.stop_status()
                if streaming_started and self.console._streaming_panel:
                    self.console._streaming_panel.finish()
                    self.console._streaming_panel = None

            # Stop spinner if we never started streaming
            if not streaming_started:
                self.console.stop_status()

            # Extract JSON tool calls from accumulated content if no native tool calls
            # This handles models that output tool calls as JSON in content
            if response_content and not tool_calls:
                remaining_content, parsed_calls = extract_json_tool_call(response_content)
                if parsed_calls:
                    tool_calls = parsed_calls
                    response_content = remaining_content or ""

            # Deduplicate tool calls - some models output the same call multiple times
            if tool_calls:
                seen = set()
                unique_calls = []
                for tc in tool_calls:
                    # Create a key from name and serialized arguments
                    key = (tc.name, str(sorted(tc.arguments.items()) if tc.arguments else []))
                    if key not in seen:
                        seen.add(key)
                        unique_calls.append(tc)
                tool_calls = unique_calls

            # Finish streaming if we started (and we have real content, not tool calls)
            if streaming_started and response_content and not tool_calls:
                self.console.finish_streaming()
            elif streaming_started:
                # We were streaming but got tool calls - discard the streaming panel
                if self.console._streaming_panel:
                    self.console._streaming_panel.finish()
                    self.console._streaming_panel = None

            # If we have tool calls, execute them
            if tool_calls:
                # Add assistant message with tool calls
                self.messages.append(
                    Message(
                        role="assistant",
                        content=response_content or "",
                        tool_calls=tool_calls,
                    )
                )

                # Execute each tool and add results
                for tc in tool_calls:
                    # Check for interrupt before each tool
                    if await interrupt_controller.check_interrupted() != InterruptType.NONE:
                        interrupted = True
                        self.console.print_warning("Interrupted by user.")
                        break

                    self.console.print_tool_call(tc.name, tc.arguments)

                    # Check if tool is allowed in plan mode
                    plan_manager = PlanManager()
                    if not plan_manager.is_tool_allowed(tc.name):
                        self.console.print_warning(
                            f"Tool '{tc.name}' blocked in plan mode. "
                            "Only read-only tools allowed until plan is approved."
                        )
                        self.messages.append(
                            Message(
                                role="tool",
                                content=f"Tool '{tc.name}' is not allowed in plan mode. "
                                        "Please use create_plan to create a plan, then the user "
                                        "will /approve it before write operations can proceed.",
                            )
                        )
                        continue

                    # Show status while executing
                    self.console.start_status(f"Running {tc.name}...")

                    # Check if confirmation is needed
                    if self._requires_confirmation(tc.name, tc.arguments):
                        # Get the directory this tool operates in
                        target_dir = self._get_tool_directory(tc.name, tc.arguments)

                        # Check if directory is already approved
                        if not self._is_directory_approved(target_dir):
                            # Stop spinner for permission prompt
                            self.console.stop_status()
                            # Ask for directory-level permission
                            if self.console.confirm_directory(target_dir, tc.name):
                                # Approve this directory for future operations
                                self._approve_directory(target_dir)
                                # Restart spinner for execution
                                self.console.start_status(f"Running {tc.name}...")
                            else:
                                # Add refusal to conversation
                                self.messages.append(
                                    Message(
                                        role="tool",
                                        content="User declined to execute this tool.",
                                    )
                                )
                                continue

                    # Execute the tool
                    result = await self.registry.execute(tc.name, tc.arguments)

                    # Stop status spinner and show result
                    self.console.stop_status()

                    # Extract additional context for display
                    file_path = result.metadata.get("path") or tc.arguments.get("file_path") or tc.arguments.get("path")
                    content = None

                    # For write operations, include the content for preview
                    if tc.name == "write_file":
                        content = tc.arguments.get("content", "")
                        lines_count = len(content.split("\n")) if content else 0
                        # Update output message to be more Claude Code-like
                        if result.success:
                            result_output = f"Wrote {lines_count} lines to {file_path}"
                        else:
                            result_output = result.output
                    else:
                        result_output = result.output

                    self.console.print_tool_result(
                        result_output,
                        result.success,
                        file_path=file_path,
                        content=content,
                        tool_name=tc.name,
                    )

                    # Add tool result to conversation
                    self.messages.append(
                        Message(
                            role="tool",
                            content=result.output if result.success else f"Error: {result.error}",
                        )
                    )

                # If interrupted during tool execution, break out
                if interrupted:
                    break

                # Continue the loop to get next response
                continue

            else:
                # No tool calls - this is a final response
                if response_content:
                    # Check if this looks like a failed/incomplete tool call attempt
                    stripped = response_content.strip()
                    is_truncated_json = (
                        stripped.startswith("{") and (
                            '"name"' in stripped or
                            '"name' in stripped or  # Incomplete JSON
                            'name":' in stripped
                        ) and
                        not stripped.endswith("}")  # Not a complete JSON object
                    )

                    # Also check for suspiciously short responses that start with JSON
                    is_very_short = len(stripped) < 100 and stripped.startswith("{")

                    if is_truncated_json or is_very_short:
                        # This looks like a malformed/truncated tool call
                        self.console.print_warning(
                            f"Model response appears truncated ({len(stripped)} chars). Retrying..."
                        )
                        # Add a hint to the conversation to help the model
                        self.messages.append(
                            Message(role="assistant", content=response_content)
                        )
                        self.messages.append(
                            Message(
                                role="user",
                                content="Your previous response was incomplete or truncated. "
                                        "Please provide a complete response - either respond with text "
                                        "explaining what you'll do, or use a tool with valid JSON."
                            )
                        )
                        # Continue the loop to retry
                        continue

                    # Only print if we didn't already stream it
                    # (finish_streaming already rendered the content)
                    if not streaming_started:
                        self.console.print_assistant_message(response_content)
                    self.messages.append(
                        Message(role="assistant", content=response_content)
                    )

                # Done
                break

        if iteration >= max_iterations:
            self.console.print_warning(
                f"Reached maximum iterations ({max_iterations}). Stopping."
            )

        # Update status line with final context usage
        self._update_context_status()

    def _requires_confirmation(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Check if a tool execution requires user confirmation."""
        # Check permission mode first
        mode = permission_mode_manager.current

        # Auto-accept mode skips all confirmations
        if mode == PermissionMode.AUTO_ACCEPT:
            return False

        # Plan mode should block writes (handled separately), but still ask for confirmations
        # on any operations that do get through

        # File writes need confirmation
        if tool_name in ("write_file", "edit_file") and self.settings.confirm_writes:
            return True

        # Shell commands need confirmation
        if tool_name == "bash" and self.settings.confirm_commands:
            command = arguments.get("command", "")
            # Skip confirmation for read-only commands
            safe_commands = [
                "ls", "cat", "head", "tail", "grep", "find", "pwd", "echo",
                "which", "type", "file", "wc", "du", "df", "date", "whoami",
                "uname", "env", "printenv", "test", "[",
            ]
            first_word = command.split()[0] if command.split() else ""
            if first_word in safe_commands:
                return False
            return True

        # Git commits need confirmation
        if tool_name == "git_commit" and self.settings.confirm_writes:
            return True

        return False

    def _update_context_status(self) -> None:
        """Update the console status bar with current context usage."""
        usage = self.get_context_usage()
        self.console.update_status_bar(
            context_used=usage.used,
            context_max=usage.max_tokens,
            model=self.provider.model,
        )

    def _get_tool_directory(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Extract the target directory from tool arguments."""
        # For file operations, get the parent directory of the file path
        if tool_name in ("write_file", "edit_file", "read_file"):
            file_path = arguments.get("path", arguments.get("file_path", ""))
            if file_path:
                path = Path(file_path)
                if path.is_absolute():
                    return str(path.parent.resolve())
                else:
                    return str((Path.cwd() / path).parent.resolve())

        # For bash commands, use the working directory if specified, else cwd
        if tool_name == "bash":
            cwd = arguments.get("cwd", arguments.get("working_dir", ""))
            if cwd:
                return str(Path(cwd).resolve())
            return str(Path.cwd().resolve())

        # For git operations, use the repo path or cwd
        if tool_name in ("git_commit", "git_status", "git_diff"):
            repo_path = arguments.get("repo_path", arguments.get("path", ""))
            if repo_path:
                return str(Path(repo_path).resolve())
            return str(Path.cwd().resolve())

        # Default to current working directory
        return str(Path.cwd().resolve())

    def _is_directory_approved(self, directory: str) -> bool:
        """Check if a directory (or its parent) has been approved."""
        dir_path = Path(directory).resolve()

        for approved in self.approved_directories:
            approved_path = Path(approved).resolve()
            # Check if the directory is the approved dir or a subdirectory of it
            try:
                dir_path.relative_to(approved_path)
                return True
            except ValueError:
                continue

        return False

    def _approve_directory(self, directory: str) -> None:
        """Approve a directory for tool execution."""
        resolved = str(Path(directory).resolve())
        self.approved_directories.add(resolved)
        self.console.print_info(f"Directory approved: {resolved}")
