"""Main agent orchestration loop."""

from typing import Any

from claude_clone.config import Settings
from claude_clone.llm.ollama_provider import ChatChunk, Message, OllamaProvider, ToolCall
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

    async def process_message(self, user_input: str) -> None:
        """Process a user message and generate response with tool execution."""
        # Add user message
        self.messages.append(Message(role="user", content=user_input))

        # Get tool schemas
        tools = self.registry.get_schemas()

        # Agentic loop - keep going until we get a final response
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get response from LLM
            # We always buffer first, then decide whether to display
            # This handles models that output JSON tool calls in content
            response_content = ""
            tool_calls: list[ToolCall] = []

            self.console.print_info("Thinking...")

            async for chunk in self.provider.chat(
                messages=self.messages,
                tools=tools,
                stream=False,  # Use non-streaming for reliable tool parsing
            ):
                if chunk.content:
                    response_content += chunk.content

                if chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)

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
                    self.console.print_tool_call(tc.name, tc.arguments)

                    # Check if confirmation is needed
                    if self._requires_confirmation(tc.name, tc.arguments):
                        if not self.console.confirm(
                            f"Allow {tc.name} with these arguments?"
                        ):
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

                    self.console.print_tool_result(result.output, result.success)

                    # Add tool result to conversation
                    self.messages.append(
                        Message(
                            role="tool",
                            content=result.output if result.success else f"Error: {result.error}",
                        )
                    )

                # Continue the loop to get next response
                continue

            else:
                # No tool calls - this is a final response
                if response_content:
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

    def _requires_confirmation(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Check if a tool execution requires user confirmation."""
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
