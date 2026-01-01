"""Ollama LLM provider with streaming and tool support."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import ollama
from ollama import AsyncClient

from claude_clone.config import Settings, get_model_config


def extract_json_tool_call(content: str) -> tuple[str | None, list["ToolCall"]]:
    """
    Extract tool calls from JSON in content for models that don't use native tool calling.

    Some models (like qwen2.5-coder) output tool calls as JSON in the content field
    instead of using Ollama's native tool_calls format.

    Handles multiple formats:
    1. {"name": "tool_name", "arguments": {...}}
    2. tool_name {...}  (tool name followed by JSON arguments)
    3. tool_name({...}) (function call style)

    Returns (remaining_content, tool_calls)
    """
    tool_calls = []
    matches = []

    # Known tool names to look for
    known_tools = [
        "read_file", "write_file", "edit_file", "grep", "glob",
        "bash", "git_status", "git_diff", "git_commit", "git_log", "git_branch",
        "web_search", "web_fetch", "create_plan", "todo_write"
    ]

    def find_balanced_json(text: str, start_idx: int) -> tuple[str | None, int]:
        """Find a balanced JSON object starting at start_idx. Returns (json_str, end_idx) or (None, -1)."""
        if start_idx >= len(text) or text[start_idx] != '{':
            return None, -1

        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        j = start_idx

        while j < len(text):
            char = text[j]

            if escape_next:
                escape_next = False
                j += 1
                continue

            if char == '\\' and in_string:
                escape_next = True
                j += 1
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_idx:j + 1], j + 1
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1

            j += 1

        return None, -1

    # First, try to find {"name": ..., "arguments": ...} format
    i = 0
    while i < len(content):
        if content[i] == '{':
            json_str, end_idx = find_balanced_json(content, i)
            if json_str:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "name" in data and "arguments" in data:
                        tool_calls.append(ToolCall(
                            name=data["name"],
                            arguments=data["arguments"] if isinstance(data["arguments"], dict) else {}
                        ))
                        matches.append((i, end_idx))
                        i = end_idx
                        continue
                except json.JSONDecodeError:
                    pass
            i += 1
        else:
            i += 1

    # If no standard format found, try "tool_name {...}" or "tool_name({...})" formats
    if not tool_calls:
        for tool_name in known_tools:
            # Pattern: tool_name { ... } or tool_name({ ... })
            patterns = [
                (tool_name + r'\s*\(\s*\{', 2),  # tool_name({ - need to skip "(" before "{"
                (tool_name + r'\s*\{', 0),       # tool_name { - brace immediately follows
            ]

            for pattern, offset_adjust in patterns:
                for match in re.finditer(pattern, content):
                    # Find the opening brace
                    brace_start = match.end() - 1
                    json_str, end_idx = find_balanced_json(content, brace_start)

                    if json_str:
                        try:
                            args = json.loads(json_str)
                            if isinstance(args, dict):
                                tool_calls.append(ToolCall(name=tool_name, arguments=args))
                                # Adjust match bounds to include tool name and optional parens
                                match_end = end_idx
                                # Check for closing paren if we had opening paren
                                if offset_adjust == 2 and match_end < len(content) and content[match_end:match_end+1] == ')':
                                    match_end += 1
                                matches.append((match.start(), match_end))
                        except json.JSONDecodeError:
                            pass

    # If we found tool calls, remove them from content
    if tool_calls:
        remaining = content
        for start, end in reversed(sorted(matches)):  # Sort and reverse to maintain positions
            remaining = remaining[:start] + remaining[end:]
        remaining = remaining.strip()
        return remaining if remaining else None, tool_calls

    return content, []


@dataclass
class ToolCall:
    """Represents a tool call from the model."""

    name: str
    arguments: dict[str, Any]


@dataclass
class ChatChunk:
    """A chunk from the streaming response."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool responses

    def to_dict(self) -> dict[str, Any]:
        """Convert to Ollama message format."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                }
                for tc in self.tool_calls
            ]
        return msg


class OllamaProvider:
    """Ollama LLM provider with streaming and tool support."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:32b",
        host: str = "http://localhost:11434",
        settings: Settings | None = None,
    ):
        self.model = model
        self.host = host
        self.client = AsyncClient(host=host)
        self.settings = settings or Settings()
        self.model_config = get_model_config(model)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[ChatChunk]:
        """
        Send a chat request with optional tool support.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            stream: Whether to stream the response

        Yields:
            ChatChunk objects containing content or tool calls
        """
        # Convert messages to Ollama format
        ollama_messages = [msg.to_dict() for msg in messages]

        # Prepare options
        options = {
            "temperature": self.settings.temperature,
            "num_ctx": self.settings.num_ctx,
        }

        try:
            if stream:
                async for chunk in await self.client.chat(
                    model=self.model,
                    messages=ollama_messages,
                    tools=tools,
                    stream=True,
                    options=options,
                ):
                    yield self._parse_chunk(chunk)
            else:
                response = await self.client.chat(
                    model=self.model,
                    messages=ollama_messages,
                    tools=tools,
                    stream=False,
                    options=options,
                )
                yield self._parse_response(response)

        except ollama.ResponseError as e:
            yield ChatChunk(content=f"Error: {e.error}", done=True)

    def _parse_chunk(self, chunk: Any) -> ChatChunk:
        """Parse a streaming chunk from Ollama."""
        content = None
        tool_calls = []
        done = False

        if hasattr(chunk, "message"):
            msg = chunk.message
            if hasattr(msg, "content") and msg.content:
                content = msg.content
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if hasattr(tc, "function"):
                        tool_calls.append(
                            ToolCall(
                                name=tc.function.name,
                                arguments=tc.function.arguments
                                if isinstance(tc.function.arguments, dict)
                                else json.loads(tc.function.arguments),
                            )
                        )

        if hasattr(chunk, "done"):
            done = chunk.done

        # Fallback: extract JSON tool calls from content if no native tool_calls
        # Only do this on final chunk to avoid partial JSON parsing
        if done and content and not tool_calls:
            content, tool_calls = extract_json_tool_call(content)

        return ChatChunk(content=content, tool_calls=tool_calls, done=done)

    def _parse_response(self, response: Any) -> ChatChunk:
        """Parse a non-streaming response from Ollama."""
        content = None
        tool_calls = []

        if hasattr(response, "message"):
            msg = response.message
            if hasattr(msg, "content") and msg.content:
                content = msg.content
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if hasattr(tc, "function"):
                        tool_calls.append(
                            ToolCall(
                                name=tc.function.name,
                                arguments=tc.function.arguments
                                if isinstance(tc.function.arguments, dict)
                                else json.loads(tc.function.arguments),
                            )
                        )

        # Fallback: extract JSON tool calls from content if no native tool_calls
        if content and not tool_calls:
            content, tool_calls = extract_json_tool_call(content)

        return ChatChunk(content=content, tool_calls=tool_calls, done=True)

    async def list_models(self) -> list[str]:
        """List available models."""
        response = await self.client.list()
        return [model.model for model in response.models]

    async def check_connection(self) -> bool:
        """Check if Ollama is accessible."""
        try:
            await self.client.list()
            return True
        except Exception:
            return False


def create_tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Create a tool schema for Ollama function calling."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required or [],
            },
        },
    }
