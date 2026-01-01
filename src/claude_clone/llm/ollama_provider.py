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

    Returns (remaining_content, tool_calls)
    """
    tool_calls = []

    # Try to find JSON tool call patterns
    # Pattern 1: {"name": "tool_name", "arguments": {...}}
    json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'

    matches = list(re.finditer(json_pattern, content, re.DOTALL))

    for match in matches:
        try:
            data = json.loads(match.group())
            if "name" in data and "arguments" in data:
                tool_calls.append(ToolCall(
                    name=data["name"],
                    arguments=data["arguments"] if isinstance(data["arguments"], dict) else {}
                ))
        except json.JSONDecodeError:
            continue

    # If we found tool calls, remove them from content
    if tool_calls:
        remaining = content
        for match in reversed(matches):  # Reverse to maintain positions
            remaining = remaining[:match.start()] + remaining[match.end():]
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
