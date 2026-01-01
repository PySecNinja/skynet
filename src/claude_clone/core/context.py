"""Context window management with auto-summarization."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from claude_clone.core.tokens import TokenCounter

if TYPE_CHECKING:
    from claude_clone.llm.ollama_provider import Message, OllamaProvider


@dataclass
class ContextUsage:
    """Token usage statistics."""

    used: int
    max_tokens: int
    available: int
    percent: float

    @property
    def is_high(self) -> bool:
        """Check if usage is getting high (>75%)."""
        return self.percent > 75

    @property
    def is_critical(self) -> bool:
        """Check if usage is critical (>90%)."""
        return self.percent > 90


class ContextManager:
    """Manage conversation context within token limits.

    Automatically summarizes older messages when the context
    window approaches its limit, preserving recent exchanges
    for immediate context.
    """

    def __init__(
        self,
        provider: "OllamaProvider",
        max_tokens: int = 32768,
        reserve_tokens: int = 4096,
        summarize_threshold: float = 0.75,
    ):
        """Initialize context manager.

        Args:
            provider: The LLM provider for generating summaries.
            max_tokens: Maximum context window size.
            reserve_tokens: Tokens to reserve for response generation.
            summarize_threshold: Fraction of max_tokens at which to summarize.
        """
        self.provider = provider
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.threshold = int(max_tokens * summarize_threshold)
        self.counter = TokenCounter()

    def get_usage(self, messages: list["Message"]) -> ContextUsage:
        """Get current token usage statistics.

        Args:
            messages: Current message list.

        Returns:
            ContextUsage with current statistics.
        """
        used = self.counter.count_messages(messages)
        available = self.max_tokens - used - self.reserve_tokens
        percent = round(used / self.max_tokens * 100, 1)

        return ContextUsage(
            used=used,
            max_tokens=self.max_tokens,
            available=max(0, available),
            percent=percent,
        )

    def should_summarize(self, messages: list["Message"]) -> bool:
        """Check if conversation should be summarized.

        Args:
            messages: Current message list.

        Returns:
            True if token usage exceeds threshold.
        """
        used = self.counter.count_messages(messages)
        return used > self.threshold

    async def summarize_conversation(
        self,
        messages: list["Message"],
        keep_recent: int = 4,
    ) -> list["Message"]:
        """Summarize older messages to reduce context size.

        Keeps the system prompt and recent exchanges intact,
        replacing older messages with a summary.

        Args:
            messages: Current message list.
            keep_recent: Number of recent exchange pairs to keep.

        Returns:
            New message list with summary replacing old messages.
        """
        from claude_clone.llm.ollama_provider import Message

        # Need at least system + some messages to summarize
        min_messages = 1 + (keep_recent * 2) + 2
        if len(messages) <= min_messages:
            return messages

        # Split messages
        system_msg = messages[0]
        # Keep last N pairs of exchanges (user + assistant)
        recent_count = keep_recent * 2
        old_messages = messages[1:-recent_count]
        recent_messages = messages[-recent_count:]

        # Nothing to summarize
        if len(old_messages) < 2:
            return messages

        # Build summary prompt
        conversation_text = self._format_messages_for_summary(old_messages)

        summary_messages = [
            Message(
                role="system",
                content="You are a helpful assistant that creates concise conversation summaries. "
                        "Summarize the key points, decisions, and context from the conversation. "
                        "Focus on information needed to continue the conversation coherently. "
                        "Be concise but preserve important details.",
            ),
            Message(
                role="user",
                content=f"Summarize this conversation:\n\n{conversation_text}\n\n"
                        "Provide a brief summary capturing key points and context.",
            ),
        ]

        # Generate summary
        summary = ""
        async for chunk in self.provider.chat(summary_messages, tools=None, stream=False):
            if chunk.content:
                summary += chunk.content

        # Create new message list with summary
        summary_msg = Message(
            role="system",
            content=f"[Previous conversation summary]\n{summary}\n[End of summary]",
        )

        return [system_msg, summary_msg, *recent_messages]

    def _format_messages_for_summary(self, messages: list["Message"]) -> str:
        """Format messages as text for summarization.

        Args:
            messages: Messages to format.

        Returns:
            Formatted conversation text.
        """
        lines = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content[:1000]  # Truncate very long messages
            if len(msg.content) > 1000:
                content += "... [truncated]"

            lines.append(f"{role}: {content}")

            # Include tool calls if present
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"  [Called tool: {tc.name}]")

        return "\n\n".join(lines)

    def estimate_tokens_for_response(self, messages: list["Message"]) -> int:
        """Estimate tokens available for response.

        Args:
            messages: Current message list.

        Returns:
            Estimated tokens available for the next response.
        """
        used = self.counter.count_messages(messages)
        return max(0, self.max_tokens - used - self.reserve_tokens)
