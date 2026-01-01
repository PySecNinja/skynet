"""Token counting for context management."""

from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from claude_clone.llm.ollama_provider import Message


class TokenCounter:
    """Count tokens for context window management.

    Uses tiktoken with cl100k_base encoding as an approximation.
    Ollama models use various tokenizers, but cl100k_base provides
    a reasonable estimate for most modern models.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize with a tiktoken encoding.

        Args:
            encoding_name: The tiktoken encoding to use.
                          cl100k_base works well for most modern models.
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_message(self, message: "Message") -> int:
        """Count tokens in a single message.

        Includes overhead for message structure (role, etc).

        Args:
            message: The message to count.

        Returns:
            Approximate token count for the message.
        """
        # Base tokens for message structure
        tokens = 4  # Approximate overhead for role, separators

        # Content tokens
        tokens += self.count(message.content)

        # Tool call tokens (if any)
        if message.tool_calls:
            for tc in message.tool_calls:
                tokens += self.count(tc.name)
                tokens += self.count(str(tc.arguments))
                tokens += 4  # Overhead for tool call structure

        return tokens

    def count_messages(self, messages: list["Message"]) -> int:
        """Count total tokens in a list of messages.

        Args:
            messages: List of messages to count.

        Returns:
            Total token count.
        """
        total = 3  # Overhead for message list structure
        for msg in messages:
            total += self.count_message(msg)
        return total


# Singleton instance for convenience
_counter: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    """Get the global token counter instance."""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def count_tokens(text: str) -> int:
    """Convenience function to count tokens in text."""
    return get_token_counter().count(text)
