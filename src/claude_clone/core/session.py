"""Session persistence for saving and resuming conversations."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_clone.llm.ollama_provider import Message, ToolCall


@dataclass
class SessionMetadata:
    """Metadata about a saved session."""

    id: str
    model: str
    created_at: str
    updated_at: str
    message_count: int
    title: str  # First user message or auto-generated


class SessionManager:
    """Manages saving and loading conversation sessions."""

    def __init__(self, session_dir: Path | None = None):
        self.session_dir = session_dir or Path.home() / ".claude-clone" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_id: str | None = None

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_session_path(self, session_id: str) -> Path:
        """Get the path for a session file."""
        return self.session_dir / f"{session_id}.json"

    def _message_to_dict(self, msg: Message) -> dict[str, Any]:
        """Convert a Message to a serializable dict."""
        data: dict[str, Any] = {
            "role": msg.role,
            "content": msg.content,
        }
        if msg.tool_calls:
            data["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        return data

    def _dict_to_message(self, data: dict[str, Any]) -> Message:
        """Convert a dict back to a Message."""
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [
                ToolCall(name=tc["name"], arguments=tc["arguments"])
                for tc in data["tool_calls"]
            ]
        return Message(
            role=data["role"],
            content=data["content"],
            tool_calls=tool_calls,
        )

    def save_session(
        self,
        messages: list[Message],
        model: str,
        session_id: str | None = None,
    ) -> str:
        """
        Save a conversation session to disk.

        Args:
            messages: The conversation messages
            model: The model being used
            session_id: Optional existing session ID to update

        Returns:
            The session ID
        """
        if session_id is None:
            session_id = self._generate_session_id()

        self.current_session_id = session_id
        session_path = self._get_session_path(session_id)

        # Generate title from first user message
        title = "New conversation"
        for msg in messages:
            if msg.role == "user":
                title = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
                break

        # Check if session exists for created_at
        created_at = datetime.now().isoformat()
        if session_path.exists():
            try:
                with open(session_path) as f:
                    existing = json.load(f)
                    created_at = existing.get("created_at", created_at)
            except (json.JSONDecodeError, KeyError):
                pass

        session_data = {
            "id": session_id,
            "model": model,
            "created_at": created_at,
            "updated_at": datetime.now().isoformat(),
            "title": title,
            "messages": [self._message_to_dict(msg) for msg in messages],
        }

        with open(session_path, "w") as f:
            json.dump(session_data, f, indent=2)

        return session_id

    def load_session(self, session_id: str) -> tuple[list[Message], str] | None:
        """
        Load a conversation session from disk.

        Args:
            session_id: The session ID to load

        Returns:
            Tuple of (messages, model) or None if not found
        """
        session_path = self._get_session_path(session_id)

        if not session_path.exists():
            return None

        try:
            with open(session_path) as f:
                data = json.load(f)

            messages = [self._dict_to_message(msg) for msg in data["messages"]]
            model = data.get("model", "qwen2.5-coder:32b")
            self.current_session_id = session_id

            return messages, model

        except (json.JSONDecodeError, KeyError) as e:
            return None

    def list_sessions(self, limit: int = 10) -> list[SessionMetadata]:
        """
        List recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata, newest first
        """
        sessions = []

        for path in self.session_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)

                sessions.append(
                    SessionMetadata(
                        id=data["id"],
                        model=data.get("model", "unknown"),
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                        message_count=len(data.get("messages", [])),
                        title=data.get("title", "Untitled"),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by updated_at, newest first
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions[:limit]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            session_path.unlink()
            return True
        return False

    def get_last_session_id(self) -> str | None:
        """Get the most recently updated session ID."""
        sessions = self.list_sessions(limit=1)
        return sessions[0].id if sessions else None
