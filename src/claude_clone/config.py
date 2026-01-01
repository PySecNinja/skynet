"""Configuration management for Claude Clone."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PermissionMode(Enum):
    """Permission modes for tool execution."""

    NORMAL = "normal"  # Ask for each operation
    AUTO_ACCEPT = "auto_accept"  # Accept all operations automatically
    PLAN_MODE = "plan_mode"  # Read-only until plan approved


class PermissionModeManager:
    """Manages permission mode state with cycling.

    Singleton pattern for consistent state across the application.
    """

    _instance = None
    _MODES = [PermissionMode.NORMAL, PermissionMode.AUTO_ACCEPT, PermissionMode.PLAN_MODE]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._current = PermissionMode.NORMAL
        return cls._instance

    @property
    def current(self) -> PermissionMode:
        """Get current permission mode."""
        return self._current

    def cycle(self) -> PermissionMode:
        """Cycle to the next permission mode."""
        idx = self._MODES.index(self._current)
        self._current = self._MODES[(idx + 1) % len(self._MODES)]
        return self._current

    def set_mode(self, mode: PermissionMode) -> None:
        """Set a specific permission mode."""
        self._current = mode

    def get_display_info(self) -> tuple[str, str]:
        """Return (short_name, color) for status display."""
        return {
            PermissionMode.NORMAL: ("Normal", "green"),
            PermissionMode.AUTO_ACCEPT: ("Auto", "yellow"),
            PermissionMode.PLAN_MODE: ("Plan", "cyan"),
        }[self._current]


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_CLONE_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Ollama settings
    model: str = Field(default="qwen2.5-coder:32b", description="Ollama model to use")
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama API host")

    # Model parameters
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    num_ctx: int = Field(default=16384, description="Context window size")

    # Safety settings
    confirm_writes: bool = Field(default=True, description="Require confirmation for file writes")
    confirm_commands: bool = Field(default=True, description="Require confirmation for shell commands")
    allowed_directories: list[str] = Field(
        default_factory=lambda: ["."],
        description="Directories the AI can access",
    )

    # UI settings
    streaming: bool = Field(default=True, description="Enable streaming responses")

    # Session settings
    session_dir: Path = Field(
        default_factory=lambda: Path.home() / ".claude-clone" / "sessions",
        description="Directory to store session files",
    )
    history_file: Path = Field(
        default_factory=lambda: Path.home() / ".claude-clone" / "history",
        description="Command history file",
    )


# Model configurations for different models
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen2.5-coder:32b": {
        "context_length": 16384,  # Model's actual default context
        "supports_tools": True,
        "temperature": 0.3,
    },
    "qwen2.5-coder:14b": {
        "context_length": 32768,
        "supports_tools": True,
        "temperature": 0.3,
    },
    "qwen2.5-coder:7b": {
        "context_length": 32768,
        "supports_tools": True,
        "temperature": 0.3,
    },
    "deepseek-coder:33b": {
        "context_length": 16384,
        "supports_tools": True,
        "temperature": 0.3,
    },
    "codellama:34b": {
        "context_length": 16384,
        "supports_tools": False,
        "temperature": 0.4,
    },
    "llama3.1:8b": {
        "context_length": 128000,
        "supports_tools": True,
        "temperature": 0.5,
    },
}


def get_model_config(model: str) -> dict[str, Any]:
    """Get configuration for a specific model, with defaults for unknown models."""
    return MODEL_CONFIGS.get(
        model,
        {
            "context_length": 8192,
            "supports_tools": True,
            "temperature": 0.5,
        },
    )


# Global settings instance
settings = Settings()

# Global permission mode manager
permission_mode_manager = PermissionModeManager()
