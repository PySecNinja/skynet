"""Interrupt handling for async operations."""

import asyncio
from enum import Enum


class InterruptType(Enum):
    """Types of interrupt signals."""

    NONE = "none"
    SOFT = "soft"  # Stop current operation gracefully
    HARD = "hard"  # Force stop immediately


class InterruptController:
    """Thread-safe interrupt signaling between UI and Agent.

    Singleton pattern ensures consistent state across the application.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._interrupt_type = InterruptType.NONE
        self._lock = asyncio.Lock()
        self._initialized = True

    async def signal_interrupt(
        self, interrupt_type: InterruptType = InterruptType.SOFT
    ) -> None:
        """Signal an interrupt to stop current operation."""
        async with self._lock:
            self._interrupt_type = interrupt_type

    async def check_interrupted(self) -> InterruptType:
        """Check if an interrupt has been signaled."""
        async with self._lock:
            return self._interrupt_type

    async def clear(self) -> None:
        """Clear any pending interrupt."""
        async with self._lock:
            self._interrupt_type = InterruptType.NONE

    def is_interrupted_sync(self) -> bool:
        """Non-async check for use in synchronous callbacks."""
        return self._interrupt_type != InterruptType.NONE

    def signal_interrupt_sync(
        self, interrupt_type: InterruptType = InterruptType.SOFT
    ) -> None:
        """Synchronous version for use in key binding callbacks."""
        self._interrupt_type = interrupt_type

    def clear_sync(self) -> None:
        """Synchronous clear for use in callbacks."""
        self._interrupt_type = InterruptType.NONE


# Global instance
interrupt_controller = InterruptController()
