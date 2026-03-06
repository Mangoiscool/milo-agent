"""
Persistent memory implementation
Extends ShortTermMemory with save/load capabilities
"""

import json
from pathlib import Path
from typing import List, Optional

from core.llm.base import Message
from core.memory.base import BaseMemory
from core.memory.short_term import ShortTermMemory


class PersistentMemory(ShortTermMemory):
    """
    Persistent memory with file-based storage

    Extends ShortTermMemory with save/load functionality
    """

    def __init__(self, max_messages: int = 50, storage_path: Optional[str] = None):
        """
        Initialize persistent memory

        Args:
            max_messages: Maximum number of messages to store (default: 50)
            storage_path: Path to save/load memory from (default: ~/.milo-agent/memory.json)
        """
        super().__init__(max_messages)
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".milo-agent" / "memory.json"

    def save(self) -> None:
        """
        Save memory to storage file

        Creates parent directories if needed.
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        messages_data = [msg.to_api_format() for msg in self.get_all()]

        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Memory saved to {self.storage_path}")

    def load(self) -> int:
        """
        Load memory from storage file

        Returns:
            Number of messages loaded (0 if file doesn't exist)
        """
        if not self.storage_path.exists():
            self.logger.info(f"No memory file found at {self.storage_path}")
            return 0

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                messages_data = json.load(f)

            # Convert API format back to Message objects
            messages = [Message.from_api_format(data) for data in messages_data]

            # Clear current memory and add loaded messages
            self._messages.clear()
            for msg in messages:
                super().add(msg)

            self.logger.info(f"Loaded {len(messages)} messages from {self.storage_path}")
            return len(messages)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to load memory: {e}")
            return 0

    def add(self, message: Message) -> None:
        """
        Add a message to memory

        Extends parent to automatically save after adding.

        Args:
            message: Message to add
        """
        super().add(message)
        # Auto-save after adding (could be made configurable)
        self.save()

    def clear(self) -> None:
        """
        Clear all messages from memory

        Extends parent to delete storage file.
        """
        super().clear()
        if self.storage_path.exists():
            self.storage_path.unlink()
            self.logger.info(f"Memory file deleted: {self.storage_path}")
