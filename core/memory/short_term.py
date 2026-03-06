"""
Short-term memory implementation
Simple list-based memory with message limit and pruning
"""

from typing import List

from core.llm.base import Message, Role
from core.memory.base import BaseMemory


class ShortTermMemory(BaseMemory):
    """
    Short-term memory with automatic pruning
    
    Features:
    - List-based storage
    - Configurable message limit
    - Smart pruning: preserves SYSTEM messages + recent messages
    
    Pruning strategy:
    - When limit is exceeded, removes the oldest non-SYSTEM messages
    - All SYSTEM messages are preserved
    - Recent messages are prioritized
    """
    
    def __init__(self, max_messages: int = 50):
        """
        Initialize short-term memory
        
        Args:
            max_messages: Maximum number of messages to store (default: 50)
        """
        super().__init__()
        self.max_messages = max_messages
        self._messages: List[Message] = []
    
    def add(self, message: Message) -> None:
        """
        Add a message to memory
        
        Automatically triggers pruning if limit is exceeded.
        
        Args:
            message: Message to add
        """
        self._messages.append(message)
        self.logger.debug(f"Added message: {message.role.value} - {message.content[:50]}...")
        
        # Prune if needed
        if self.count() > self.max_messages:
            self._prune()
    
    def get_all(self) -> List[Message]:
        """
        Get all messages
        
        Returns:
            Copy of all messages list
        """
        return self._messages.copy()
    
    def clear(self) -> None:
        """Clear all messages from memory"""
        self._messages.clear()
        self.logger.info("Memory cleared")
    
    def get_recent(self, n: int) -> List[Message]:
        """
        Get the most recent n messages
        
        Args:
            n: Number of messages to retrieve
        
        Returns:
            List of recent messages (may be less than n if not enough messages)
        """
        if n <= 0:
            return []
        return self._messages[-n:]
    
    def count(self) -> int:
        """
        Get total message count
        
        Returns:
            Number of messages currently in memory
        """
        return len(self._messages)
    
    def _prune(self) -> None:
        """
        Prune messages when limit is exceeded
        
        Strategy:
        1. Separate SYSTEM and non-SYSTEM messages
        2. Keep all SYSTEM messages
        3. Keep the most recent non-SYSTEM messages
        4. Remove the oldest non-SYSTEM messages
        """
        # Separate messages by role
        system_messages = [m for m in self._messages if m.role == Role.SYSTEM]
        other_messages = [m for m in self._messages if m.role != Role.SYSTEM]
        
        # Calculate how many non-SYSTEM messages we can keep
        available_slots = self.max_messages - len(system_messages)
        
        if available_slots < 0:
            # Edge case: too many SYSTEM messages
            self.logger.warning(
                f"Too many SYSTEM messages ({len(system_messages)}) "
                f"exceeds limit ({self.max_messages})"
            )
            # Keep all SYSTEM messages anyway
            self._messages = system_messages
            return
        
        # Keep the most recent non-SYSTEM messages
        kept_other = other_messages[-available_slots:] if available_slots > 0 else []
        
        # Reconstruct message list preserving order
        # Strategy: maintain original order as much as possible
        result = []
        system_idx = 0
        other_idx = 0
        
        # First pass: add all SYSTEM messages in order
        for msg in self._messages:
            if msg.role == Role.SYSTEM:
                result.append(msg)
        
        # Second pass: add kept non-SYSTEM messages
        result.extend(kept_other)
        
        removed_count = self.count() - len(result)
        self._messages = result
        
        self.logger.info(
            f"Pruned {removed_count} message(s). "
            f"Kept {len(system_messages)} SYSTEM + {len(kept_other)} recent messages"
        )
