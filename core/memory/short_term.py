"""
Short-term memory implementation
Simple list-based memory with message limit and pruning
"""

from typing import List, Optional

from core.llm.base import Message, Role
from core.memory.base import BaseMemory
from core.memory.scoring import MessageScorer


class ShortTermMemory(BaseMemory):
    """
    Short-term memory with automatic pruning

    Features:
    - List-based storage
    - Configurable message limit
    - Smart pruning: preserves SYSTEM messages + recent messages
    - Intelligent scoring-based pruning (optional)

    Pruning strategies:
    1. Simple (default): removes the oldest non-SYSTEM messages
       - When limit is exceeded, removes the oldest non-SYSTEM messages
       - All SYSTEM messages are preserved
       - Recent messages are prioritized
    2. Intelligent (optional): scores messages based on importance
       - SYSTEM messages always preserved
       - Other messages scored by role, length, recency, and keywords
       - Highest scoring messages are kept
    """

    def __init__(
        self,
        max_messages: int = 50,
        use_intelligent_pruning: bool = False
    ):
        """
        Initialize short-term memory

        Args:
            max_messages: Maximum number of messages to store (default: 50)
            use_intelligent_pruning: Enable scoring-based pruning (default: False)
        """
        super().__init__()
        self.max_messages = max_messages
        self.use_intelligent_pruning = use_intelligent_pruning
        self._messages: List[Message] = []
        self._scorer: Optional[MessageScorer] = MessageScorer() if use_intelligent_pruning else None
    
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

        Strategies:
        1. Simple: removes the oldest non-SYSTEM messages
           - Keep all SYSTEM messages
           - Keep the most recent non-SYSTEM messages
        2. Intelligent: scores messages by importance
           - SYSTEM messages always preserved
           - Other messages scored and highest kept
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

        # Select pruning strategy
        if self.use_intelligent_pruning and self._scorer:
            # Intelligent pruning: score and keep best messages
            scores = self._scorer.score_messages(other_messages)

            # Pair messages with scores and sort by score
            scored_messages = list(zip(other_messages, scores))
            scored_messages.sort(key=lambda x: x[1], reverse=True)

            # Keep highest scoring messages
            kept_messages = [msg for msg, _ in scored_messages[:available_slots]]

            # Reconstruct maintaining original order for kept messages
            kept_set = set(kept_messages)
            result = [m for m in self._messages if m in kept_set or m.role == Role.SYSTEM]

            removed_count = len(self._messages) - len(result)
            self._messages = result

            self.logger.info(
                f"Pruned {removed_count} message(s) using intelligent strategy. "
                f"Kept {len(system_messages)} SYSTEM + {len(kept_messages)} scored messages"
            )
        else:
            # Simple pruning: keep most recent messages
            kept_other = other_messages[-available_slots:] if available_slots > 0 else []

            # Reconstruct message list preserving order
            result = [m for m in self._messages if m.role == Role.SYSTEM] + kept_other

            removed_count = len(self._messages) - len(result)
            self._messages = result

            self.logger.info(
                f"Pruned {removed_count} message(s) using simple strategy. "
                f"Kept {len(system_messages)} SYSTEM + {len(kept_other)} recent messages"
            )
