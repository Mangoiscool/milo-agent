"""
Short-term memory implementation
Simple list-based memory with message limit and pruning
"""

from typing import Dict, List, Optional

from core.llm.base import Message, Role
from core.memory.base import BaseMemory


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

    # 角色基础分（用于智能修剪）
    _ROLE_WEIGHTS: Dict[Role, int] = {
        Role.SYSTEM: 100,      # 系统消息最重要
        Role.ASSISTANT: 60,    # 助手回复
        Role.USER: 40,         # 用户输入
        Role.TOOL: 20,         # 工具结果通常可丢弃
    }

    # 关键词加分（用于智能修剪）
    _KEYWORD_BOOSTS: Dict[str, int] = {
        "错误": 30,
        "error": 30,
        "总结": 20,
        "summary": 20,
        "重要": 20,
        "important": 20,
        "失败": 25,
        "failed": 25,
        "成功": 20,
        "success": 20,
        "注意": 15,
        "note": 15,
        "warn": 15,
        "warning": 15,
    }

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

    def _score_message(self, message: Message, position: int, total: int) -> float:
        """
        评分单条消息重要性（私有方法）

        评分因素：
        - 角色权重：SYSTEM > ASSISTANT > USER > TOOL
        - 内容长度：较长的消息可能更重要
        - 时间衰减：最近消息得分更高
        - 关键词：包含特定关键词得分更高

        Args:
            message: 消息对象
            position: 消息在历史中的位置（从0开始）
            total: 消息总数

        Returns:
            得分 (0-100+)
        """
        score = 0.0

        # 1. 角色基础分
        score += self._ROLE_WEIGHTS.get(message.role, 10)

        # 2. 内容长度得分 (每10字+1分，上限20分)
        if message.content:
            length_score = min(len(message.content) // 10, 20)
            score += length_score

        # 3. 时间衰减 (最近消息得分更高)
        recency = position / total if total > 0 else 0
        score += recency * 30

        # 4. 关键词得分
        if message.content:
            content_lower = message.content.lower()
            for keyword, boost in self._KEYWORD_BOOSTS.items():
                if keyword.lower() in content_lower:
                    score += boost

        # 5. 工具调用加分
        if message.tool_calls:
            score += 15

        return score

    def _score_messages(self, messages: List[Message]) -> List[float]:
        """
        批量评分消息（私有方法）

        Args:
            messages: 消息列表

        Returns:
            每条消息的得分列表
        """
        total = len(messages)
        return [self._score_message(msg, i, total) for i, msg in enumerate(messages)]

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
        if self.use_intelligent_pruning:
            # Intelligent pruning: score and keep best messages
            scores = self._score_messages(other_messages)

            # Pair messages with scores and sort by score
            scored_messages = list(zip(other_messages, scores))
            scored_messages.sort(key=lambda x: x[1], reverse=True)

            # Keep highest scoring messages
            kept_messages = [msg for msg, _ in scored_messages[:available_slots]]
            kept_ids = {id(msg) for msg in kept_messages}  # 使用 id() 创建标识符集合

            # Reconstruct maintaining original order for kept messages
            result = [m for m in self._messages if id(m) in kept_ids or m.role == Role.SYSTEM]

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
