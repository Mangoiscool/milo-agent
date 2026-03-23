"""Memory system for milo-agent."""

from .base import BaseMemory
from .short_term import ShortTermMemory
from .long_term import LongTermMemory, MemoryEntry, RetrievedMemory
from .hybrid import HybridMemory

# 向后兼容：PersistentMemory 现在是 ShortTermMemory 的别名
PersistentMemory = ShortTermMemory

__all__ = [
    "BaseMemory",
    "ShortTermMemory",
    "PersistentMemory",
    "LongTermMemory",
    "MemoryEntry",
    "RetrievedMemory",
    "HybridMemory",
]
