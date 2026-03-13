"""
Agent Configuration

Provides a unified configuration class for SimpleAgent
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """
    Agent configuration

    Attributes:
        enable_stream_fallback: Enable automatic fallback from stream to async chat
        max_memory_messages: Maximum messages to store in memory
        system_prompt: System prompt for the agent
        auto_save: Enable auto-save after each message add (default: True, set False for streaming)
        use_intelligent_pruning: Enable intelligent message scoring for memory pruning
    """
    enable_stream_fallback: bool = True
    max_memory_messages: int = 50
    system_prompt: Optional[str] = None
    auto_save: bool = True
    use_intelligent_pruning: bool = False  # 默认关闭智能裁剪，保持向后兼容

    def __init__(self, **kwargs):
        """Initialize config from kwargs for flexibility"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
