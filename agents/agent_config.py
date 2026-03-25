"""
Agent Configuration

统一配置类，用于 BaseAgent、SimpleAgent、MiloAgent 等所有 Agent 类型
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """
    Agent 统一配置

    Attributes:
        enable_stream_fallback: 流式输出失败时是否自动回退到异步聊天
        max_memory_messages: 短期记忆最大消息数
        system_prompt: 系统提示词（可被构造参数覆盖）
        use_intelligent_pruning: 是否启用智能消息裁剪（基于重要性评分）
    """
    enable_stream_fallback: bool = True
    max_memory_messages: int = 50
    system_prompt: Optional[str] = None
    use_intelligent_pruning: bool = False

    def __init__(self, **kwargs):
        """Initialize config from kwargs for flexibility"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
