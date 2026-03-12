"""
Message importance scoring for intelligent memory pruning
"""

from typing import List
from core.llm.base import Message, Role


class MessageScorer:
    """
    评分系统：评估消息重要性

    评分因素：
    - 角色权重：SYSTEM > ASSISTANT > USER > TOOL
    - 内容长度：较长的消息可能更重要
    - 时间衰减：最近消息得分更高
    - 关键词：包含特定关键词（如 "错误"、"总结"）得分更高
    """

    # 角色基础分
    ROLE_WEIGHTS = {
        Role.SYSTEM: 100,      # 系统消息最重要
        Role.ASSISTANT: 60,    # 助手回复
        Role.USER: 40,         # 用户输入
        Role.TOOL: 20,         # 工具结果通常可丢弃
    }

    # 关键词加分
    KEYWORD_BOOSTS = {
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

    def score(self, message: Message, position: int, total: int) -> float:
        """
        评分单条消息

        Args:
            message: 消息对象
            position: 消息在历史中的位置（从0开始）
            total: 消息总数

        Returns:
            得分 (0-100+)
        """
        score = 0.0

        # 1. 角色基础分
        score += self.ROLE_WEIGHTS.get(message.role, 10)

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
            for keyword, boost in self.KEYWORD_BOOSTS.items():
                if keyword.lower() in content_lower:
                    score += boost

        # 5. 工具调用加分（说明进行了重要操作）
        if message.tool_calls:
            score += 15

        return score

    def score_messages(self, messages: List[Message]) -> List[float]:
        """
        批量评分

        Args:
            messages: 消息列表

        Returns:
            每条消息的得分列表
        """
        total = len(messages)
        return [self.score(msg, i, total) for i, msg in enumerate(messages)]
