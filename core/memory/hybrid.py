"""
混合记忆系统
结合短期记忆和长期记忆的优势
"""

from __future__ import annotations

from typing import List, Optional

from core.llm.base import Message, Role
from core.memory.base import BaseMemory
from core.memory.long_term import LongTermMemory
from core.memory.short_term import ShortTermMemory


class HybridMemory(BaseMemory):
    """
    混合记忆系统

    结合短期记忆和长期记忆：
    - 短期记忆：保留最近对话上下文，保证对话连贯
    - 长期记忆：语义检索历史重要信息，补充相关上下文

    使用场景：
    - 需要记住用户偏好的 Agent
    - 需要跨会话记忆的对话系统
    - 需要长期知识积累的应用

    使用示例：
        from core.memory.hybrid import HybridMemory
        from core.memory.short_term import ShortTermMemory
        from core.memory.long_term import LongTermMemory
        from core.rag.embeddings import create_embedding

        # 初始化
        embedding = create_embedding("ollama")
        memory = HybridMemory(
            short_term=ShortTermMemory(max_messages=20),
            long_term=LongTermMemory(embedding_model=embedding)
        )

        # 添加消息（同时存入短期和长期）
        memory.add(Message(role=Role.USER, content="我喜欢 Python"))

        # 构建上下文（短期 + 检索到的长期记忆）
        context = memory.build_context("编程相关话题")
        for msg in context:
            print(f"{msg.role}: {msg.content}")
    """

    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        short_term_limit: int = 20,
        long_term_limit: int = 5
    ):
        """
        初始化混合记忆

        Args:
            short_term: 短期记忆实例（可选，不提供则创建默认实例）
            long_term: 长期记忆实例（可选，不提供则创建默认实例）
            short_term_limit: 短期记忆最大消息数
            long_term_limit: 长期记忆检索时的默认返回数
        """
        super().__init__()

        # 短期记忆
        self.short_term = short_term or ShortTermMemory(max_messages=short_term_limit)

        # 长期记忆（需要 embedding_model）
        self.long_term = long_term
        if self.long_term is None:
            self.logger.warning(
                "LongTermMemory not provided. "
                "Long-term memory features will be disabled. "
                "Provide a LongTermMemory instance for full functionality."
            )

        self.short_term_limit = short_term_limit
        self.long_term_limit = long_term_limit

    def add(self, message: Message) -> None:
        """
        添加消息到记忆

        同时存入短期记忆和长期记忆

        Args:
            message: 要添加的消息
        """
        # 添加到短期记忆
        self.short_term.add(message)

        # 添加到长期记忆
        if self.long_term:
            self.long_term.add(message)

        self.logger.debug(
            f"Added message to HybridMemory: "
            f"short_term={self.short_term.count()}, "
            f"long_term={self.long_term.count() if self.long_term else 0}"
        )

    def build_context(
        self,
        current_query: str,
        short_term_limit: Optional[int] = None,
        long_term_limit: Optional[int] = None,
        include_system_prompt: bool = True
    ) -> List[Message]:
        """
        构建完整上下文

        组合短期记忆和长期记忆，构建适合 LLM 输入的上下文：

        1. 检索长期记忆：找到与当前话题相关的历史信息
        2. 添加系统提示：说明这是相关历史信息
        3. 添加短期记忆：最近的对话历史

        Args:
            current_query: 当前用户输入/话题
            short_term_limit: 短期记忆返回数量
            long_term_limit: 长期记忆检索数量
            include_system_prompt: 是否包含系统提示说明

        Returns:
            完整的上下文消息列表
        """
        messages = []

        short_term_limit = short_term_limit or self.short_term_limit
        long_term_limit = long_term_limit or self.long_term_limit

        # 1. 检索长期记忆
        if self.long_term and current_query:
            try:
                retrieved = self.long_term.retrieve(
                    query=current_query,
                    top_k=long_term_limit,
                    min_similarity=0.3  # 过滤掉太不相关的结果
                )

                if retrieved and include_system_prompt:
                    # 构建系统提示
                    context_parts = ["以下是与当前话题相关的历史信息：\n"]
                    for i, mem in enumerate(retrieved, 1):
                        # 格式化记忆内容
                        role = mem.entry.metadata.get("role", "unknown")
                        content_preview = mem.entry.content[:200]
                        if len(mem.entry.content) > 200:
                            content_preview += "..."
                        context_parts.append(f"{i}. [{role}] {content_preview}")

                    context_content = "\n".join(context_parts)
                    messages.append(Message(
                        role=Role.SYSTEM,
                        content=context_content
                    ))

                    self.logger.debug(
                        f"Added {len(retrieved)} retrieved memories to context"
                    )

            except Exception as e:
                self.logger.error(f"Failed to retrieve long-term memories: {e}")

        # 2. 添加短期记忆（最近对话）
        recent_messages = self.short_term.get_recent(short_term_limit)
        messages.extend(recent_messages)

        self.logger.debug(
            f"Built context: {len(messages)} messages "
            f"({len(messages) - len(recent_messages)} from long-term, "
            f"{len(recent_messages)} from short-term)"
        )

        return messages

    def get_all(self) -> List[Message]:
        """
        获取所有消息

        Returns:
            短期记忆中的所有消息
        """
        return self.short_term.get_all()

    def get_recent(self, n: int) -> List[Message]:
        """
        获取最近 N 条消息

        Args:
            n: 消息数量

        Returns:
            最近的 N 条消息
        """
        return self.short_term.get_recent(n)

    def clear(self) -> None:
        """
        清空所有记忆

        同时清空短期记忆和长期记忆
        """
        self.short_term.clear()
        if self.long_term:
            self.long_term.clear()
        self.logger.info("HybridMemory cleared")

    def clear_short_term(self) -> None:
        """仅清空短期记忆"""
        self.short_term.clear()
        self.logger.info("Short-term memory cleared")

    def clear_long_term(self) -> None:
        """仅清空长期记忆"""
        if self.long_term:
            self.long_term.clear()
            self.logger.info("Long-term memory cleared")

    def count(self) -> int:
        """
        获取消息总数

        Returns:
            短期记忆中的消息数
        """
        return self.short_term.count()

    def count_long_term(self) -> int:
        """
        获取长期记忆条目数

        Returns:
            长期记忆中的条目数
        """
        if self.long_term:
            return self.long_term.count()
        return 0

    def search_long_term(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List:
        """
        搜索长期记忆

        直接暴露长期记忆的检索功能

        Args:
            query: 查询文本
            top_k: 返回数量
            min_similarity: 最小相似度

        Returns:
            RetrievedMemory 列表
        """
        if not self.long_term:
            self.logger.warning("Long-term memory not available")
            return []

        return self.long_term.retrieve(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity
        )

    def get_memory_stats(self) -> dict:
        """
        获取记忆统计信息

        Returns:
            包含短期和长期记忆统计的字典
        """
        stats = {
            "short_term": {
                "count": self.short_term.count(),
                "max_messages": self.short_term.max_messages
            },
            "long_term": {
                "available": self.long_term is not None,
                "count": self.long_term.count() if self.long_term else 0
            }
        }
        return stats

    def __repr__(self) -> str:
        return (
            f"<HybridMemory "
            f"short_term={self.short_term.count()}/{self.short_term.max_messages} "
            f"long_term={self.long_term.count() if self.long_term else 'N/A'}>"
        )