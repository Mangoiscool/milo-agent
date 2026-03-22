"""
长期记忆实现
支持语义检索和跨会话记忆
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.llm.base import Message, Role
from core.memory.base import BaseMemory
from core.rag.embeddings import BaseEmbedding
from core.rag.vector_store import VectorStore
from config.settings import settings


def _get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _resolve_workspace_dir() -> Path:
    """
    解析 workspace 目录

    - 如果设置了 workspace_dir，使用它
    - 如果是相对路径，基于项目根目录解析
    - 如果没设置，使用默认值 ~/.milo-agent/workspace
    """
    s = settings()
    project_root = _get_project_root()

    if s.workspace_dir:
        # 如果设置的是相对路径，基于项目根目录
        if not s.workspace_dir.is_absolute():
            return project_root / s.workspace_dir
        return s.workspace_dir

    # 默认路径
    return Path.home() / ".milo-agent" / "workspace"


def _get_default_memory_dir() -> Path:
    """获取默认长期记忆存储目录（向量存储）"""
    return _resolve_workspace_dir() / "memory_storage" / "vector_store"


DEFAULT_MEMORY_DIR = _get_default_memory_dir()


@dataclass
class MemoryEntry:
    """
    记忆条目

    存储在向量数据库中的一条记忆，包含：
    - 内容文本
    - 向量嵌入
    - 时间戳和会话 ID
    - 重要性分数
    - 元数据
    """
    id: str
    content: str
    embedding: List[float]
    timestamp: datetime
    session_id: str
    importance: float = 1.0
    memory_type: str = "message"  # message / summary / fact
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<MemoryEntry {self.id[:8]}... type={self.memory_type} score={self.importance:.2f}>"


@dataclass
class RetrievedMemory:
    """
    检索到的记忆

    包含原始记忆条目和检索相关性信息
    """
    entry: MemoryEntry
    similarity: float  # 相似度分数 (0-1，越大越相似)
    distance: float    # 向量距离 (越小越相似)

    def __repr__(self) -> str:
        return f"<RetrievedMemory sim={self.similarity:.3f} dist={self.distance:.3f}>"


class LongTermMemory(BaseMemory):
    """
    长期记忆系统

    特性：
    - 语义检索：基于向量相似度检索相关记忆
    - 持久化：自动保存到 ChromaDB 向量数据库
    - 重要性评分：自动评估记忆重要性
    - 跨会话支持：通过 session_id 区分不同会话

    使用示例：
        from core.rag.embeddings import create_embedding
        from core.memory.long_term import LongTermMemory

        # 初始化
        embedding = create_embedding("ollama", model="qwen3-embedding:0.6b")
        memory = LongTermMemory(
            embedding_model=embedding,
            session_id="session_001"
        )

        # 添加记忆
        memory.add(Message(role=Role.USER, content="我喜欢 Python 编程"))

        # 语义检索
        results = memory.retrieve("编程爱好", top_k=5)
        for r in results:
            print(f"[{r.similarity:.2f}] {r.entry.content}")
    """

    # 重要性关键词（用于计算重要性分数）
    IMPORTANT_KEYWORDS = [
        # 中文关键词
        "记住", "重要", "决定", "计划", "偏好", "喜欢", "讨厌", "目标",
        "密码", "地址", "电话", "邮箱", "名字", "叫做", "是", "工作",
        "职业", "项目", "会议", "日期", "时间", "提醒", "任务",
        # 英文关键词
        "remember", "important", "decision", "plan", "prefer", "like",
        "hate", "goal", "password", "address", "phone", "email", "name",
        "project", "meeting", "date", "time", "reminder", "task"
    ]

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        session_id: Optional[str] = None,
        persist_directory: str = str(DEFAULT_MEMORY_DIR),
        default_top_k: int = 5
    ):
        """
        初始化长期记忆

        Args:
            embedding_model: Embedding 模型实例
            session_id: 会话 ID，用于区分不同会话的记忆
            persist_directory: 向量数据库持久化目录
            default_top_k: 默认检索数量
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.session_id = session_id or str(uuid.uuid4())
        self.default_top_k = default_top_k
        self.persist_directory = persist_directory

        # 初始化向量存储
        self.vector_store = VectorStore(
            collection_name="long_term_memory",
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )

        # 内存中的条目映射（用于快速访问）
        self._entries: dict[str, MemoryEntry] = {}

        self.logger.info(
            f"LongTermMemory initialized with session_id={self.session_id[:8]}..."
        )

    def add(self, message: Message) -> None:
        """
        添加消息到长期记忆

        自动执行：
        1. 计算重要性分数
        2. 生成向量嵌入
        3. 创建 MemoryEntry
        4. 存入向量数据库

        Args:
            message: 要添加的消息
        """
        # 跳过空内容
        if not message.content or not message.content.strip():
            self.logger.debug("Skipping empty message")
            return

        # 1. 计算重要性分数
        importance = self._calculate_importance(message)

        # 2. 生成 embedding
        try:
            embedding = self.embedding_model.embed(message.content)
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return

        # 3. 创建记忆条目
        entry_id = str(uuid.uuid4())
        entry = MemoryEntry(
            id=entry_id,
            content=message.content,
            embedding=embedding,
            timestamp=datetime.now(),
            session_id=self.session_id,
            importance=importance,
            memory_type="message",
            metadata={
                "role": message.role.value,
                "has_tool_calls": bool(message.tool_calls),
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "importance": importance
            }
        )

        # 4. 存入向量存储
        try:
            self.vector_store.add_texts(
                texts=[entry.content],
                metadatas=[entry.metadata],
                ids=[entry.id]
            )
            self._entries[entry.id] = entry

            self.logger.info(
                f"Added memory entry: {entry.id[:8]}... "
                f"importance={importance:.2f} "
                f"content={entry.content[:50]}..."
            )
        except Exception as e:
            self.logger.error(f"Failed to add entry to vector store: {e}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.0,
        session_filter: Optional[str] = None
    ) -> List[RetrievedMemory]:
        """
        语义检索记忆

        Args:
            query: 查询文本
            top_k: 返回数量，默认使用 default_top_k
            min_similarity: 最小相似度阈值（0-1）
            session_filter: 可选的会话 ID 过滤

        Returns:
            检索到的记忆列表，按相似度降序排列
        """
        top_k = top_k or self.default_top_k

        if not query or not query.strip():
            return []

        try:
            # 生成查询向量
            query_embedding = self.embedding_model.embed(query)

            # 构建过滤条件
            where_filter = None
            if session_filter:
                where_filter = {"session_id": session_filter}

            # 向量检索
            results = self.vector_store.query_by_embedding(
                query_embedding=query_embedding,
                n_results=top_k * 2,  # 获取更多，再过滤
                where=where_filter
            )

            # 转换为 RetrievedMemory
            memories = []
            for result in results:
                # ChromaDB 使用余弦距离，转换为相似度
                # 余弦距离 = 1 - 余弦相似度
                distance = result.get("distance", 0)
                similarity = 1 - distance

                if similarity >= min_similarity:
                    # 重建 MemoryEntry
                    metadata = result.get("metadata", {})
                    entry = MemoryEntry(
                        id=result["id"],
                        content=result["text"],
                        embedding=[],  # 不需要返回 embedding
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                        session_id=metadata.get("session_id", ""),
                        importance=metadata.get("importance", 1.0),
                        memory_type=metadata.get("memory_type", "message"),
                        metadata=metadata
                    )

                    memories.append(RetrievedMemory(
                        entry=entry,
                        similarity=similarity,
                        distance=distance
                    ))

            # 按相似度排序
            memories.sort(key=lambda x: x.similarity, reverse=True)
            return memories[:top_k]

        except Exception as e:
            self.logger.error(f"Failed to retrieve memories: {e}")
            return []

    def retrieve_by_session(self, session_id: str, limit: int = 100) -> List[MemoryEntry]:
        """
        检索指定会话的所有记忆

        Args:
            session_id: 会话 ID
            limit: 最大返回数量

        Returns:
            该会话的记忆列表
        """
        try:
            results = self.vector_store.get(
                where={"session_id": session_id},
                limit=limit
            )

            entries = []
            for result in results:
                metadata = result.get("metadata", {})
                entry = MemoryEntry(
                    id=result["id"],
                    content=result["text"],
                    embedding=[],
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                    session_id=metadata.get("session_id", ""),
                    importance=metadata.get("importance", 1.0),
                    memory_type=metadata.get("memory_type", "message"),
                    metadata=metadata
                )
                entries.append(entry)

            return entries

        except Exception as e:
            self.logger.error(f"Failed to retrieve session memories: {e}")
            return []

    def _calculate_importance(self, message: Message) -> float:
        """
        计算消息重要性分数（0-10）

        评分因素：
        - 内容长度：较长的消息可能更重要
        - 关键词：包含重要关键词得分更高
        - 角色类型：USER 消息通常更重要
        - 工具调用：包含工具调用的消息更重要
        - 用户标记：包含"重要"等标记得分更高

        Args:
            message: 消息对象

        Returns:
            重要性分数 (0-10)
        """
        score = 1.0

        if not message.content:
            return score

        content = message.content

        # 1. 基于内容长度（每 100 字 +0.5，上限 2.0）
        length_score = min(len(content) / 100 * 0.5, 2.0)
        score += length_score

        # 2. 关键词匹配
        content_lower = content.lower()
        keyword_matches = sum(
            1 for keyword in self.IMPORTANT_KEYWORDS
            if keyword.lower() in content_lower
        )
        # 每匹配一个关键词 +0.3，上限 3.0
        score += min(keyword_matches * 0.3, 3.0)

        # 3. 角色权重
        if message.role == Role.USER:
            score += 1.0  # 用户消息更重要
        elif message.role == Role.ASSISTANT:
            score += 0.5

        # 4. 工具调用
        if message.tool_calls:
            score += 1.5

        # 5. 用户明确标记
        explicit_markers = ["【重要】", "[important]", "!important", "记住：", "remember:"]
        for marker in explicit_markers:
            if marker in content_lower:
                score += 2.0
                break

        # 6. 特殊内容类型
        if any(pattern in content for pattern in ["密码", "password", "密钥", "secret", "key"]):
            score += 2.0

        # 限制在 0-10 范围
        return min(max(score, 0), 10.0)

    def get_all(self) -> List[Message]:
        """
        获取所有记忆（不推荐，长期记忆量可能很大）

        Returns:
            空列表（长期记忆不适合全量获取）
        """
        self.logger.warning("get_all() called on LongTermMemory, returning empty list")
        return []

    def get_recent(self, n: int) -> List[Message]:
        """
        获取最近 N 条记忆（长期记忆不按时间排序）

        Returns:
            空列表（长期记忆应使用 retrieve 方法）
        """
        self.logger.warning("get_recent() called on LongTermMemory, returning empty list")
        return []

    def clear(self) -> None:
        """清空长期记忆"""
        try:
            self.vector_store.delete_collection()
            self._entries.clear()
            self.logger.info("LongTermMemory cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")

    def count(self) -> int:
        """获取记忆条目数"""
        try:
            return self.vector_store.count()
        except Exception:
            return len(self._entries)

    def delete_by_session(self, session_id: str) -> int:
        """
        删除指定会话的所有记忆

        Args:
            session_id: 会话 ID

        Returns:
            删除的条目数
        """
        try:
            # 获取该会话的所有条目
            entries = self.retrieve_by_session(session_id)
            count = len(entries)

            # 删除
            if entries:
                ids = [e.id for e in entries]
                self.vector_store.delete(ids=ids)

                # 更新内存映射
                for id in ids:
                    self._entries.pop(id, None)

            self.logger.info(f"Deleted {count} memories from session {session_id[:8]}...")
            return count

        except Exception as e:
            self.logger.error(f"Failed to delete session memories: {e}")
            return 0

    def update_importance(self, entry_id: str, new_importance: float) -> bool:
        """
        更新记忆条目的重要性分数

        Args:
            entry_id: 条目 ID
            new_importance: 新的重要性分数

        Returns:
            是否更新成功
        """
        try:
            # 获取条目
            results = self.vector_store.get(ids=[entry_id])
            if not results:
                return False

            # 更新元数据
            metadata = results[0].get("metadata", {})
            metadata["importance"] = new_importance

            self.vector_store.update(
                ids=[entry_id],
                metadatas=[metadata]
            )

            # 更新内存映射
            if entry_id in self._entries:
                self._entries[entry_id].importance = new_importance
                self._entries[entry_id].metadata["importance"] = new_importance

            self.logger.info(f"Updated importance of {entry_id[:8]}... to {new_importance:.2f}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update importance: {e}")
            return False