"""
Persistent memory implementation with session isolation
支持会话隔离的持久化记忆实现
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from core.llm.base import Message
from core.memory.base import BaseMemory
from core.memory.short_term import ShortTermMemory
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


class PersistentMemory(ShortTermMemory):
    """
    支持会话隔离的持久化短期记忆

    特性：
    - 会话隔离：每个会话有独立的存储文件
    - 容量上限：支持 max_messages 限制和自动修剪
    - 持久化：自动保存到文件系统
    - 会话恢复：可以加载历史会话继续对话

    存储结构：
    workspace/
      memory_storage/
        sessions/
          {session_id}.json     # 单个会话的记忆
          {session_id}.json     # 另一个会话

    使用示例：
        # 创建新会话
        memory = PersistentMemory(session_id=None)  # 自动生成 session_id

        # 恢复已有会话
        memory = PersistentMemory(session_id="abc-123")

        # 添加消息（自动保存）
        memory.add(Message(role=Role.USER, content="你好"))

        # 切换会话
        memory.switch_session("new-session-id")

        # 列出所有会话
        sessions = PersistentMemory.list_sessions()
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_messages: int = 50,
        use_intelligent_pruning: bool = False,
        auto_save: bool = True
    ):
        """
        初始化持久化记忆

        Args:
            session_id: 会话 ID，None 则自动生成新的 UUID
            max_messages: 单个会话的最大消息数（默认 50）
            use_intelligent_pruning: 启用智能修剪（基于重要性评分）
            auto_save: 每次 add 后自动保存
        """
        # 先调用父类初始化（此时 _messages 为空）
        super().__init__(max_messages, use_intelligent_pruning)

        # 确定存储目录
        base_dir = _resolve_workspace_dir()
        self.storage_dir = base_dir / "memory_storage" / "sessions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 会话 ID
        self.session_id = session_id or str(uuid.uuid4())
        self._storage_path = self._get_storage_path(self.session_id)

        self.auto_save = auto_save

        # 尝试加载已有会话
        loaded = self.load()
        if loaded > 0:
            self.logger.info(f"Loaded session {self.session_id[:8]}... with {loaded} messages")
        else:
            self.logger.info(f"Created new session {self.session_id[:8]}...")

    def _get_storage_path(self, session_id: str) -> Path:
        """获取会话的存储路径"""
        return self.storage_dir / f"{session_id}.json"

    @property
    def storage_path(self) -> Path:
        """当前会话的存储路径"""
        return self._storage_path

    def save(self) -> None:
        """
        保存当前会话到文件

        存储格式：
        {
            "session_id": "uuid",
            "created_at": "isoformat",
            "updated_at": "isoformat",
            "message_count": N,
            "messages": [...]  # API 格式的消息列表
        }
        """
        from datetime import datetime

        messages_data = [msg.to_api_format() for msg in self.get_all()]

        data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": len(messages_data),
            "max_messages": self.max_messages,
            "messages": messages_data
        }

        with open(self._storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.debug(f"Session {self.session_id[:8]}... saved ({len(messages_data)} messages)")

    def load(self) -> int:
        """
        从文件加载会话

        Returns:
            加载的消息数量（0 表示文件不存在或加载失败）
        """
        if not self._storage_path.exists():
            return 0

        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 验证 session_id 匹配
            stored_session_id = data.get("session_id")
            if stored_session_id and stored_session_id != self.session_id:
                self.logger.warning(
                    f"Session ID mismatch: expected {self.session_id[:8]}..., "
                    f"found {stored_session_id[:8]}..."
                )

            # 恢复消息
            messages_data = data.get("messages", [])
            messages = [Message.from_api_format(m) for m in messages_data]

            # 清空当前内存并添加（会触发修剪）
            self._messages.clear()
            for msg in messages:
                # 绕过自动保存，避免重复写入
                super().add(msg)

            return len(messages)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(f"Failed to load session {self.session_id[:8]}...: {e}")
            return 0

    def add(self, message: Message) -> None:
        """
        添加消息到当前会话

        自动触发保存（如果 auto_save=True）
        """
        super().add(message)  # 父类会处理修剪

        if self.auto_save:
            self.save()

    def switch_session(self, new_session_id: str, auto_save_current: bool = True) -> int:
        """
        切换到另一个会话

        Args:
            new_session_id: 要切换到的会话 ID
            auto_save_current: 是否先保存当前会话

        Returns:
            新会话加载的消息数量
        """
        if auto_save_current:
            self.save()

        # 清空当前内存
        self._messages.clear()

        # 更新会话 ID 和路径
        self.session_id = new_session_id
        self._storage_path = self._get_storage_path(new_session_id)

        # 加载新会话
        loaded = self.load()
        self.logger.info(f"Switched to session {new_session_id[:8]}... ({loaded} messages)")

        return loaded

    def clear(self) -> None:
        """清空当前会话（内存 + 文件）"""
        super().clear()

        if self._storage_path.exists():
            self._storage_path.unlink()
            self.logger.info(f"Session file deleted: {self._storage_path}")

    def delete_session(self, session_id: Optional[str] = None) -> bool:
        """
        删除指定会话（或当前会话）

        Args:
            session_id: 要删除的会话 ID，None 表示删除当前会话

        Returns:
            是否成功删除
        """
        target_id = session_id or self.session_id
        target_path = self._get_storage_path(target_id)

        if target_path.exists():
            target_path.unlink()
            self.logger.info(f"Deleted session: {target_id[:8]}...")

            # 如果删除的是当前会话，清空内存
            if target_id == self.session_id:
                self._messages.clear()

            return True
        return False

    @classmethod
    def list_sessions(cls, storage_dir: Optional[Path] = None) -> List[Dict]:
        """
        列出所有可用的会话

        Args:
            storage_dir: 存储目录，None 则使用默认路径

        Returns:
            会话信息列表，每个会话包含：
            - session_id: 会话 ID
            - created_at: 创建时间
            - updated_at: 更新时间
            - message_count: 消息数量
            - max_messages: 容量上限
        """
        base_dir = storage_dir or _resolve_workspace_dir()
        sessions_dir = base_dir / "memory_storage" / "sessions"

        if not sessions_dir.exists():
            return []

        sessions = []
        for file_path in sessions_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                sessions.append({
                    "session_id": data.get("session_id", file_path.stem),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": data.get("message_count", 0),
                    "max_messages": data.get("max_messages", 50)
                })
            except (json.JSONDecodeError, IOError):
                continue

        # 按更新时间排序（最新的在前）
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

    @classmethod
    def delete_all_sessions(cls, storage_dir: Optional[Path] = None) -> int:
        """
        删除所有会话

        Args:
            storage_dir: 存储目录，None 则使用默认路径

        Returns:
            删除的会话数量
        """
        base_dir = storage_dir or _resolve_workspace_dir()
        sessions_dir = base_dir / "memory_storage" / "sessions"

        if not sessions_dir.exists():
            return 0

        count = 0
        for file_path in sessions_dir.glob("*.json"):
            try:
                file_path.unlink()
                count += 1
            except IOError:
                pass

        return count

    def get_session_info(self) -> Dict:
        """获取当前会话信息"""
        return {
            "session_id": self.session_id,
            "storage_path": str(self._storage_path),
            "message_count": self.count(),
            "max_messages": self.max_messages,
            "use_intelligent_pruning": self.use_intelligent_pruning,
            "auto_save": self.auto_save
        }

    def __repr__(self) -> str:
        return (
            f"<PersistentMemory "
            f"session={self.session_id[:8]}... "
            f"messages={self.count()}/{self.max_messages} "
            f"path={self._storage_path}>"
        )
