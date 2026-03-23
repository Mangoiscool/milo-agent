"""
短期记忆实现
基于列表的内存管理，支持消息数量限制、修剪和可选的持久化
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.llm.base import Message, Role
from core.memory.base import BaseMemory


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
    try:
        from config.settings import settings
        s = settings()
        project_root = _get_project_root()

        if s.workspace_dir:
            if not s.workspace_dir.is_absolute():
                return project_root / s.workspace_dir
            return s.workspace_dir
    except ImportError:
        pass

    return Path.home() / ".milo-agent" / "workspace"


class ShortTermMemory(BaseMemory):
    """
    支持自动修剪的短期记忆，可选持久化到文件

    特性：
    - 基于列表的存储
    - 可配置的消息数量限制
    - 智能修剪：保留 SYSTEM 消息 + 最近的消息
    - 基于评分的智能修剪（可选）
    - 可选的持久化：保存到文件系统，支持会话隔离

    修剪策略：
    1. 简单策略（默认）：删除最旧的非 SYSTEM 消息
       - 超出限制时，删除最旧的非 SYSTEM 消息
       - 保留所有 SYSTEM 消息
       - 优先保留最近的消息
    2. 智能策略（可选）：根据重要性评分
       - 始终保留 SYSTEM 消息
       - 根据角色、长度、时效性和关键词对其他消息评分
       - 保留得分最高的消息

    使用示例：
        # 纯内存模式（默认）
        memory = ShortTermMemory(max_messages=50)

        # 启用持久化
        memory = ShortTermMemory(
            max_messages=50,
            persist=True,
            session_id="my-session"  # 可选，默认自动生成
        )
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
        use_intelligent_pruning: bool = False,
        persist: bool = False,
        session_id: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        初始化短期记忆

        参数：
            max_messages: 最大存储消息数量（默认：50）
            use_intelligent_pruning: 启用基于评分的修剪（默认：False）
            persist: 启用持久化到文件（默认：False）
            session_id: 会话 ID，None 则自动生成（仅在 persist=True 时有效）
            auto_save: 每次 add 后自动保存（仅在 persist=True 时有效）
        """
        super().__init__()
        self.max_messages = max_messages
        self.use_intelligent_pruning = use_intelligent_pruning
        self._messages: List[Message] = []

        # 持久化相关配置
        self.persist = persist
        self.auto_save = auto_save
        self.session_id = session_id
        self._storage_path: Optional[Path] = None

        if persist:
            self._init_persistence(session_id)

    def add(self, message: Message) -> None:
        """
        添加消息到记忆

        如果超出限制，会自动触发修剪。
        如果启用了持久化且 auto_save=True，会自动保存。

        参数：
            message: 要添加的消息
        """
        self._messages.append(message)
        self.logger.debug(f"添加消息: {message.role.value} - {message.content[:50]}...")

        # 如需要则修剪
        if self.count() > self.max_messages:
            self._prune()

        # 如启用持久化则自动保存
        if self.persist and self.auto_save:
            self.save()

    def get_all(self) -> List[Message]:
        """
        获取所有消息

        返回：
            所有消息列表的副本
        """
        return self._messages.copy()

    def get_recent(self, n: int) -> List[Message]:
        """
        获取最近的 n 条消息

        参数：
            n: 要获取的消息数量

        返回：
            最近消息列表（如果消息不足可能少于 n 条）
        """
        if n <= 0:
            return []
        return self._messages[-n:]

    def count(self) -> int:
        """
        获取消息总数

        返回：
            当前内存中的消息数量
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

        参数：
            message: 消息对象
            position: 消息在历史中的位置（从0开始）
            total: 消息总数

        返回：
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

        参数：
            messages: 消息列表

        返回：
            每条消息的得分列表
        """
        total = len(messages)
        return [self._score_message(msg, i, total) for i, msg in enumerate(messages)]

    def _prune(self) -> None:
        """
        当超出限制时修剪消息

        策略：
        1. 简单策略：删除最旧的非 SYSTEM 消息
           - 保留所有 SYSTEM 消息
           - 保留最近的非 SYSTEM 消息
        2. 智能策略：根据重要性评分
           - 始终保留 SYSTEM 消息
           - 对其他消息评分并保留得分最高的
        """
        # 按角色分离消息
        system_messages = [m for m in self._messages if m.role == Role.SYSTEM]
        other_messages = [m for m in self._messages if m.role != Role.SYSTEM]

        # 计算可以保留多少条非 SYSTEM 消息
        available_slots = self.max_messages - len(system_messages)

        if available_slots < 0:
            # 边界情况：SYSTEM 消息过多
            self.logger.warning(
                f"SYSTEM 消息过多 ({len(system_messages)})，"
                f"超出限制 ({self.max_messages})"
            )
            # 仍然保留所有 SYSTEM 消息
            self._messages = system_messages
            return

        # 选择修剪策略
        if self.use_intelligent_pruning:
            # 智能修剪：评分并保留最佳消息
            scores = self._score_messages(other_messages)

            # 将消息与评分配对并按评分排序
            scored_messages = list(zip(other_messages, scores))
            scored_messages.sort(key=lambda x: x[1], reverse=True)

            # 保留得分最高的消息
            kept_messages = [msg for msg, _ in scored_messages[:available_slots]]
            kept_ids = {id(msg) for msg in kept_messages}  # 使用 id() 创建标识符集合

            # 重建列表，保持保留消息的原始顺序
            result = [m for m in self._messages if id(m) in kept_ids or m.role == Role.SYSTEM]

            removed_count = len(self._messages) - len(result)
            self._messages = result

            self.logger.info(
                f"使用智能策略修剪了 {removed_count} 条消息，"
                f"保留了 {len(system_messages)} 条 SYSTEM + {len(kept_messages)} 条高分消息"
            )
        else:
            # 简单修剪：保留最近的消息
            kept_other = other_messages[-available_slots:] if available_slots > 0 else []

            # 重建消息列表，保持顺序
            result = [m for m in self._messages if m.role == Role.SYSTEM] + kept_other

            removed_count = len(self._messages) - len(result)
            self._messages = result

            self.logger.info(
                f"使用简单策略修剪了 {removed_count} 条消息，"
                f"保留了 {len(system_messages)} 条 SYSTEM + {len(kept_other)} 条最近消息"
            )

    # ==================== 持久化方法 ====================

    def _init_persistence(self, session_id: Optional[str] = None) -> None:
        """初始化持久化配置"""
        base_dir = _resolve_workspace_dir()
        storage_dir = base_dir / "memory_storage" / "sessions"
        storage_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or str(uuid.uuid4())
        self._storage_path = storage_dir / f"{self.session_id}.json"

        # 尝试加载已有会话
        loaded = self.load()
        if loaded > 0:
            self.logger.info(f"已加载会话 {self.session_id[:8]}... ({loaded} 条消息)")
        else:
            self.logger.info(f"创建新会话 {self.session_id[:8]}...")

    def save(self) -> None:
        """
        保存当前会话到文件

        仅在 persist=True 时有效。
        """
        if not self.persist or not self._storage_path:
            return

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

        self.logger.debug(f"会话 {self.session_id[:8]}... 已保存 ({len(messages_data)} 条消息)")

    def load(self) -> int:
        """
        从文件加载会话

        仅在 persist=True 时有效。

        返回：
            加载的消息数量（0 表示文件不存在或加载失败）
        """
        if not self.persist or not self._storage_path or not self._storage_path.exists():
            return 0

        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 恢复消息
            messages_data = data.get("messages", [])
            messages = [Message.from_api_format(m) for m in messages_data]

            # 清空当前内存并添加（会触发修剪）
            self._messages.clear()
            for msg in messages:
                super().add(msg)  # 使用父类 add 避免触发保存

            return len(messages)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(f"加载会话失败 {self.session_id[:8]}...: {e}")
            return 0

    def switch_session(self, new_session_id: str, auto_save_current: bool = True) -> int:
        """
        切换到另一个会话

        仅在 persist=True 时有效。

        参数：
            new_session_id: 要切换到的会话 ID
            auto_save_current: 是否先保存当前会话

        返回：
            新会话加载的消息数量
        """
        if not self.persist:
            self.logger.warning("未启用持久化，无法切换会话")
            return 0

        if auto_save_current:
            self.save()

        # 清空当前内存
        self._messages.clear()

        # 更新会话 ID 和路径
        self.session_id = new_session_id
        base_dir = _resolve_workspace_dir()
        self._storage_path = base_dir / "memory_storage" / "sessions" / f"{new_session_id}.json"

        # 加载新会话
        loaded = self.load()
        self.logger.info(f"切换到会话 {new_session_id[:8]}... ({loaded} 条消息)")
        return loaded

    def clear(self, delete_file: bool = False) -> None:
        """
        清除所有消息

        参数：
            delete_file: 是否同时删除持久化文件（仅在 persist=True 时有效）
        """
        self._messages.clear()

        if self.persist and delete_file and self._storage_path and self._storage_path.exists():
            self._storage_path.unlink()
            self.logger.info(f"会话文件已删除: {self._storage_path}")
        else:
            self.logger.info("记忆已清除")

    def delete_session(self, session_id: Optional[str] = None) -> bool:
        """
        删除指定会话（或当前会话）

        仅在 persist=True 时有效。

        参数：
            session_id: 要删除的会话 ID，None 表示删除当前会话

        返回：
            是否成功删除
        """
        if not self.persist:
            return False

        target_id = session_id or self.session_id
        base_dir = _resolve_workspace_dir()
        target_path = base_dir / "memory_storage" / "sessions" / f"{target_id}.json"

        if target_path.exists():
            target_path.unlink()
            self.logger.info(f"已删除会话: {target_id[:8]}...")

            # 如果删除的是当前会话，清空内存
            if target_id == self.session_id:
                self._messages.clear()

            return True
        return False

    @classmethod
    def list_sessions(cls) -> List[Dict]:
        """
        列出所有可用的会话

        返回：
            会话信息列表
        """
        base_dir = _resolve_workspace_dir()
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
    def delete_all_sessions(cls) -> int:
        """
        删除所有会话

        返回：
            删除的会话数量
        """
        base_dir = _resolve_workspace_dir()
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
            "persist": self.persist,
            "storage_path": str(self._storage_path) if self._storage_path else None,
            "message_count": self.count(),
            "max_messages": self.max_messages,
            "use_intelligent_pruning": self.use_intelligent_pruning,
            "auto_save": self.auto_save
        }

    def __repr__(self) -> str:
        if self.persist:
            return (
                f"<ShortTermMemory "
                f"session={self.session_id[:8]}... "
                f"messages={self.count()}/{self.max_messages} "
                f"persist=True>"
            )
        return f"<ShortTermMemory messages={self.count()}/{self.max_messages}>"
