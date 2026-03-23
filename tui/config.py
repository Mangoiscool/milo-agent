"""TUI Configuration Manager"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List


@dataclass
class TUIConfig:
    """TUI 配置"""
    # 外观
    theme: str = "claude"
    show_timestamps: bool = True
    show_token_count: bool = True
    code_wrap: bool = False
    message_spacing: int = 1

    # 行为
    auto_save_chat: bool = True
    max_history: int = 100
    stream_delay: float = 0.01
    enter_to_send: bool = True

    # 默认能力
    default_rag: bool = False
    default_browser: bool = False
    default_react: bool = False

    # 快捷键
    keybindings: dict = field(default_factory=lambda: {
        "quit": "ctrl+q",
        "clear": "ctrl+l",
        "new_chat": "ctrl+n",
        "settings": "ctrl+,",
        "focus_input": "ctrl+i",
        "toggle_sidebar": "ctrl+s",
    })

    # 会话历史
    last_session_id: Optional[str] = None
    recent_sessions: List[str] = field(default_factory=list)


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self.config_dir = Path.home() / ".config" / "milo-agent"
        self.config_file = self.config_dir / "tui-config.json"
        self.config = TUIConfig()
        self.load()

    def load(self) -> TUIConfig:
        """加载配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 合并配置
                    for key, value in data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
            except Exception:
                pass
        return self.config

    def save(self):
        """保存配置"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)

    def get(self, key: str, default=None):
        """获取配置项"""
        return getattr(self.config, key, default)

    def set(self, key: str, value):
        """设置配置项"""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save()

    def reset(self):
        """重置配置"""
        self.config = TUIConfig()
        self.save()


# 全局配置实例
config = ConfigManager()
