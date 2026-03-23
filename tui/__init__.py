"""Milo Agent TUI Module

基于 Textual 的现代化终端交互界面，Claude Code 风格设计。

特性:
- 多轮对话（带历史记录）
- 真实流式输出
- 工具调用可视化
- 思考过程折叠
- 代码语法高亮
- 会话管理
- 主题切换
- 快捷键支持

启动方式:
    python -m tui.main
    python -m tui.main --rag --react
"""

__version__ = "0.2.0"

from tui.app import MiloApp, run_tui
from tui.config import config

__all__ = ["MiloApp", "run_tui", "config"]
