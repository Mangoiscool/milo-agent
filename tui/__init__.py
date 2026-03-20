"""Milo Agent TUI - Terminal User Interface

基于 Textual 的终端交互界面，支持：
- 多轮对话（带历史记录）
- 实时流式输出
- 工具调用可视化
- 思考过程折叠
- 快捷键支持

启动方式：
    python -m tui.main
    python -m cli.main tui  # 如果集成到 cli
"""

__version__ = "0.1.0"

from tui.app import MiloTUIApp

__all__ = ["MiloTUIApp"]
