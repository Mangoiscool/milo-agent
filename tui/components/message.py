"""Message Components - Modern Chat UI"""

from datetime import datetime
from typing import Optional

from rich import box
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.console import Group
from rich.rule import Rule

from textual.widgets import Static
from textual.reactive import reactive

from tui.utils.formatters import format_message_content, format_timestamp, estimate_tokens


class MessageBubble(Static):
    """消息气泡组件"""

    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    content = reactive("")
    timestamp = reactive(None)
    tokens = reactive(0)
    is_streaming = reactive(False)

    def __init__(
        self,
        role: str,
        content: str = "",
        timestamp: Optional[datetime] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.tokens = estimate_tokens(content)
        self.is_streaming = False

    def render(self):
        is_user = self.role == "user"

        # 构建头部
        header_parts = []
        if self.timestamp:
            header_parts.append((f"{format_timestamp(self.timestamp)} ", "dim"))

        if is_user:
            header_parts.append(("You", "bold blue"))
        else:
            header_parts.append(("Milo", "bold orange"))

        if self.tokens > 0 and not self.is_streaming:
            header_parts.append((f" · {self.tokens} tokens", "dim"))

        header_text = Text.assemble(*header_parts)

        # 内容区域
        if self.is_streaming:
            # 流式输出中显示光标
            display_content = self.content + "▌"
        else:
            display_content = self.content

        # 使用 Panel 包装
        if is_user:
            # 用户消息 - 右对齐风格
            panel = Panel(
                Text(display_content, style="blue") if not self.is_streaming else Text(display_content),
                title=Align.right(header_text),
                title_align="right",
                border_style="blue dim",
                box=box.ROUNDED,
            )
        else:
            # 助手消息 - 左对齐风格
            panel = Panel(
                format_message_content(display_content) if not self.is_streaming else Text(display_content),
                title=header_text,
                title_align="left",
                border_style="orange dim" if not self.is_streaming else "orange",
                box=box.ROUNDED,
            )

        return panel

    def update_content(self, content: str, streaming: bool = False):
        """更新内容"""
        self.content = content
        self.is_streaming = streaming
        self.tokens = estimate_tokens(content)
        self.refresh()

    def finalize(self):
        """完成流式输出"""
        self.is_streaming = False
        self.refresh()


class StreamingMessage(Static):
    """流式消息组件 - 支持逐字输出"""

    DEFAULT_CSS = """
    StreamingMessage {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content = ""
        self.timestamp = datetime.now()

    def render(self):
        header = Text.assemble(
            (format_timestamp(self.timestamp), "dim"),
            (" · ", "dim"),
            ("Milo", "bold orange"),
        )

        # 显示内容 + 闪烁光标
        content_text = Text(self.content + "▌", style="orange")

        return Panel(
            content_text,
            title=header,
            title_align="left",
            border_style="orange",
        )

    def append_text(self, text: str):
        """追加文本"""
        self.content += text
        self.refresh()

    def finalize(self, full_content: str):
        """完成流式输出"""
        self.content = full_content
        self.refresh()


class ToolCallCard(Static):
    """工具调用卡片"""

    DEFAULT_CSS = """
    ToolCallCard {
        width: 95%;
        height: auto;
        margin: 0 2;
    }
    """

    is_running = reactive(True)
    result = reactive(None)

    def __init__(self, name: str, args: dict, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = name
        self.args = args
        self.result = None
        self.is_running = True
        self.start_time = datetime.now()

    def render(self):
        from rich import box

        # 状态图标
        if self.is_running:
            icon = "◉"
            status_color = "yellow"
            status_text = "running"
        else:
            icon = "✓"
            status_color = "green"
            status_text = "done"

        # 构建标题
        title = Text.assemble(
            (icon, status_color),
            (" ", "default"),
            (self.tool_name, f"bold {status_color}"),
            (f" · {status_text}", "dim"),
        )

        # 构建内容
        import json
        lines = []

        # 参数
        args_str = json.dumps(self.args, ensure_ascii=False, indent=2)
        lines.append(Text("Arguments:", style="dim"))
        lines.append(Text(args_str, style="yellow"))

        # 结果
        if self.result is not None:
            lines.append(Text(""))
            lines.append(Text("Result:", style="dim"))
            result_str = str(self.result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            lines.append(Text(result_str, style="green"))

        content = Group(*lines)

        return Panel(
            content,
            title=title,
            title_align="left",
            border_style=status_color,
            box=box.SIMPLE_HEAVY,
        )

    def set_result(self, result):
        """设置结果"""
        self.result = result
        self.is_running = False
        self.refresh()


class ThinkingCard(Static):
    """思考过程卡片"""

    DEFAULT_CSS = """
    ThinkingCard {
        width: 95%;
        height: auto;
        margin: 0 2;
    }
    """

    content = reactive("")

    def __init__(self, content: str = "", **kwargs):
        super().__init__(**kwargs)
        self.thinking_content = content

    def render(self):
        from rich import box

        title = Text.assemble(
            ("◉ ", "dim"),
            ("thinking", "dim italic"),
        )

        return Panel(
            Text(self.thinking_content, style="dim italic"),
            title=title,
            title_align="left",
            border_style="dim",
            box=box.SIMPLE,
        )

    def update(self, content: str):
        """更新内容"""
        self.thinking_content = content
        self.refresh()


class SystemMessage(Static):
    """系统消息"""

    DEFAULT_CSS = """
    SystemMessage {
        width: 100%;
        height: auto;
        padding: 0 1;
        content-align: center middle;
    }
    """

    def __init__(self, content: str, level: str = "info", **kwargs):
        super().__init__(**kwargs)
        self.content = content
        self.level = level  # info, warning, error, success

    def render(self):
        color_map = {
            "info": "blue",
            "warning": "yellow",
            "error": "red",
            "success": "green",
        }
        color = color_map.get(self.level, "white")

        return Align.center(Text(f"◉ {self.content}", style=f"dim {color}"))


class MessageGroup:
    """消息组 - 用于管理相关的消息"""

    def __init__(self, container):
        self.container = container
        self.user_msg = None
        self.assistant_msg = None
        self.tool_calls = []
        self.thinking = None

    def add_user(self, content: str) -> MessageBubble:
        """添加用户消息"""
        self.user_msg = MessageBubble(role="user", content=content)
        self.container.mount(self.user_msg)
        return self.user_msg

    def start_assistant(self) -> StreamingMessage:
        """开始助手回复"""
        self.assistant_msg = StreamingMessage()
        self.container.mount(self.assistant_msg)
        return self.assistant_msg

    def finalize_assistant(self, content: str) -> MessageBubble:
        """完成助手回复"""
        if self.assistant_msg:
            self.assistant_msg.remove()

        self.assistant_msg = MessageBubble(role="assistant", content=content)
        self.container.mount(self.assistant_msg)
        return self.assistant_msg

    def add_tool_call(self, name: str, args: dict) -> ToolCallCard:
        """添加工具调用"""
        tool = ToolCallCard(name=name, args=args)
        self.tool_calls.append(tool)
        self.container.mount(tool)
        return tool

    def add_thinking(self, content: str) -> ThinkingCard:
        """添加思考过程"""
        self.thinking = ThinkingCard(content=content)
        self.container.mount(self.thinking)
        return self.thinking
