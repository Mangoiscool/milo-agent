"""Modern TUI Widgets - Claude Code style"""

from textual.widgets import Static, Input, Button
from textual.reactive import reactive
from textual.color import Color
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Group
from rich.align import Align
import json


class MessageItem(Static):
    """单条消息组件 - 简洁风格"""

    DEFAULT_CSS = """
    MessageItem {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 0;
    }
    """

    def __init__(self, role: str, content: str = "", **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def render(self):
        if self.role == "user":
            # 用户消息 - 简洁右对齐
            return Panel(
                Text(self.content, style="bold blue"),
                border_style="blue",
                title="▸ You",
                title_align="right",
                padding=(0, 1)
            )
        else:
            # 助手消息 - 左对齐
            return Panel(
                self.content or "▸",
                border_style="green",
                title="◉ Milo",
                title_align="left",
                padding=(0, 1)
            )

    def update_content(self, content: str):
        """更新内容"""
        self.content = content
        self.refresh()


class ToolCallItem(Static):
    """工具调用项 - 可折叠的简洁展示"""

    DEFAULT_CSS = """
    ToolCallItem {
        width: 100%;
        height: auto;
        padding: 0 2;
        margin: 0;
    }
    """

    def __init__(self, name: str, args: dict, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = name
        self.args = args
        self.result = None
        self.is_running = True

    def render(self):
        icon = "◌" if self.is_running else "✓"
        color = "yellow" if self.is_running else "green"

        # 简洁的单行展示
        header = Text(f"{icon} {self.tool_name}", style=f"dim {color}")

        if self.result:
            # 有结果时显示结果摘要
            result_preview = str(self.result)[:60] + "..." if len(str(self.result)) > 60 else str(self.result)
            content = f"Args: {json.dumps(self.args, ensure_ascii=False)[:50]}\nResult: {result_preview}"
            return Panel(
                content,
                border_style=color,
                title=header,
                title_align="left",
                padding=(0, 1)
            )
        else:
            # 运行中
            return Panel(
                f"Args: {json.dumps(self.args, ensure_ascii=False)[:60]}...",
                border_style=color,
                title=header,
                title_align="left",
                padding=(0, 1)
            )

    def set_result(self, result: str):
        """设置结果"""
        self.result = result
        self.is_running = False
        self.refresh()


class ThinkingItem(Static):
    """思考过程项"""

    DEFAULT_CSS = """
    ThinkingItem {
        width: 100%;
        height: auto;
        padding: 0 2;
        margin: 0;
    }
    """

    def __init__(self, content: str, **kwargs):
        super().__init__(**kwargs)
        self.thinking_content = content

    def render(self):
        return Panel(
            self.thinking_content,
            border_style="dim",
            title="◉ thinking",
            title_align="left",
            padding=(0, 1)
        )


class ChatContainer(Static):
    """聊天容器 - 管理所有消息"""

    DEFAULT_CSS = """
    ChatContainer {
        width: 100%;
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = []

    def add_user_message(self, content: str) -> MessageItem:
        """添加用户消息"""
        msg = MessageItem(role="user", content=content)
        self.messages.append(msg)
        self.mount(msg)
        msg.scroll_visible(animate=False)
        return msg

    def add_assistant_message(self, content: str = "") -> MessageItem:
        """添加助手消息"""
        msg = MessageItem(role="assistant", content=content)
        self.messages.append(msg)
        self.mount(msg)
        msg.scroll_visible(animate=False)
        return msg

    def add_tool_call(self, name: str, args: dict) -> ToolCallItem:
        """添加工具调用"""
        tool = ToolCallItem(name=name, args=args)
        self.messages.append(tool)
        self.mount(tool)
        tool.scroll_visible(animate=False)
        return tool

    def add_thinking(self, content: str) -> ThinkingItem:
        """添加思考过程"""
        thinking = ThinkingItem(content=content)
        self.messages.append(thinking)
        self.mount(thinking)
        thinking.scroll_visible(animate=False)
        return thinking

    def clear_all(self):
        """清空所有消息"""
        for msg in self.messages:
            msg.remove()
        self.messages.clear()


class PromptInput(Static):
    """提示输入框 - 类似 Claude Code 的风格"""

    DEFAULT_CSS = """
    PromptInput {
        width: 100%;
        height: auto;
        min-height: 3;
        dock: bottom;
        padding: 0;
    }

    PromptInput #input_container {
        width: 100%;
        height: auto;
        min-height: 3;
        background: $surface-darken-1;
    }

    PromptInput #prompt_symbol {
        width: 3;
        content-align: center middle;
        color: $primary;
    }

    PromptInput Input {
        width: 1fr;
        border: none;
        background: transparent;
    }

    PromptInput Button {
        width: 8;
        display: none;
    }

    PromptInput.loading Button {
        display: block;
    }
    """

    is_loading = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_loading = False

    def compose(self):
        from textual.containers import Horizontal
        with Horizontal(id="input_container"):
            yield Static(">", id="prompt_symbol")
            yield Input(placeholder="Message Milo...", id="chat_input")
            yield Button("Stop", id="stop_btn", variant="error")

    def watch_is_loading(self, loading: bool):
        """监听加载状态"""
        # 确保组件已挂载
        if not self.is_mounted:
            return
        try:
            btn = self.query_one("#stop_btn", Button)
            inp = self.query_one("#chat_input", Input)
            if loading:
                btn.styles.display = "block"
                inp.disabled = True
            else:
                btn.styles.display = "none"
                inp.disabled = False
                inp.focus()
        except Exception:
            # 组件可能还未完全挂载
            pass

    def set_loading(self, loading: bool):
        """设置加载状态"""
        self.is_loading = loading

    def clear(self):
        """清空输入"""
        self.query_one("#chat_input", Input).value = ""
        self.query_one("#chat_input", Input).focus()

    def get_value(self) -> str:
        """获取输入值"""
        return self.query_one("#chat_input", Input).value


class StatusFooter(Static):
    """状态栏"""

    DEFAULT_CSS = """
    StatusFooter {
        dock: bottom;
        height: 1;
        background: $surface-darken-1;
        color: $text-muted;
        content-align: left middle;
        padding: 0 1;
    }
    """

    provider = reactive("ollama")
    model = reactive("default")
    capabilities = reactive([])

    def render(self):
        caps = " ".join(f"[{c}]" for c in self.capabilities) if self.capabilities else ""
        return f"◉ {self.provider}/{self.model} {caps}"

    def update_status(self, provider: str, model: str, capabilities: list = None):
        """更新状态"""
        self.provider = provider
        self.model = model
        self.capabilities = capabilities or []
