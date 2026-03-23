"""Status Bar Component"""

from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text


class StatusBar(Static):
    """底部状态栏"""

    DEFAULT_CSS = """
    StatusBar {
        width: 100%;
        height: 1;
        dock: bottom;
        background: $surface-darken-1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    # 响应式状态
    provider = reactive("ollama")
    model = reactive("default")
    capabilities = reactive(list)
    is_connected = reactive(True)
    message_count = reactive(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capabilities = []

    def render(self):
        """渲染状态栏"""
        # 左侧: 连接状态 + Provider/Model
        left_parts = []

        # 连接状态
        if self.is_connected:
            left_parts.append(("● ", "green"))
        else:
            left_parts.append(("○ ", "red"))

        # Provider/Model
        left_parts.append((f"{self.provider}/{self.model}", "bold"))

        # 能力指示器
        if self.capabilities:
            caps_text = " ".join(f"[{c}]" for c in self.capabilities)
            left_parts.append((f" {caps_text}", "dim"))

        left_text = Text.assemble(*left_parts)

        # 右侧: 消息计数 + 帮助提示
        right_text = Text.assemble(
            (f"{self.message_count} msgs  ", "dim"),
            ("Ctrl+Q quit", "dim"),
        )

        # 组合
        return Text.assemble(
            left_text,
            Text(" ").join([Text(" ")] * 100),  # 占位
            right_text,
        )

    def update_provider(self, provider: str, model: str = "default"):
        """更新 provider"""
        self.provider = provider
        self.model = model

    def update_capabilities(self, capabilities: list):
        """更新能力列表"""
        self.capabilities = capabilities

    def update_connection(self, connected: bool):
        """更新连接状态"""
        self.is_connected = connected

    def update_message_count(self, count: int):
        """更新消息计数"""
        self.message_count = count

    def toggle_capability(self, name: str, enabled: bool):
        """切换能力状态"""
        if enabled and name not in self.capabilities:
            self.capabilities = self.capabilities + [name]
        elif not enabled and name in self.capabilities:
            self.capabilities = [c for c in self.capabilities if c != name]
