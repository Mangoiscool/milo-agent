"""Input Component - Modern Chat Input"""

from textual.widgets import Input, Button, Static
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual import events
from rich.text import Text


class ChatInput(Static):
    """聊天输入组件"""

    DEFAULT_CSS = """
    ChatInput {
        width: 100%;
        height: auto;
        min-height: 3;
        max-height: 10;
        dock: bottom;
        background: $surface;
        border-top: solid $primary 50%;
    }

    ChatInput #input_container {
        width: 100%;
        height: auto;
        min-height: 3;
        padding: 0 1;
    }

    ChatInput #prompt_symbol {
        width: 3;
        content-align: center middle;
        color: $primary;
    }

    ChatInput Input {
        width: 1fr;
        height: auto;
        min-height: 1;
        max-height: 8;
        border: none;
        background: transparent;
        padding: 1 2;
    }

    ChatInput Input:focus {
        border: none;
    }

    ChatInput Button {
        width: auto;
        min-width: 8;
        display: none;
    }

    ChatInput.loading Button {
        display: block;
    }

    ChatInput Button#send_btn {
        display: block;
    }

    ChatInput #hint_bar {
        width: 100%;
        height: 1;
        padding: 0 2;
        color: $text-muted;
        text-style: dim;
    }
    """

    is_loading = reactive(False)
    multiline = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_loading = False

    def compose(self):
        """组合界面"""
        with Vertical():
            with Horizontal(id="input_container"):
                yield Static(">", id="prompt_symbol")
                yield Input(
                    placeholder="Message Milo... (Enter to send, Shift+Enter for new line)",
                    id="chat_input",
                    multiline=False,
                )
                yield Button("Stop", id="stop_btn", variant="error")

            yield Static(
                "Enter to send · Ctrl+L clear · Ctrl+N new chat · Ctrl+, settings",
                id="hint_bar"
            )

    def on_mount(self):
        """挂载后"""
        self.query_one("#chat_input", Input).focus()

    def watch_is_loading(self, loading: bool):
        """监听加载状态"""
        if not self.is_mounted:
            return

        try:
            stop_btn = self.query_one("#stop_btn", Button)
            inp = self.query_one("#chat_input", Input)
            hint = self.query_one("#hint_bar", Static)

            if loading:
                stop_btn.styles.display = "block"
                inp.disabled = True
                hint.update("Processing...")
            else:
                stop_btn.styles.display = "none"
                inp.disabled = False
                inp.focus()
                hint.update("Enter to send · Ctrl+L clear · Ctrl+N new chat · Ctrl+, settings")
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed):
        """输入变化"""
        # 自动调整高度
        lines = event.value.count('\n') + 1
        if lines > 1 and not self.multiline:
            self.multiline = True
            self.query_one("#chat_input", Input).multiline = True
        elif lines == 1 and self.multiline:
            self.multiline = False
            self.query_one("#chat_input", Input).multiline = False

    def on_key(self, event: events.Key):
        """按键处理"""
        if event.key == "enter" and not event.shift:
            # 发送消息
            if not self.multiline:
                event.stop()
                self.post_message(self.Submitted(self.get_value()))

    class Submitted(events.Message):
        """提交事件"""
        def __init__(self, value: str):
            super().__init__()
            self.value = value

    def get_value(self) -> str:
        """获取输入值"""
        return self.query_one("#chat_input", Input).value.strip()

    def clear(self):
        """清空输入"""
        self.query_one("#chat_input", Input).value = ""
        self.query_one("#chat_input", Input).focus()

    def set_loading(self, loading: bool):
        """设置加载状态"""
        self.is_loading = loading
