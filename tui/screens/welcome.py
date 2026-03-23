"""Welcome Screen"""

from textual.screen import Screen
from textual.widgets import Static, Button
from textual.containers import Vertical, Horizontal, Center
from textual import events
from rich.text import Text
from rich.panel import Panel
from rich.align import Align


class WelcomeScreen(Screen):
    """欢迎界面"""

    CSS = """
    WelcomeScreen {
        align: center middle;
        background: $surface;
    }

    WelcomeScreen #welcome_container {
        width: 80;
        height: auto;
        padding: 2 4;
    }

    WelcomeScreen #logo {
        width: 100%;
        height: auto;
        content-align: center middle;
        color: $primary;
        text-style: bold;
        margin: 2 0;
    }

    WelcomeScreen #subtitle {
        width: 100%;
        height: auto;
        content-align: center middle;
        color: $text-secondary;
        margin: 1 0;
    }

    WelcomeScreen #features {
        width: 100%;
        height: auto;
        margin: 2 0;
    }

    WelcomeScreen #start_btn {
        width: auto;
        min-width: 20;
        margin: 2 0;
    }
    """

    BINDINGS = [
        ("enter", "start", "Start"),
        ("escape", "app.quit", "Quit"),
    ]

    def compose(self):
        """组合界面"""
        with Vertical(id="welcome_container"):
            yield Static(self._get_logo(), id="logo")
            yield Static("Terminal AI Assistant", id="subtitle")
            yield Static(self._get_features(), id="features")
            with Center():
                yield Button("Start Chatting", id="start_btn", variant="primary")

    def _get_logo(self) -> str:
        """获取 Logo"""
        return """
╭────────────────────────────╮
│                            │
│     ◉ Milo Agent           │
│                            │
╰────────────────────────────╯
        """

    def _get_features(self) -> str:
        """获取功能列表"""
        return """
Features:
  ● Multi-provider LLM support (Ollama, OpenAI, DeepSeek, etc.)
  ● Built-in tools: calculator, web search, file operations
  ● RAG: Knowledge base retrieval
  ● Browser automation
  ● ReAct reasoning mode

Shortcuts:
  Enter      - Send message
  Ctrl+Q     - Quit
  Ctrl+L     - Clear chat
  Ctrl+N     - New chat
  Ctrl+S     - Toggle sidebar
  Ctrl+,     - Settings
        """

    def on_button_pressed(self, event: Button.Pressed):
        """按钮点击"""
        if event.button.id == "start_btn":
            self.action_start()

    def action_start(self):
        """开始聊天"""
        self.dismiss()
