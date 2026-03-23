"""Modern TUI Application - Claude Code Style"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual import work

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tui.components import ChatArea, ChatInput, Sidebar, StatusBar
from tui.components.message import MessageBubble, StreamingMessage, ToolCallCard
from tui.screens import WelcomeScreen
from tui.theme import CLAUDE_THEME, THEMES
from tui.config import config


class MiloApp(App):
    """Milo Agent TUI Application"""

    CSS = """
    Screen {
        background: $surface;
    }

    #main_layout {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }

    #content_area {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }

    #chat_container {
        width: 100%;
        height: 1fr;
    }

    #status_bar {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+n", "new_chat", "New Chat"),
        ("ctrl+s", "toggle_sidebar", "Toggle Sidebar"),
        ("ctrl+t", "toggle_thinking", "Toggle Thinking"),
        ("ctrl+r", "toggle_rag", "Toggle RAG"),
        ("ctrl+b", "toggle_browser", "Toggle Browser"),
    ]

    TITLE = "Milo Agent"
    SUB_TITLE = "Terminal AI"

    # 状态
    show_sidebar = reactive(True)
    is_processing = reactive(False)
    current_streaming = reactive(None)

    def __init__(self, agent=None, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.provider = "ollama"
        self.model = "default"
        self.enable_rag = config.get("default_rag", False)
        self.enable_browser = config.get("default_browser", False)
        self.enable_react = config.get("default_react", False)
        self.chat_history = []

    def compose(self) -> ComposeResult:
        """组合界面"""
        with Horizontal(id="main_layout"):
            # 侧边栏
            yield Sidebar(id="sidebar")

            # 主内容区
            with Vertical(id="content_area"):
                yield ChatArea(id="chat_container")
                yield StatusBar(id="status_bar")
                yield ChatInput(id="chat_input")

    def on_mount(self):
        """挂载后初始化"""
        # 注册主题
        self.register_theme(CLAUDE_THEME)

        # 应用主题
        theme_name = config.get("theme", "claude")
        if theme_name in THEMES:
            self.theme = theme_name

        # 显示欢迎界面（如果是新会话）
        if not self.agent:
            self.push_screen(WelcomeScreen(), callback=self._on_welcome_dismissed)
        else:
            self._init_agent()

    def _on_welcome_dismissed(self, result=None):
        """欢迎界面关闭后"""
        self._init_agent()

    def _init_agent(self):
        """初始化 Agent"""
        from agents.milo_agent import MiloAgent
        from core.llm.factory import create_llm
        from core.rag.embeddings import create_embedding

        chat_area = self.query_one("#chat_container", ChatArea)
        status_bar = self.query_one("#status_bar", StatusBar)

        try:
            # 创建 LLM
            llm = create_llm(self.provider)

            # 创建 embedding（如果需要 RAG）
            embedding = None
            if self.enable_rag:
                try:
                    embedding = create_embedding("ollama")
                except Exception as e:
                    chat_area.add_system_message(f"RAG init failed: {e}", level="warning")

            # 创建 Agent
            self.agent = MiloAgent(
                llm=llm,
                enable_builtin_tools=True,
                enable_rag=self.enable_rag and embedding is not None,
                embedding_model=embedding,
                enable_browser=self.enable_browser,
                enable_react=self.enable_react,
            )

            # 更新状态栏
            caps = []
            if self.enable_rag:
                caps.append("RAG")
            if self.enable_browser:
                caps.append("Browser")
            if self.enable_react:
                caps.append("Think")

            status_bar.update_provider(self.provider, self.model)
            status_bar.update_capabilities(caps)
            status_bar.update_connection(True)

            chat_area.add_system_message("Agent ready", level="success")

        except Exception as e:
            chat_area.add_system_message(f"Agent init failed: {e}", level="error")
            status_bar.update_connection(False)

    def watch_show_sidebar(self, show: bool):
        """监听侧边栏显示状态"""
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.styles.display = "block" if show else "none"

    def watch_is_processing(self, processing: bool):
        """监听处理状态"""
        chat_input = self.query_one("#chat_input", ChatInput)
        chat_input.set_loading(processing)

    def on_chat_input_submitted(self, event: ChatInput.Submitted):
        """处理输入提交"""
        if not event.value.strip():
            return

        # 处理命令
        if event.value.startswith("/"):
            self._handle_command(event.value)
            return

        # 发送消息
        self._send_message(event.value)

    def _send_message(self, message: str):
        """发送消息"""
        chat_area = self.query_one("#chat_container", ChatArea)
        chat_input = self.query_one("#chat_input", ChatInput)
        status_bar = self.query_one("#status_bar", StatusBar)

        # 清空输入
        chat_input.clear()

        # 添加用户消息
        chat_area.add_user_message(message)
        status_bar.update_message_count(chat_area.get_message_count())

        # 设置处理状态
        self.is_processing = True

        # 处理消息
        self._process_message(message)

    @work(exclusive=True)
    async def _process_message(self, message: str):
        """处理消息（后台工作）"""
        chat_area = self.query_one("#chat_container", ChatArea)

        try:
            # 创建流式消息
            streaming_msg = chat_area.add_assistant_message()

            # 调用 Agent
            response = await asyncio.to_thread(
                self.agent.chat_with_tools,
                message,
                show_reasoning=self.enable_react
            )

            # 流式显示
            if isinstance(response, str):
                # 逐字显示
                current = ""
                for char in response:
                    current += char
                    streaming_msg.append_text(char)
                    await asyncio.sleep(0.01)

                # 完成
                chat_area.finalize_assistant_message(streaming_msg, response)
            else:
                text = str(response)
                streaming_msg.finalize(text)
                chat_area.finalize_assistant_message(streaming_msg, text)

        except Exception as e:
            import traceback
            error_msg = f"Error: {e}\n```\n{traceback.format_exc()[:500]}\n```"
            chat_area.add_assistant_message(error_msg)

        finally:
            self.is_processing = False
            status_bar = self.query_one("#status_bar", StatusBar)
            status_bar.update_message_count(chat_area.get_message_count())

    def _handle_command(self, command: str):
        """处理命令"""
        chat_area = self.query_one("#chat_container", ChatArea)
        chat_input = self.query_one("#chat_input", ChatInput)

        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/help":
            help_text = self._get_help_text()
            chat_area.add_assistant_message(help_text)

        elif cmd == "/clear":
            chat_area.clear_all()
            chat_area.add_system_message("Chat cleared", level="info")

        elif cmd == "/model" and args:
            self.model = args[0]
            chat_area.add_system_message(f"Model set to: {self.model}", level="success")
            self._init_agent()

        elif cmd == "/quit":
            self.exit()

        else:
            chat_area.add_system_message(f"Unknown command: {cmd}", level="warning")

        chat_input.clear()

    def _get_help_text(self) -> str:
        """获取帮助文本"""
        return """## Commands

- `/help` - Show this help
- `/clear` - Clear chat history
- `/model <name>` - Change model
- `/quit` - Exit

## Shortcuts

- `Enter` - Send message
- `Ctrl+Q` - Quit
- `Ctrl+L` - Clear chat
- `Ctrl+N` - New chat
- `Ctrl+S` - Toggle sidebar
- `Ctrl+T` - Toggle thinking mode
- `Ctrl+R` - Toggle RAG
- `Ctrl+B` - Toggle browser
        """

    # 快捷键动作
    def action_clear(self):
        """清空对话"""
        chat_area = self.query_one("#chat_container", ChatArea)
        chat_area.clear_all()
        chat_area.add_system_message("Chat cleared", level="info")

    def action_new_chat(self):
        """新建对话"""
        chat_area = self.query_one("#chat_container", ChatArea)
        chat_area.clear_all()
        self._init_agent()

    def action_toggle_sidebar(self):
        """切换侧边栏"""
        self.show_sidebar = not self.show_sidebar

    def action_toggle_thinking(self):
        """切换思考模式"""
        self.enable_react = not self.enable_react
        status = "enabled" if self.enable_react else "disabled"
        chat_area = self.query_one("#chat_container", ChatArea)
        chat_area.add_system_message(f"Thinking mode {status}", level="info")
        self._init_agent()

    def action_toggle_rag(self):
        """切换 RAG"""
        self.enable_rag = not self.enable_rag
        status = "enabled" if self.enable_rag else "disabled"
        chat_area = self.query_one("#chat_container", ChatArea)
        chat_area.add_system_message(f"RAG {status}", level="info")
        self._init_agent()

    def action_toggle_browser(self):
        """切换浏览器"""
        self.enable_browser = not self.enable_browser
        status = "enabled" if self.enable_browser else "disabled"
        chat_area = self.query_one("#chat_container", ChatArea)
        chat_area.add_system_message(f"Browser {status}", level="info")
        self._init_agent()


def run_tui(agent=None):
    """运行 TUI"""
    app = MiloApp(agent=agent)
    app.run()


if __name__ == "__main__":
    run_tui()
