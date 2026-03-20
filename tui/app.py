"""Modern TUI Application - Claude Code style"""

import asyncio
from typing import Optional
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Input
from textual.reactive import reactive
from textual import work
from rich.text import Text
from rich.panel import Panel

from agents.main import MainAgent
from core.llm.factory import create_llm
from core.rag.embeddings import create_embedding
from tui.widgets import (
    ChatContainer,
    PromptInput,
    StatusFooter,
)


class MiloTUIApp(App):
    """Milo Agent TUI - Claude Code style"""

    CSS = """
    Screen {
        align: center middle;
        background: $surface;
    }

    #main_container {
        width: 100%;
        height: 100%;
        layout: vertical;
    }

    #chat_area {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+t", "toggle_thinking", "Think"),
        ("ctrl+r", "toggle_rag", "RAG"),
        ("ctrl+b", "toggle_browser", "Browser"),
    ]

    TITLE = "Milo Agent"
    SUB_TITLE = "Terminal UI"

    def __init__(self, agent: Optional[MainAgent] = None, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.is_processing = False
        self.show_thinking = False
        self.enable_rag = False
        self.enable_browser = False
        self.enable_react = False
        self.provider = "ollama"
        self.model = "default"

    def compose(self) -> ComposeResult:
        """构建界面"""
        yield Header(show_clock=False)

        with VerticalScroll(id="main_container"):
            yield ChatContainer(id="chat_area")
            yield StatusFooter(id="status_footer")
            yield PromptInput(id="prompt_input")

    def on_mount(self):
        """挂载后初始化"""
        # 显示欢迎消息
        self.show_welcome()

        # 初始化 Agent
        if self.agent is None:
            self.init_agent()

        # 聚焦输入
        self.query_one("#chat_input", Input).focus()

    def show_welcome(self):
        """显示欢迎消息"""
        welcome = """
╭────────────────────────────────────────╮
│  Welcome to Milo Agent TUI             │
│                                        │
│  Shortcuts:                            │
│    Ctrl+C  - Quit                      │
│    Ctrl+L  - Clear chat                │
│    Ctrl+T  - Toggle thinking mode      │
│    Ctrl+R  - Toggle RAG                │
│    Ctrl+B  - Toggle browser            │
│                                        │
│  Commands:                             │
│    /help   - Show help                 │
│    /clear  - Clear chat                │
│    /model  - Change model              │
│    /quit   - Exit                      │
╰────────────────────────────────────────╯
        """
        chat_area = self.query_one("#chat_area", ChatContainer)
        welcome_msg = chat_area.add_assistant_message(welcome)

    def init_agent(self):
        """初始化 Agent"""
        try:
            llm = create_llm(self.provider)

            embedding = None
            if self.enable_rag:
                try:
                    embedding = create_embedding("ollama")
                except Exception as e:
                    self.notify(f"Embedding init failed: {e}", severity="warning", timeout=3)

            self.agent = MainAgent(
                llm=llm,
                enable_builtin_tools=True,
                enable_rag=self.enable_rag and embedding is not None,
                embedding_model=embedding,
                enable_browser=self.enable_browser,
                enable_react=self.enable_react,
            )

            # 更新状态栏
            footer = self.query_one("#status_footer", StatusFooter)
            caps = []
            if self.enable_rag:
                caps.append("RAG")
            if self.enable_browser:
                caps.append("Browser")
            if self.enable_react:
                caps.append("Think")
            footer.update_status(self.provider, self.model, caps)

            self.notify("Agent ready", severity="information", timeout=2)

        except Exception as e:
            self.notify(f"Agent init failed: {e}", severity="error")

    def on_input_submitted(self, event):
        """输入提交事件"""
        if event.input.id == "chat_input":
            asyncio.create_task(self.handle_submit())

    def on_button_pressed(self, event):
        """按钮点击事件"""
        if event.button.id == "stop_btn":
            self.stop_processing()

    async def handle_submit(self):
        """处理提交"""
        if self.agent is None:
            self.notify("Agent not initialized", severity="error")
            return

        prompt_input = self.query_one("#prompt_input", PromptInput)
        message = prompt_input.get_value().strip()

        if not message:
            return

        # 处理命令
        if message.startswith("/"):
            await self.handle_command(message)
            return

        # 清空输入
        prompt_input.clear()

        # 添加用户消息
        chat_area = self.query_one("#chat_area", ChatContainer)
        chat_area.add_user_message(message)

        # 设置处理状态
        self.set_processing(True)

        # 处理消息
        await self.process_message(message)

    @work(exclusive=True)
    async def process_message(self, message: str):
        """处理消息"""
        chat_area = self.query_one("#chat_area", ChatContainer)
        prompt_input = self.query_one("#prompt_input", PromptInput)

        try:
            # 创建助手消息占位
            assistant_msg = chat_area.add_assistant_message("")

            # 调用 Agent
            try:
                response = await asyncio.to_thread(
                    self.agent.chat_with_tools,
                    message,
                    show_reasoning=self.enable_react
                )
            except AttributeError:
                # fallback to direct method
                response = self.agent.chat_with_tools(message)

            # 流式更新效果
            if isinstance(response, str):
                # 模拟流式输出
                words = response.split(" ")
                current = ""
                for i, word in enumerate(words):
                    current += word + " "
                    if i % 5 == 0 or i == len(words) - 1:
                        assistant_msg.update_content(current)
                        await asyncio.sleep(0.01)

                assistant_msg.update_content(response)
            else:
                assistant_msg.update_content(str(response))

        except Exception as e:
            import traceback
            error_msg = f"Error: {e}\n```\n{traceback.format_exc()[:500]}\n```"
            assistant_msg.update_content(error_msg)

        finally:
            self.set_processing(False)

    def set_processing(self, processing: bool):
        """设置处理状态"""
        self.is_processing = processing
        prompt_input = self.query_one("#prompt_input", PromptInput)
        prompt_input.set_loading(processing)

    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        self.notify("Processing stopped", severity="warning", timeout=2)
        self.set_processing(False)

    async def handle_command(self, command: str):
        """处理命令"""
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:]

        chat_area = self.query_one("#chat_area", ChatContainer)
        prompt_input = self.query_one("#prompt_input", PromptInput)

        if cmd == "/help":
            help_text = """## Available Commands

- `/help` - Show this help
- `/clear` - Clear chat history
- `/model <name>` - Change model
- `/think` - Toggle thinking mode
- `/rag` - Toggle RAG
- `/browser` - Toggle browser
- `/quit` - Exit

## Shortcuts
- `Ctrl+C` - Quit
- `Ctrl+L` - Clear chat
- `Ctrl+T` - Toggle thinking
- `Ctrl+R` - Toggle RAG
- `Ctrl+B` - Toggle browser
            """
            chat_area.add_assistant_message(help_text)

        elif cmd == "/clear":
            chat_area.clear_all()
            self.show_welcome()

        elif cmd == "/model" and args:
            self.model = args[0]
            self.notify(f"Model changed to: {self.model}", severity="information")
            self.init_agent()

        elif cmd == "/think":
            self.enable_react = not self.enable_react
            status = "enabled" if self.enable_react else "disabled"
            self.notify(f"Thinking mode {status}", severity="information")
            self.init_agent()

        elif cmd == "/rag":
            self.enable_rag = not self.enable_rag
            status = "enabled" if self.enable_rag else "disabled"
            self.notify(f"RAG {status}", severity="information")
            self.init_agent()

        elif cmd == "/browser":
            self.enable_browser = not self.enable_browser
            status = "enabled" if self.enable_browser else "disabled"
            self.notify(f"Browser {status}", severity="information")
            self.init_agent()

        elif cmd == "/quit":
            self.exit()

        else:
            self.notify(f"Unknown command: {cmd}", severity="warning")

        prompt_input.clear()

    def action_clear(self):
        """清空对话"""
        chat_area = self.query_one("#chat_area", ChatContainer)
        chat_area.clear_all()
        self.show_welcome()

    def action_toggle_thinking(self):
        """切换思考模式"""
        self.enable_react = not self.enable_react
        status = "enabled" if self.enable_react else "disabled"
        self.notify(f"Thinking mode {status}", severity="information")
        self.init_agent()

    def action_toggle_rag(self):
        """切换 RAG"""
        self.enable_rag = not self.enable_rag
        status = "enabled" if self.enable_rag else "disabled"
        self.notify(f"RAG {status}", severity="information")
        self.init_agent()

    def action_toggle_browser(self):
        """切换浏览器"""
        self.enable_browser = not self.enable_browser
        status = "enabled" if self.enable_browser else "disabled"
        self.notify(f"Browser {status}", severity="information")
        self.init_agent()


def run_tui(agent: Optional[MainAgent] = None):
    """运行 TUI"""
    app = MiloTUIApp(agent=agent)
    app.run()


if __name__ == "__main__":
    run_tui()
