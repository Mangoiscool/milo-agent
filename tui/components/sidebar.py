"""Sidebar Component - Session Management"""

from textual.widgets import Static, Button, ListView, ListItem, Label
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual import events
from rich.text import Text
from datetime import datetime


class SessionItem(ListItem):
    """会话列表项"""

    def __init__(self, session_id: str, title: str, timestamp: datetime, **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.title = title
        self.timestamp = timestamp

    def compose(self):
        with Horizontal():
            yield Label(self.title[:30] + "..." if len(self.title) > 30 else self.title)


class Sidebar(Static):
    """侧边栏 - 会话管理"""

    DEFAULT_CSS = """
    Sidebar {
        width: 25;
        height: 100%;
        dock: left;
        background: $surface-darken-1;
        border-right: solid $primary 30%;
        padding: 1;
    }

    Sidebar #sidebar_header {
        height: 3;
        content-align: center middle;
        color: $primary;
        text-style: bold;
    }

    Sidebar #new_chat_btn {
        width: 100%;
        margin: 1 0;
    }

    Sidebar #session_list {
        width: 100%;
        height: 1fr;
        border: none;
        background: transparent;
    }

    Sidebar #session_list ListItem {
        padding: 0 1;
    }

    Sidebar #session_list ListItem:hover {
        background: $primary 20%;
    }

    Sidebar #session_list ListItem:focus {
        background: $primary 30%;
    }

    Sidebar #sidebar_footer {
        height: 2;
        content-align: center middle;
        color: $text-muted;
        text-style: dim;
    }
    """

    sessions = reactive(list)
    current_session = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sessions = []

    def compose(self):
        """组合界面"""
        with Vertical():
            yield Static("◉ Milo Agent", id="sidebar_header")
            yield Button("+ New Chat", id="new_chat_btn", variant="primary")
            yield ListView(id="session_list")
            yield Static("Ctrl+S to toggle", id="sidebar_footer")

    def on_mount(self):
        """挂载后加载会话列表"""
        self.load_sessions()

    def load_sessions(self):
        """加载会话列表"""
        # TODO: 从实际存储加载
        self.sessions = [
            {"id": "1", "title": "Getting started with Python", "timestamp": datetime.now()},
            {"id": "2", "title": "API design discussion", "timestamp": datetime.now()},
        ]
        self.refresh_list()

    def refresh_list(self):
        """刷新列表"""
        list_view = self.query_one("#session_list", ListView)
        list_view.clear()

        for session in self.sessions:
            item = SessionItem(
                session_id=session["id"],
                title=session["title"],
                timestamp=session["timestamp"],
            )
            list_view.append(item)

    def on_button_pressed(self, event: Button.Pressed):
        """按钮点击"""
        if event.button.id == "new_chat_btn":
            self.post_message(self.NewChat())

    def on_list_view_selected(self, event: ListView.Selected):
        """列表选择"""
        item = event.item
        if isinstance(item, SessionItem):
            self.post_message(self.SessionSelected(item.session_id))

    class NewChat(events.Message):
        """新建会话事件"""
        pass

    class SessionSelected(events.Message):
        """会话选择事件"""
        def __init__(self, session_id: str):
            super().__init__()
            self.session_id = session_id
