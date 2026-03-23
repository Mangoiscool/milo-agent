"""Chat Area Component - Main Chat Display"""

from textual.widgets import Static
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual import events
from datetime import datetime

from tui.components.message import (
    MessageBubble,
    StreamingMessage,
    ToolCallCard,
    ThinkingCard,
    SystemMessage,
)


class ChatArea(Static):
    """聊天区域组件"""

    DEFAULT_CSS = """
    ChatArea {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    ChatArea > VerticalScroll {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }

    ChatArea MessageBubble {
        margin: 1 0;
    }

    ChatArea StreamingMessage {
        margin: 1 0;
    }

    ChatArea ToolCallCard {
        margin: 0 0 1 0;
    }

    ChatArea ThinkingCard {
        margin: 0 0 1 0;
    }

    ChatArea SystemMessage {
        margin: 1 0;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scroll_container = None
        self._message_count = 0

    def compose(self):
        """组合界面"""
        with VerticalScroll():
            pass  # 消息动态添加

    def on_mount(self):
        """挂载后"""
        self._scroll_container = self.query_one(VerticalScroll)

    def add_user_message(self, content: str) -> MessageBubble:
        """添加用户消息"""
        msg = MessageBubble(role="user", content=content)
        self._scroll_container.mount(msg)
        msg.scroll_visible(animate=False)
        self._message_count += 1
        return msg

    def add_assistant_message(self, content: str = "") -> StreamingMessage:
        """添加助手消息（流式）"""
        msg = StreamingMessage()
        self._scroll_container.mount(msg)
        msg.scroll_visible(animate=False)
        self._message_count += 1
        return msg

    def finalize_assistant_message(self, stream_msg: StreamingMessage, content: str) -> MessageBubble:
        """完成助手消息"""
        stream_msg.remove()

        msg = MessageBubble(role="assistant", content=content)
        self._scroll_container.mount(msg)
        msg.scroll_visible(animate=False)
        return msg

    def add_tool_call(self, name: str, args: dict) -> ToolCallCard:
        """添加工具调用"""
        tool = ToolCallCard(name=name, args=args)
        self._scroll_container.mount(tool)
        tool.scroll_visible(animate=False)
        return tool

    def add_thinking(self, content: str) -> ThinkingCard:
        """添加思考过程"""
        thinking = ThinkingCard(content=content)
        self._scroll_container.mount(thinking)
        thinking.scroll_visible(animate=False)
        return thinking

    def add_system_message(self, content: str, level: str = "info"):
        """添加系统消息"""
        msg = SystemMessage(content=content, level=level)
        self._scroll_container.mount(msg)
        msg.scroll_visible(animate=False)

    def clear_all(self):
        """清空所有消息"""
        if self._scroll_container:
            self._scroll_container.remove_children()
            self._message_count = 0

    def get_message_count(self) -> int:
        """获取消息计数"""
        return self._message_count

    def scroll_to_bottom(self):
        """滚动到底部"""
        if self._scroll_container:
            self._scroll_container.scroll_end(animate=False)
