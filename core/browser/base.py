"""Browser 模块基础类型"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class BrowserAction(Enum):
    """浏览器动作类型"""
    NAVIGATE = "navigate"           # 导航到 URL
    CLICK = "click"                 # 点击元素
    TYPE = "type"                   # 输入文本
    SCROLL = "scroll"               # 滚动页面
    SCREENSHOT = "screenshot"       # 截图
    GET_TEXT = "get_text"           # 获取文本
    GET_HTML = "get_html"           # 获取 HTML
    WAIT = "wait"                   # 等待
    BACK = "back"                   # 后退
    FORWARD = "forward"             # 前进
    REFRESH = "refresh"             # 刷新
    SELECT = "select"               # 下拉选择
    HOVER = "hover"                 # 悬停
    PRESS = "press"                 # 按键


class ScrollDirection(Enum):
    """滚动方向"""
    UP = "up"
    DOWN = "down"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class InteractiveElement:
    """可交互元素"""
    index: int                      # 元素索引
    tag: str                        # 标签名
    text: str                       # 文本内容
    selector: str                   # CSS 选择器
    element_type: str = ""          # 元素类型 (button, link, input, etc.)
    placeholder: str = ""           # 占位符文本
    is_visible: bool = True         # 是否可见
    is_enabled: bool = True         # 是否可用
    attributes: dict[str, str] = field(default_factory=dict)

    def to_description(self) -> str:
        """生成元素描述（给 LLM 看）"""
        parts = [f"[{self.index}]"]

        # 标签类型
        tag_desc = {
            "button": "按钮",
            "a": "链接",
            "input": "输入框",
            "textarea": "文本域",
            "select": "下拉框",
            "img": "图片",
        }.get(self.tag.lower(), self.tag)
        parts.append(f"<{tag_desc}>")

        # 文本或占位符
        if self.text:
            # 截断长文本
            text = self.text[:50] + "..." if len(self.text) > 50 else self.text
            parts.append(f'"{text}"')
        elif self.placeholder:
            parts.append(f'placeholder="{self.placeholder}"')

        return " ".join(parts)


@dataclass
class PageState:
    """页面状态"""
    url: str
    title: str
    content: str                              # 简化的页面内容
    interactive_elements: list[InteractiveElement] = field(default_factory=list)
    screenshot: Optional[bytes] = None        # 页面截图
    raw_html: str = ""                        # 原始 HTML

    def to_context(self) -> str:
        """生成页面上下文（给 LLM 看）"""
        lines = [
            f"当前页面: {self.url}",
            f"页面标题: {self.title}",
            "",
            "可交互元素："
        ]

        if self.interactive_elements:
            for elem in self.interactive_elements[:30]:  # 限制数量
                lines.append(elem.to_description())
        else:
            lines.append("（无可交互元素）")

        # 页面内容摘要
        if self.content:
            content_preview = self.content[:500]
            if len(self.content) > 500:
                content_preview += "..."
            lines.extend([
                "",
                "页面内容摘要：",
                content_preview
            ])

        return "\n".join(lines)


@dataclass
class BrowserActionResult:
    """浏览器动作结果"""
    success: bool
    message: str
    page_state: Optional[PageState] = None
    data: Any = None                          # 提取的数据
    screenshot: Optional[bytes] = None        # 截图

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data
        }


@dataclass
class BrowserConfig:
    """浏览器配置"""
    headless: bool = True                     # 无头模式
    browser_type: str = "chromium"            # 浏览器类型
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout: int = 30000                      # 默认超时（毫秒）
    slow_mo: int = 0                          # 操作延迟（毫秒）
    screenshot_on_error: bool = True          # 错误时截图