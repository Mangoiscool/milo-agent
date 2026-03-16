"""Browser 模块

浏览器自动化模块，提供：
- 浏览器控制器（Playwright 封装）
- 浏览器工具集
- Browser Agent
"""

from .base import (
    BrowserAction,
    BrowserActionResult,
    BrowserConfig,
    InteractiveElement,
    PageState,
    ScrollDirection,
)
from .controller import BrowserController
from .tools import (
    BrowserBackTool,
    BrowserClickTool,
    BrowserGetTextTool,
    BrowserNavigateTool,
    BrowserScreenshotTool,
    BrowserScrollTool,
    BrowserTypeTool,
    BrowserWaitTool,
)

__all__ = [
    # Base
    "BrowserAction",
    "BrowserConfig",
    "BrowserActionResult",
    "InteractiveElement",
    "PageState",
    "ScrollDirection",
    # Controller
    "BrowserController",
    # Tools
    "BrowserNavigateTool",
    "BrowserClickTool",
    "BrowserTypeTool",
    "BrowserScrollTool",
    "BrowserGetTextTool",
    "BrowserScreenshotTool",
    "BrowserWaitTool",
    "BrowserBackTool",
]