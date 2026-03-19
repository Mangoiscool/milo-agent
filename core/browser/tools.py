"""Browser 工具集

提供可被 Agent 调用的浏览器工具。
"""

import asyncio
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.tools.base import BaseTool, ToolResult

from .base import BrowserActionResult, ScrollDirection
from .controller import BrowserController


# 获取默认截图保存目录
def _get_screenshot_dir() -> Path:
    """获取截图保存目录"""
    # 获取项目根目录
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            screenshot_dir = parent / "workspace" / "browser_use" / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            return screenshot_dir
    # 如果找不到项目根目录，使用当前目录
    screenshot_dir = Path.cwd() / "workspace" / "browser_use" / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    return screenshot_dir


SCREENSHOT_DIR = _get_screenshot_dir()


def _run_async(coro):
    """
    在同步上下文中运行异步协程

    处理已有事件循环的情况（如 FastAPI）
    """
    try:
        loop = asyncio.get_running_loop()
        # 已有运行中的事件循环，使用线程池执行
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # 没有运行中的事件循环
        return asyncio.run(coro)


class BrowserNavigateTool(BaseTool):
    """导航工具"""

    @property
    def name(self) -> str:
        return "browser_navigate"

    @property
    def description(self) -> str:
        return "导航到指定的 URL 地址"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "目标 URL 地址，如 https://example.com"
                }
            },
            "required": ["url"]
        }

    def __init__(self, controller: BrowserController):
        self.controller = controller

    def execute(self, url: str) -> ToolResult:
        """同步执行"""
        return _run_async(self.aexecute(url=url))

    async def aexecute(self, url: str) -> ToolResult:
        """异步执行"""
        result = await self.controller.navigate(url)
        return self._to_tool_result(result)


class BrowserClickTool(BaseTool):
    """点击工具"""

    @property
    def name(self) -> str:
        return "browser_click"

    @property
    def description(self) -> str:
        return "点击页面上的元素，可以是 CSS 选择器或元素描述"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS 选择器，如 #submit-btn, .login-button"
                },
                "description": {
                    "type": "string",
                    "description": "元素的描述，如 '登录按钮'"
                }
            },
            "required": ["selector"]
        }

    def __init__(self, controller: BrowserController):
        self.controller = controller

    def execute(self, selector: str, description: str = "") -> ToolResult:
        return _run_async(self.aexecute(selector=selector))

    async def aexecute(self, selector: str, description: str = "") -> ToolResult:
        result = await self.controller.click(selector)
        return self._to_tool_result(result)


class BrowserTypeTool(BaseTool):
    """输入文本工具"""

    @property
    def name(self) -> str:
        return "browser_type"

    @property
    def description(self) -> str:
        return "在输入框中输入文本"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS 选择器，如 #username, input[name='email']"
                },
                "text": {
                    "type": "string",
                    "description": "要输入的文本内容"
                },
                "press_enter": {
                    "type": "boolean",
                    "description": "输入后是否按回车键"
                }
            },
            "required": ["selector", "text"]
        }

    def __init__(self, controller: BrowserController):
        self.controller = controller

    def execute(
        self,
        selector: str,
        text: str,
        press_enter: bool = False
    ) -> ToolResult:
        return _run_async(
            self.aexecute(selector=selector, text=text, press_enter=press_enter)
        )

    async def aexecute(
        self,
        selector: str,
        text: str,
        press_enter: bool = False
    ) -> ToolResult:
        result = await self.controller.type_text(selector, text, press_enter=press_enter)
        return self._to_tool_result(result)


class BrowserScrollTool(BaseTool):
    """滚动工具"""

    @property
    def name(self) -> str:
        return "browser_scroll"

    @property
    def description(self) -> str:
        return "滚动页面，可以向上、向下、到顶部或底部"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "top", "bottom"],
                    "description": "滚动方向"
                }
            },
            "required": ["direction"]
        }

    def __init__(self, controller: BrowserController):
        self.controller = controller

    def execute(self, direction: str) -> ToolResult:
        return _run_async(self.aexecute(direction=direction))

    async def aexecute(self, direction: str) -> ToolResult:
        direction_map = {
            "up": ScrollDirection.UP,
            "down": ScrollDirection.DOWN,
            "top": ScrollDirection.TOP,
            "bottom": ScrollDirection.BOTTOM
        }
        scroll_dir = direction_map.get(direction, ScrollDirection.DOWN)
        result = await self.controller.scroll(scroll_dir)
        return self._to_tool_result(result)


class BrowserGetTextTool(BaseTool):
    """获取文本工具"""

    @property
    def name(self) -> str:
        return "browser_get_text"

    @property
    def description(self) -> str:
        return "获取页面或元素的文本内容"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS 选择器，为空则获取整个页面的文本"
                }
            }
        }

    def __init__(self, controller: BrowserController):
        self.controller = controller

    def execute(self, selector: str = "") -> ToolResult:
        return _run_async(
            self.aexecute(selector=selector if selector else None)
        )

    async def aexecute(self, selector: Optional[str] = None) -> ToolResult:
        result = await self.controller.get_text(selector)
        return self._to_tool_result(result)


class BrowserScreenshotTool(BaseTool):
    """截图工具"""

    @property
    def name(self) -> str:
        return "browser_screenshot"

    @property
    def description(self) -> str:
        return f"""截取当前页面的截图并保存到文件。

保存位置: workspace/browser_use/screenshots/

参数：
- full_page: 是否截取整个页面（包括需要滚动的部分）

返回截图文件的保存路径。"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "full_page": {
                    "type": "boolean",
                    "description": "是否截取整个页面（包括需要滚动的部分）"
                },
                "filename": {
                    "type": "string",
                    "description": "自定义文件名（可选，默认自动生成时间戳文件名）"
                }
            }
        }

    def __init__(self, controller: BrowserController):
        self.controller = controller

    def execute(self, full_page: bool = False, filename: Optional[str] = None) -> ToolResult:
        return _run_async(self.aexecute(full_page=full_page, filename=filename))

    async def aexecute(self, full_page: bool = False, filename: Optional[str] = None) -> ToolResult:
        # 生成文件名
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"

        # 确保文件名以 .png 结尾
        if not filename.endswith(".png"):
            filename += ".png"

        # 构建保存路径
        save_path = SCREENSHOT_DIR / filename

        # 截图并保存
        result = await self.controller.screenshot(path=str(save_path), full_page=full_page)

        if result.success:
            return ToolResult(
                content=f"截图已保存: {save_path}"
            )
        else:
            return ToolResult(
                content="",
                is_error=True,
                error_message=result.message
            )


class BrowserWaitTool(BaseTool):
    """等待工具"""

    @property
    def name(self) -> str:
        return "browser_wait"

    @property
    def description(self) -> str:
        return "等待指定的时间或等待元素出现"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "等待秒数"
                },
                "selector": {
                    "type": "string",
                    "description": "等待元素出现的 CSS 选择器"
                }
            }
        }

    def __init__(self, controller: BrowserController):
        self.controller = controller

    def execute(self, seconds: float = 1.0, selector: str = "") -> ToolResult:
        return _run_async(
            self.aexecute(seconds=seconds, selector=selector if selector else None)
        )

    async def aexecute(
        self,
        seconds: float = 1.0,
        selector: Optional[str] = None
    ) -> ToolResult:
        if selector:
            result = await self.controller.wait_for_selector(selector)
        else:
            result = await self.controller.wait(seconds)
        return self._to_tool_result(result)


class BrowserBackTool(BaseTool):
    """后退工具"""

    @property
    def name(self) -> str:
        return "browser_back"

    @property
    def description(self) -> str:
        return "浏览器后退到上一页"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def __init__(self, controller: BrowserController):
        self.controller = controller

    def execute(self) -> ToolResult:
        return _run_async(self.aexecute())

    async def aexecute(self) -> ToolResult:
        result = await self.controller.back()
        return self._to_tool_result(result)


def _to_tool_result(result: BrowserActionResult) -> ToolResult:
    """转换 BrowserActionResult 到 ToolResult"""
    return ToolResult(
        content=result.message if result.success else f"错误: {result.message}",
        is_error=not result.success
    )


# 为所有工具类添加 _to_tool_result 方法
def _add_to_tool_result(cls):
    """为工具类添加 _to_tool_result 方法"""
    cls._to_tool_result = staticmethod(_to_tool_result)
    return cls

# 装饰所有工具类
BrowserNavigateTool = _add_to_tool_result(BrowserNavigateTool)
BrowserClickTool = _add_to_tool_result(BrowserClickTool)
BrowserTypeTool = _add_to_tool_result(BrowserTypeTool)
BrowserScrollTool = _add_to_tool_result(BrowserScrollTool)
BrowserGetTextTool = _add_to_tool_result(BrowserGetTextTool)
BrowserScreenshotTool = _add_to_tool_result(BrowserScreenshotTool)
BrowserWaitTool = _add_to_tool_result(BrowserWaitTool)
BrowserBackTool = _add_to_tool_result(BrowserBackTool)