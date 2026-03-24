"""
BrowserManager - 浏览器管理器

管理浏览器自动化控制器，提供异步初始化和清理。
"""

from typing import Optional

from core.browser import BrowserConfig, BrowserController
from core.logger import get_logger


class BrowserManager:
    """
    浏览器管理器

    封装浏览器控制器的生命周期管理：
    - 异步初始化
    - 自动清理
    - 上下文管理器支持

    使用示例:
        # 作为上下文管理器
        async with BrowserManager() as browser:
            controller = browser.controller
            await controller.navigate("https://example.com")

        # 手动管理
        browser = BrowserManager()
        await browser.initialize()
        # ... use browser.controller ...
        await browser.close()
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        """
        初始化浏览器管理器

        Args:
            config: 浏览器配置（可选）
        """
        self.config = config or BrowserConfig()
        self.controller = BrowserController(self.config)
        self.logger = get_logger(self.__class__.__name__)
        self._initialized = False

    async def initialize(self) -> None:
        """异步初始化浏览器"""
        if self._initialized:
            return

        self.logger.info("Initializing browser...")
        await self.controller.initialize()
        self._initialized = True
        self.logger.info("Browser initialized")

    async def close(self) -> None:
        """关闭浏览器并清理资源"""
        if not self._initialized:
            return

        self.logger.info("Closing browser...")
        await self.controller.close()
        self._initialized = False
        self.logger.info("Browser closed")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    @property
    def is_initialized(self) -> bool:
        """检查浏览器是否已初始化"""
        return self._initialized

    def ensure_initialized(self) -> None:
        """确保浏览器已初始化，否则抛出错误"""
        if not self._initialized:
            raise RuntimeError(
                "Browser not initialized. "
                "Use 'await browser.initialize()' or async context manager."
            )
