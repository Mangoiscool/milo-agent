"""Browser Controller - Playwright 封装

提供浏览器自动化能力：
- 页面导航
- 元素交互
- 状态获取
- 截图
"""

import asyncio
from pathlib import Path
from typing import Optional, Union

from .base import (
    BrowserAction,
    BrowserActionResult,
    BrowserConfig,
    InteractiveElement,
    PageState,
    ScrollDirection,
)


class BrowserController:
    """
    浏览器控制器

    封装 Playwright，提供统一的浏览器操作接口。
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        """
        初始化控制器

        Args:
            config: 浏览器配置
        """
        self.config = config or BrowserConfig()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False

    async def initialize(self):
        """初始化浏览器"""
        if self._initialized:
            return

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()

            # 选择浏览器类型
            browser_launcher = getattr(self._playwright, self.config.browser_type)
            self._browser = await browser_launcher.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo
            )

            # 创建上下文
            self._context = await self._browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height
                }
            )

            # 创建页面
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.config.timeout)

            self._initialized = True

        except ImportError:
            raise ImportError(
                "playwright is required. "
                "Install with: pip install playwright && playwright install"
            )

    async def close(self):
        """关闭浏览器"""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

        self._initialized = False

    def _ensure_initialized(self):
        """确保浏览器已初始化"""
        if not self._initialized:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

    # ═══════════════════════════════════════════════════════════════
    # 页面导航
    # ═══════════════════════════════════════════════════════════════

    async def navigate(self, url: str) -> BrowserActionResult:
        """
        导航到 URL

        Args:
            url: 目标 URL

        Returns:
            操作结果
        """
        self._ensure_initialized()

        try:
            # 确保 URL 有协议
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            await self._page.goto(url, wait_until="domcontentloaded")
            page_state = await self.get_page_state()

            return BrowserActionResult(
                success=True,
                message=f"导航成功: {url}",
                page_state=page_state
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"导航失败: {str(e)}"
            )

    async def back(self) -> BrowserActionResult:
        """后退"""
        self._ensure_initialized()

        try:
            await self._page.go_back()
            page_state = await self.get_page_state()

            return BrowserActionResult(
                success=True,
                message="后退成功",
                page_state=page_state
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"后退失败: {str(e)}"
            )

    async def forward(self) -> BrowserActionResult:
        """前进"""
        self._ensure_initialized()

        try:
            await self._page.go_forward()
            page_state = await self.get_page_state()

            return BrowserActionResult(
                success=True,
                message="前进成功",
                page_state=page_state
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"前进失败: {str(e)}"
            )

    async def refresh(self) -> BrowserActionResult:
        """刷新页面"""
        self._ensure_initialized()

        try:
            await self._page.reload()
            page_state = await self.get_page_state()

            return BrowserActionResult(
                success=True,
                message="刷新成功",
                page_state=page_state
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"刷新失败: {str(e)}"
            )

    # ═══════════════════════════════════════════════════════════════
    # 元素交互
    # ═══════════════════════════════════════════════════════════════

    async def click(
        self,
        selector: str,
        timeout: Optional[int] = None
    ) -> BrowserActionResult:
        """
        点击元素

        Args:
            selector: CSS 选择器或元素描述
            timeout: 超时时间

        Returns:
            操作结果
        """
        self._ensure_initialized()

        try:
            await self._page.click(selector, timeout=timeout or self.config.timeout)
            await asyncio.sleep(0.5)  # 等待页面稳定

            page_state = await self.get_page_state()

            return BrowserActionResult(
                success=True,
                message=f"点击成功: {selector}",
                page_state=page_state
            )
        except Exception as e:
            screenshot = None
            if self.config.screenshot_on_error:
                screenshot = await self._take_screenshot()

            return BrowserActionResult(
                success=False,
                message=f"点击失败: {str(e)}",
                screenshot=screenshot
            )

    async def type_text(
        self,
        selector: str,
        text: str,
        clear_first: bool = True,
        press_enter: bool = False
    ) -> BrowserActionResult:
        """
        输入文本

        Args:
            selector: CSS 选择器
            text: 要输入的文本
            clear_first: 是否先清空
            press_enter: 是否按回车

        Returns:
            操作结果
        """
        self._ensure_initialized()

        try:
            if clear_first:
                await self._page.fill(selector, "")
            await self._page.type(selector, text)

            if press_enter:
                await self._page.press(selector, "Enter")
                await asyncio.sleep(0.5)

            page_state = await self.get_page_state()

            return BrowserActionResult(
                success=True,
                message=f"输入成功: {text}",
                page_state=page_state
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"输入失败: {str(e)}"
            )

    async def press(self, key: str) -> BrowserActionResult:
        """
        按键

        Args:
            key: 按键名称 (Enter, Escape, Tab, etc.)

        Returns:
            操作结果
        """
        self._ensure_initialized()

        try:
            await self._page.keyboard.press(key)
            await asyncio.sleep(0.3)

            page_state = await self.get_page_state()

            return BrowserActionResult(
                success=True,
                message=f"按键成功: {key}",
                page_state=page_state
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"按键失败: {str(e)}"
            )

    async def hover(self, selector: str) -> BrowserActionResult:
        """悬停"""
        self._ensure_initialized()

        try:
            await self._page.hover(selector)

            return BrowserActionResult(
                success=True,
                message=f"悬停成功: {selector}"
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"悬停失败: {str(e)}"
            )

    async def select_option(
        self,
        selector: str,
        value: str
    ) -> BrowserActionResult:
        """下拉选择"""
        self._ensure_initialized()

        try:
            await self._page.select_option(selector, value)

            return BrowserActionResult(
                success=True,
                message=f"选择成功: {value}"
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"选择失败: {str(e)}"
            )

    # ═══════════════════════════════════════════════════════════════
    # 页面操作
    # ═══════════════════════════════════════════════════════════════

    async def scroll(self, direction: ScrollDirection, distance: int = 300) -> BrowserActionResult:
        """
        滚动页面

        Args:
            direction: 滚动方向
            distance: 滚动距离（像素）

        Returns:
            操作结果
        """
        self._ensure_initialized()

        try:
            if direction == ScrollDirection.UP:
                await self._page.evaluate(f"window.scrollBy(0, -{distance})")
            elif direction == ScrollDirection.DOWN:
                await self._page.evaluate(f"window.scrollBy(0, {distance})")
            elif direction == ScrollDirection.TOP:
                await self._page.evaluate("window.scrollTo(0, 0)")
            elif direction == ScrollDirection.BOTTOM:
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            await asyncio.sleep(0.3)
            page_state = await self.get_page_state()

            return BrowserActionResult(
                success=True,
                message=f"滚动成功: {direction.value}",
                page_state=page_state
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"滚动失败: {str(e)}"
            )

    async def wait(self, seconds: float) -> BrowserActionResult:
        """等待"""
        await asyncio.sleep(seconds)
        return BrowserActionResult(success=True, message=f"等待 {seconds} 秒")

    async def wait_for_selector(
        self,
        selector: str,
        timeout: Optional[int] = None
    ) -> BrowserActionResult:
        """等待元素出现"""
        self._ensure_initialized()

        try:
            await self._page.wait_for_selector(
                selector,
                timeout=timeout or self.config.timeout
            )

            return BrowserActionResult(
                success=True,
                message=f"元素出现: {selector}"
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"等待超时: {str(e)}"
            )

    # ═══════════════════════════════════════════════════════════════
    # 信息获取
    # ═══════════════════════════════════════════════════════════════

    async def get_page_state(self) -> PageState:
        """获取当前页面状态"""
        self._ensure_initialized()

        url = self._page.url
        title = await self._page.title()

        # 获取页面内容
        content = await self._get_page_content()

        # 获取可交互元素
        interactive_elements = await self._get_interactive_elements()

        return PageState(
            url=url,
            title=title,
            content=content,
            interactive_elements=interactive_elements
        )

    async def _get_page_content(self) -> str:
        """获取页面文本内容"""
        try:
            # 获取 body 的文本内容
            content = await self._page.evaluate("""
                () => {
                    const body = document.body;
                    // 移除脚本和样式
                    const scripts = body.querySelectorAll('script, style, noscript');
                    scripts.forEach(s => s.remove());
                    return body.innerText;
                }
            """)
            return content or ""
        except Exception:
            return ""

    async def _get_interactive_elements(self) -> list[InteractiveElement]:
        """获取页面上的可交互元素"""
        try:
            elements_data = await self._page.evaluate("""
                () => {
                    const elements = [];
                    const selectors = 'button, a, input, textarea, select, [role="button"], [onclick]';

                    document.querySelectorAll(selectors).forEach((el, index) => {
                        // 跳过隐藏元素
                        if (el.hidden || el.style.display === 'none') return;

                        const rect = el.getBoundingClientRect();
                        if (rect.width === 0 || rect.height === 0) return;

                        // 生成选择器
                        let selector = el.id ? `#${el.id}` : null;
                        if (!selector && el.className) {
                            const classes = el.className.split(' ').filter(c => c);
                            if (classes.length > 0) {
                                selector = `${el.tagName.toLowerCase()}.${classes.slice(0, 2).join('.')}`;
                            }
                        }
                        if (!selector) {
                            selector = el.tagName.toLowerCase();
                        }

                        elements.push({
                            index: index,
                            tag: el.tagName.toLowerCase(),
                            text: el.innerText || el.value || el.placeholder || '',
                            selector: selector,
                            type: el.type || '',
                            placeholder: el.placeholder || '',
                            isVisible: true,
                            isEnabled: !el.disabled,
                            attributes: {
                                id: el.id || '',
                                name: el.name || '',
                                href: el.href || ''
                            }
                        });
                    });

                    return elements.slice(0, 50);  // 限制数量
                }
            """)

            return [
                InteractiveElement(
                    index=el["index"],
                    tag=el["tag"],
                    text=el["text"].strip(),
                    selector=el["selector"],
                    element_type=el["type"],
                    placeholder=el["placeholder"],
                    is_visible=el["isVisible"],
                    is_enabled=el["isEnabled"],
                    attributes=el["attributes"]
                )
                for el in elements_data
            ]
        except Exception:
            return []

    async def get_text(self, selector: Optional[str] = None) -> BrowserActionResult:
        """
        获取文本内容

        Args:
            selector: CSS 选择器，为空则获取整个页面

        Returns:
            操作结果
        """
        self._ensure_initialized()

        try:
            if selector:
                text = await self._page.text_content(selector) or ""
            else:
                text = await self._get_page_content()

            return BrowserActionResult(
                success=True,
                message="获取文本成功",
                data=text
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"获取文本失败: {str(e)}"
            )

    async def get_html(self, selector: Optional[str] = None) -> BrowserActionResult:
        """获取 HTML"""
        self._ensure_initialized()

        try:
            if selector:
                html = await self._page.inner_html(selector)
            else:
                html = await self._page.content()

            return BrowserActionResult(
                success=True,
                message="获取 HTML 成功",
                data=html
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"获取 HTML 失败: {str(e)}"
            )

    # ═══════════════════════════════════════════════════════════════
    # 截图
    # ═══════════════════════════════════════════════════════════════

    async def screenshot(
        self,
        path: Optional[Union[str, Path]] = None,
        full_page: bool = False
    ) -> BrowserActionResult:
        """
        截图

        Args:
            path: 保存路径
            full_page: 是否截取整页

        Returns:
            操作结果
        """
        self._ensure_initialized()

        try:
            screenshot_bytes = await self._page.screenshot(
                path=path,
                full_page=full_page
            )

            return BrowserActionResult(
                success=True,
                message="截图成功",
                screenshot=screenshot_bytes,
                data=str(path) if path else None
            )
        except Exception as e:
            return BrowserActionResult(
                success=False,
                message=f"截图失败: {str(e)}"
            )

    async def _take_screenshot(self) -> Optional[bytes]:
        """内部截图方法"""
        try:
            if self._page:
                return await self._page.screenshot()
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════════════════════════
    # 上下文管理
    # ═══════════════════════════════════════════════════════════════

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()