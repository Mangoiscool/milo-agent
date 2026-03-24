"""
ToolManager - 工具管理器

统一管理所有工具：
- 内置工具（计算器、搜索、文件操作等）
- RAG 工具（知识库检索）
- Browser 工具（网页自动化）
"""

from typing import List, Optional

from core.logger import get_logger
from core.tools.base import BaseTool
from core.tools.registry import ToolRegistry


class ToolManager:
    """
    工具管理器

    统一管理和组织各种工具，提供分类信息和便捷访问。

    使用示例:
        manager = ToolManager()

        # 注册内置工具
        manager.register_builtin_tools()

        # 注册自定义工具
        manager.register_tools([MyTool()])

        # 获取所有工具定义
        definitions = manager.get_definitions()

        # 执行工具
        result = manager.execute("calculator", expression="1 + 1")
    """

    # 工具分类定义
    BUILTIN_TOOL_NAMES = {
        "calculator", "datetime", "random", "weather",
        "web_search", "file_read", "file_write", "list_dir", "code_execution"
    }

    RAG_TOOL_NAMES = {
        "knowledge_search", "knowledge_add", "knowledge_list", "knowledge_remove"
    }

    BROWSER_TOOL_NAMES = {
        "browser_navigate", "browser_click", "browser_type", "browser_scroll",
        "browser_get_text", "browser_screenshot", "browser_wait", "browser_back"
    }

    def __init__(self):
        """初始化工具管理器"""
        self.registry = ToolRegistry()
        self.logger = get_logger(self.__class__.__name__)

    def register_builtin_tools(self) -> None:
        """注册内置工具"""
        from core.tools.builtin import (
            CalculatorTool,
            CodeExecutionTool,
            DateTimeTool,
            FileReadTool,
            FileWriteTool,
            ListDirTool,
            RandomTool,
            WeatherTool,
            WebSearchTool,
        )

        builtin_tools: List[BaseTool] = [
            CalculatorTool(),
            DateTimeTool(),
            RandomTool(),
            WeatherTool(),
            WebSearchTool(engine="duckduckgo"),
            FileReadTool(),
            FileWriteTool(),
            ListDirTool(),
            CodeExecutionTool(),
        ]

        self.register_tools(builtin_tools)
        self.logger.info(f"Registered {len(builtin_tools)} builtin tools")

    def register_rag_tools(self, retriever, vector_store, splitter, document_loader) -> None:
        """
        注册 RAG 工具

        Args:
            retriever: 文档检索器
            vector_store: 向量存储
            splitter: 文本切分器
            document_loader: 文档加载器
        """
        from core.rag import (
            RAGAddDocumentTool,
            RAGListSourcesTool,
            RAGRemoveSourceTool,
            RAGSearchTool,
        )

        rag_tools = [
            RAGSearchTool(retriever),
            RAGAddDocumentTool(vector_store, splitter, document_loader),
            RAGListSourcesTool(vector_store),
            RAGRemoveSourceTool(vector_store),
        ]

        self.register_tools(rag_tools)
        self.logger.info(f"Registered {len(rag_tools)} RAG tools")

    def register_browser_tools(self, browser_controller) -> None:
        """
        注册 Browser 工具

        Args:
            browser_controller: 浏览器控制器
        """
        from core.browser.tools import (
            BrowserBackTool,
            BrowserClickTool,
            BrowserGetTextTool,
            BrowserNavigateTool,
            BrowserScreenshotTool,
            BrowserScrollTool,
            BrowserTypeTool,
            BrowserWaitTool,
        )

        browser_tools = [
            BrowserNavigateTool(browser_controller),
            BrowserClickTool(browser_controller),
            BrowserTypeTool(browser_controller),
            BrowserScrollTool(browser_controller),
            BrowserGetTextTool(browser_controller),
            BrowserScreenshotTool(browser_controller),
            BrowserWaitTool(browser_controller),
            BrowserBackTool(browser_controller),
        ]

        self.register_tools(browser_tools)
        self.logger.info(f"Registered {len(browser_tools)} browser tools")

    def register_tool(self, tool: BaseTool) -> None:
        """注册单个工具"""
        self.registry.register(tool)

    def register_tools(self, tools: List[BaseTool]) -> None:
        """批量注册工具"""
        for tool in tools:
            self.registry.register(tool)

    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        return self.registry.unregister(name)

    def get_definitions(self) -> Optional[List[dict]]:
        """获取所有工具定义（用于 LLM）"""
        if self.registry.count() == 0:
            return None
        return self.registry.get_all_definitions()

    def execute(self, name: str, **kwargs):
        """执行工具"""
        return self.registry.execute(name, **kwargs)

    async def aexecute(self, name: str, **kwargs):
        """异步执行工具"""
        return await self.registry.aexecute(name, **kwargs)

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return self.registry.list_tools()

    def count(self) -> int:
        """获取工具数量"""
        return self.registry.count()

    def get_tool_info(self) -> dict:
        """
        获取工具分类信息

        Returns:
            包含分类统计的工具信息
        """
        all_tools = self.list_tools()

        return {
            "total_count": len(all_tools),
            "builtin_tools": [t for t in all_tools if t in self.BUILTIN_TOOL_NAMES],
            "rag_tools": [t for t in all_tools if t in self.RAG_TOOL_NAMES],
            "browser_tools": [t for t in all_tools if t in self.BROWSER_TOOL_NAMES],
            "all_tools": all_tools,
        }

    def has_tool(self, name: str) -> bool:
        """检查是否存在指定工具"""
        return name in self.registry._tools

    def clear(self) -> None:
        """清空所有工具"""
        self.registry._tools.clear()
        self.logger.info("All tools cleared")
