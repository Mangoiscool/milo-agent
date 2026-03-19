"""MainAgent - 统一的主 Agent

提供完整功能的 Agent：
- 内置工具（默认启用）
- RAG 能力（可选）
- Browser 能力（可选）

使用示例：
    from core.llm.factory import LLMFactory
    from core.rag import create_embedding
    from agents.main import MainAgent

    llm = LLMFactory.create("qwen", api_key="...")
    embedding = create_embedding("ollama", model="nomic-embed-text")

    # 创建具备所有能力的 Agent
    agent = MainAgent(
        llm=llm,
        enable_builtin_tools=True,
        enable_rag=True,
        embedding_model=embedding,
        enable_browser=True
    )

    # 添加知识
    agent.add_document("guide.pdf")

    # 对话
    response = agent.chat_with_tools("帮我查一下...")
"""

from pathlib import Path
from typing import Any, List, Optional

from agents.agent_config import AgentConfig
from agents.base import AgentEvent, BaseAgent


# 获取项目根目录
def _get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = _get_project_root()
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "workspace" / "knowledge_base"
from core.browser import BrowserConfig, BrowserController
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
from core.llm.base import BaseLLM
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.rag import (
    RAGAddDocumentTool,
    RAGListSourcesTool,
    RAGRemoveSourceTool,
    RAGSearchTool,
    VectorStore,
    create_retriever,
    create_splitter,
)
from core.rag.document_loader import create_default_registry
from core.rag.embeddings import BaseEmbedding
from core.rag.text_splitter import SplitConfig
from core.tools.base import BaseTool


class MainAgent(BaseAgent):
    """
    统一的主 Agent

    特性：
    - 默认启用所有内置工具（计算器、搜索、文件操作等）
    - 可选启用 RAG 能力（知识库检索和自主学习）
    - 可选启用 Browser 能力（网页自动化）

    使用示例：
        # 最简单的用法
        agent = MainAgent(llm)
        response = agent.chat_with_tools("今天天气怎么样？")

        # 启用 RAG
        agent = MainAgent(
            llm=llm,
            enable_rag=True,
            embedding_model=embedding
        )
        agent.add_document("company_guide.pdf")
        response = agent.chat_with_tools("公司的请假流程是什么？")

        # 启用 Browser
        agent = MainAgent(llm, enable_browser=True)
        await agent.initialize()
        response = agent.chat_with_tools("打开百度搜索 Python")
        await agent.close()

        # 完整功能
        agent = MainAgent(
            llm=llm,
            enable_rag=True,
            embedding_model=embedding,
            enable_browser=True
        )
    """

    DEFAULT_SYSTEM_PROMPT = """你是一个智能助手，可以使用多种工具来帮助用户完成任务。

你可以使用以下类型的工具：
1. 内置工具：计算器、日期时间、网络搜索、文件操作等
2. 知识库检索：查询已有的文档和资料
3. 浏览器操作：打开网页、点击、输入、截图等

工作原则：
1. 仔细理解用户需求
2. 选择最合适的工具完成任务
3. 如果一个工具不够，可以组合使用多个工具
4. 完成任务后给出清晰的总结"""

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        max_tool_iterations: int = 10,
        # ═══════════════════════════════════════════════════════════════
        # 能力开关
        # ═══════════════════════════════════════════════════════════════
        enable_builtin_tools: bool = True,
        enable_rag: bool = False,
        enable_browser: bool = False,
        # ═══════════════════════════════════════════════════════════════
        # RAG 配置
        # ═══════════════════════════════════════════════════════════════
        embedding_model: Optional[BaseEmbedding] = None,
        persist_directory: Optional[str] = None,
        knowledge_base_name: str = "main_agent_kb",
        retriever_type: str = "similarity",
        splitter_config: Optional[SplitConfig] = None,
        # ═══════════════════════════════════════════════════════════════
        # Browser 配置
        # ═══════════════════════════════════════════════════════════════
        browser_config: Optional[BrowserConfig] = None,
    ):
        """
        初始化 MainAgent

        Args:
            llm: LLM 实例
            memory: 记忆系统
            system_prompt: 系统提示词
            config: Agent 配置
            max_tool_iterations: 最大工具调用迭代次数
            enable_builtin_tools: 是否启用内置工具（默认 True）
            enable_rag: 是否启用 RAG 能力（默认 False）
            enable_browser: 是否启用浏览器能力（默认 False）
            embedding_model: Embedding 模型（启用 RAG 时必需）
            persist_directory: 知识库持久化目录
            knowledge_base_name: 知识库名称
            retriever_type: 检索器类型
            splitter_config: 文本切分配置
            browser_config: 浏览器配置
        """
        # 使用默认系统提示词
        effective_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # 初始化基类
        super().__init__(
            llm=llm,
            memory=memory,
            tools=None,  # 稍后注册
            system_prompt=effective_prompt,
            config=config,
            max_tool_iterations=max_tool_iterations
        )

        # 保存配置
        self.enable_rag = enable_rag
        self.enable_browser = enable_browser
        self.knowledge_base_name = knowledge_base_name
        self.persist_directory = persist_directory or str(DEFAULT_PERSIST_DIR)

        # ═══════════════════════════════════════════════════════════════
        # 注册工具
        # ═══════════════════════════════════════════════════════════════

        # 1. 内置工具
        if enable_builtin_tools:
            self._register_builtin_tools()

        # 2. RAG
        if enable_rag:
            if embedding_model is None:
                raise ValueError("embedding_model is required when enable_rag=True")
            self._setup_rag(
                embedding_model=embedding_model,
                persist_directory=persist_directory,
                knowledge_base_name=knowledge_base_name,
                retriever_type=retriever_type,
                splitter_config=splitter_config
            )

        # 3. Browser
        if enable_browser:
            self._setup_browser(browser_config)

    # ═══════════════════════════════════════════════════════════════
    # 工具注册
    # ═══════════════════════════════════════════════════════════════

    def _register_builtin_tools(self) -> None:
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

    def _setup_rag(
        self,
        embedding_model: BaseEmbedding,
        persist_directory: Optional[str],
        knowledge_base_name: str,
        retriever_type: str,
        splitter_config: Optional[SplitConfig]
    ) -> None:
        """
        设置 RAG 能力

        Args:
            embedding_model: Embedding 模型
            persist_directory: 持久化目录（默认为 workspace/knowledge_base）
            knowledge_base_name: 知识库名称
            retriever_type: 检索器类型
            splitter_config: 切分配置
        """
        self.embedding_model = embedding_model

        # 设置默认持久化目录
        effective_persist_dir = persist_directory or str(DEFAULT_PERSIST_DIR)

        # 向量存储
        self.vector_store = VectorStore(
            collection_name=knowledge_base_name,
            persist_directory=effective_persist_dir,
            embedding_model=embedding_model
        )

        # 文本切分器
        self.splitter = create_splitter(
            "recursive",
            config=splitter_config or SplitConfig()
        )

        # 文档加载器
        self.document_loader = create_default_registry()

        # 检索器
        self.retriever = create_retriever(
            self.vector_store,
            embedding_model,
            retriever_type
        )

        # 注册 RAG 工具
        rag_tools = [
            RAGSearchTool(self.retriever),
            RAGAddDocumentTool(self.vector_store, self.splitter, self.document_loader),
            RAGListSourcesTool(self.vector_store),
            RAGRemoveSourceTool(self.vector_store),
        ]

        self.register_tools(rag_tools)
        self.logger.info(f"Registered {len(rag_tools)} RAG tools")

    def _setup_browser(self, browser_config: Optional[BrowserConfig]) -> None:
        """
        设置 Browser 能力

        Args:
            browser_config: 浏览器配置
        """
        self.browser_config = browser_config or BrowserConfig()
        self.browser_controller = BrowserController(self.browser_config)

        # 注册浏览器工具
        browser_tools = [
            BrowserNavigateTool(self.browser_controller),
            BrowserClickTool(self.browser_controller),
            BrowserTypeTool(self.browser_controller),
            BrowserScrollTool(self.browser_controller),
            BrowserGetTextTool(self.browser_controller),
            BrowserScreenshotTool(self.browser_controller),
            BrowserWaitTool(self.browser_controller),
            BrowserBackTool(self.browser_controller),
        ]

        self.register_tools(browser_tools)
        self.logger.info(f"Registered {len(browser_tools)} browser tools")

    # ═══════════════════════════════════════════════════════════════
    # 知识库管理 API
    # ═══════════════════════════════════════════════════════════════

    def add_document(
        self,
        file_path: str | Path,
        metadata: Optional[dict[str, Any]] = None
    ) -> int:
        """
        添加文档到知识库

        Args:
            file_path: 文档路径
            metadata: 额外的元数据

        Returns:
            添加的 chunk 数量
        """
        if not self.enable_rag:
            raise RuntimeError("RAG is not enabled. Set enable_rag=True")

        path = Path(file_path)
        self.logger.info(f"Loading document: {path}")

        # 加载文档
        documents = self.document_loader.load(path)

        # 添加元数据
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        # 切分
        chunks = self.splitter.split_documents(documents)

        # 存储
        ids = self.vector_store.add_chunks(chunks)

        self.logger.info(f"Added {len(ids)} chunks from {path.name}")
        return len(ids)

    def add_text(
        self,
        text: str,
        source: str = "user_input",
        metadata: Optional[dict[str, Any]] = None
    ) -> int:
        """
        添加文本到知识库

        Args:
            text: 文本内容
            source: 来源标识
            metadata: 元数据

        Returns:
            添加的 chunk 数量
        """
        if not self.enable_rag:
            raise RuntimeError("RAG is not enabled. Set enable_rag=True")

        from core.rag.base import Document

        # 创建文档
        doc = Document.from_text(text, source=source)

        # 切分
        chunks = self.splitter.split_document(doc)

        # 添加元数据
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        # 存储
        ids = self.vector_store.add_chunks(chunks)

        self.logger.info(f"Added {len(ids)} chunks from text")
        return len(ids)

    def add_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        extensions: Optional[list[str]] = None
    ) -> dict[str, int]:
        """
        批量添加目录下的文档

        Args:
            directory: 目录路径
            recursive: 是否递归子目录
            extensions: 文件扩展名过滤

        Returns:
            文件名 -> chunk 数量的字典
        """
        if not self.enable_rag:
            raise RuntimeError("RAG is not enabled. Set enable_rag=True")

        results = {}
        documents = self.document_loader.load_directory(
            directory,
            recursive=recursive,
            extensions=extensions
        )

        # 按来源分组
        by_source: dict[str, list] = {}
        for doc in documents:
            source = doc.source or "unknown"
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(doc)

        # 切分并存储
        for source, docs in by_source.items():
            chunks = self.splitter.split_documents(docs)
            ids = self.vector_store.add_chunks(chunks)
            results[source] = len(ids)
            self.logger.info(f"Added {len(ids)} chunks from {source}")

        return results

    def list_sources(self) -> list[str]:
        """
        列出知识库中的所有文档来源

        Returns:
            来源列表
        """
        if not self.enable_rag:
            raise RuntimeError("RAG is not enabled. Set enable_rag=True")

        all_docs = self.vector_store.get(limit=10000)

        sources = set()
        for doc in all_docs:
            source = doc.get("metadata", {}).get("source", "")
            if source:
                sources.add(source)

        return sorted(list(sources))

    def remove_document(self, source: str) -> int:
        """
        移除指定来源的文档

        Args:
            source: 文档来源标识

        Returns:
            移除的文档数量
        """
        if not self.enable_rag:
            raise RuntimeError("RAG is not enabled. Set enable_rag=True")

        count_before = self.vector_store.count()
        self.vector_store.delete(where={"source": source})
        count_after = self.vector_store.count()

        removed = count_before - count_after
        self.logger.info(f"Removed {removed} chunks from source: {source}")
        return removed

    def get_knowledge_base_stats(self) -> dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息字典
        """
        if not self.enable_rag:
            return {"enabled": False}

        return {
            "enabled": True,
            "knowledge_base_name": self.knowledge_base_name,
            "document_count": self.vector_store.count(),
            "sources": self.list_sources(),
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model.model_name if hasattr(self, 'embedding_model') else None,
        }

    # ═══════════════════════════════════════════════════════════════
    # Browser 管理
    # ═══════════════════════════════════════════════════════════════

    async def initialize(self) -> None:
        """初始化异步资源（如浏览器）"""
        if self.enable_browser and hasattr(self, 'browser_controller'):
            self.logger.info("Initializing browser...")
            await self.browser_controller.initialize()
            self.logger.info("Browser initialized")

    async def close(self) -> None:
        """清理资源"""
        if self.enable_browser and hasattr(self, 'browser_controller'):
            self.logger.info("Closing browser...")
            await self.browser_controller.close()
            self.logger.info("Browser closed")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    # ═══════════════════════════════════════════════════════════════
    # 工具信息
    # ═══════════════════════════════════════════════════════════════

    def get_tool_info(self) -> dict[str, Any]:
        """
        获取工具信息

        Returns:
            工具信息字典
        """
        tools = self.list_tools()

        # 分类
        builtin = ["calculator", "datetime", "random", "weather", "web_search", "file_read", "file_write", "list_dir", "code_execution"]
        rag = ["knowledge_search", "knowledge_add", "knowledge_list", "knowledge_remove"]
        browser = ["browser_navigate", "browser_click", "browser_type", "browser_scroll", "browser_get_text", "browser_screenshot", "browser_wait", "browser_back"]

        return {
            "total_count": len(tools),
            "builtin_tools": [t for t in tools if t in builtin],
            "rag_tools": [t for t in tools if t in rag],
            "browser_tools": [t for t in tools if t in browser],
            "all_tools": tools,
        }

    def __repr__(self) -> str:
        tools = self.list_tools()
        capabilities = []
        if any(t in tools for t in ["knowledge_search"]):
            capabilities.append("RAG")
        if any(t in tools for t in ["browser_navigate"]):
            capabilities.append("Browser")

        cap_str = f" capabilities=[{', '.join(capabilities)}]" if capabilities else ""
        return f"<MainAgent llm={self.llm} tools={len(tools)}{cap_str}>"