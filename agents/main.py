"""MainAgent - 统一的主 Agent

提供完整功能的 Agent：
- 内置工具（默认启用）
- RAG 能力（可选）
- Browser 能力（可选）
- ReAct 推理（可选，Phase 4）
- 长期记忆（可选，Phase 4）

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

Phase 4 示例（启用 ReAct 和长期记忆）：
    from core.llm.factory import create_llm
    from core.rag.embeddings import create_embedding
    from agents.main import MainAgent

    llm = create_llm("qwen", api_key="...")
    embedding = create_embedding("ollama")

    # 启用 ReAct 和长期记忆
    agent = MainAgent(
        llm=llm,
        enable_react=True,
        enable_long_term_memory=True,
        embedding_model=embedding
    )

    # 对话（显示思考过程）
    response = agent.chat_with_tools("北京今天天气如何？", show_reasoning=True)
    # 可以看到 ReAct 的 Thought → Action → Observation 过程
"""

import json
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

# Phase 4: ReAct and Long-term Memory
from core.reasoning.react import ReActTrace, ThoughtStep, ActionStep, ObservationStep
from core.memory.hybrid import HybridMemory
from core.memory.long_term import LongTermMemory


class MainAgent(BaseAgent):
    """
    统一的主 Agent

    特性：
    - 默认启用所有内置工具（计算器、搜索、文件操作等）
    - 可选启用 RAG 能力（知识库检索和自主学习）
    - 可选启用 Browser 能力（网页自动化）
    - 可选启用 ReAct 推理模式（Phase 4：显式思考过程）
    - 可选启用长期记忆（Phase 4：跨会话语义检索）

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

    Phase 4 使用示例：
        # 启用 ReAct 推理（显示思考过程）
        agent = MainAgent(
            llm=llm,
            enable_react=True
        )
        response = agent.chat_with_tools("北京今天气温多少？明天降温5度后呢？", show_reasoning=True)

        # 启用长期记忆（跨会话记忆）
        agent = MainAgent(
            llm=llm,
            enable_long_term_memory=True,
            embedding_model=embedding
        )
        agent.chat_with_tools("我叫 Mango，是一名架构师")  # 记住
        # 新会话...
        agent.chat_with_tools("还记得我是谁吗？")  # 能检索到之前的记忆

        # 同时启用 ReAct + 长期记忆
        agent = MainAgent(
            llm=llm,
            enable_react=True,
            enable_long_term_memory=True,
            embedding_model=embedding
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
4. 完成任务后给出清晰的总结

工具使用优先级：
- 当用户询问当前浏览器页面的内容时，必须使用 browser_get_text 获取页面文本，然后直接基于获取的内容回答用户问题
- 当用户已经在浏览器中执行了搜索操作，页面显示了搜索结果，此时应该用 browser_get_text 读取当前页面内容，而不是调用 web_search
- web_search 工具用于从互联网搜索新信息，而不是读取当前浏览器已打开的页面
- 如果已通过浏览器工具获取了页面内容（文本或截图），应该直接分析并回答用户问题，不要重复调用其他搜索工具

Phase 4 能力（如果启用）：
- ReAct 模式：会显示思考过程（Thought → Action → Observation → Final Answer）
- 长期记忆：会记住跨会话的重要信息，并在需要时检索"""

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
        # Phase 4: ReAct and Long-term Memory
        # ═══════════════════════════════════════════════════════════════
        enable_react: bool = False,
        enable_long_term_memory: bool = False,
        max_react_iterations: int = 10,
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
        # ═══════════════════════════════════════════════════════════════
        # Long-term Memory 配置
        # ═══════════════════════════════════════════════════════════════
        memory_persist_directory: Optional[str] = None,
        memory_session_id: Optional[str] = None,
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
            enable_react: 是否启用 ReAct 推理模式（默认 False）
            enable_long_term_memory: 是否启用长期记忆（默认 False）
            max_react_iterations: ReAct 最大迭代次数
            embedding_model: Embedding 模型（启用 RAG 或长期记忆时必需）
            persist_directory: 知识库持久化目录
            knowledge_base_name: 知识库名称
            retriever_type: 检索器类型
            splitter_config: 文本切分配置
            browser_config: 浏览器配置
            memory_persist_directory: 长期记忆持久化目录
            memory_session_id: 记忆会话 ID
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
        self.enable_react = enable_react
        self.enable_long_term_memory = enable_long_term_memory
        self.max_react_iterations = max_react_iterations
        self.knowledge_base_name = knowledge_base_name
        self.persist_directory = persist_directory or str(DEFAULT_PERSIST_DIR)

        # ═══════════════════════════════════════════════════════════════
        # Phase 4: 设置长期记忆
        # ═══════════════════════════════════════════════════════════════
        if enable_long_term_memory:
            if embedding_model is None:
                raise ValueError("embedding_model is required when enable_long_term_memory=True")
            self._setup_long_term_memory(
                embedding_model=embedding_model,
                persist_directory=memory_persist_directory,
                session_id=memory_session_id
            )

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
    # Phase 4: 长期记忆设置
    # ═══════════════════════════════════════════════════════════════

    def _setup_long_term_memory(
        self,
        embedding_model: BaseEmbedding,
        persist_directory: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """
        设置长期记忆

        Args:
            embedding_model: Embedding 模型
            persist_directory: 持久化目录
            session_id: 会话 ID
        """
        from core.memory.short_term import ShortTermMemory

        self.embedding_model = embedding_model

        # 创建长期记忆
        memory_persist_dir = persist_directory or str(PROJECT_ROOT / "workspace" / "long_term_memory")
        long_term = LongTermMemory(
            embedding_model=embedding_model,
            session_id=session_id,
            persist_directory=memory_persist_dir
        )

        # 创建短期记忆（如果外部没有提供）
        if self.memory is None or isinstance(self.memory, ShortTermMemory):
            short_term = ShortTermMemory(
                max_messages=self.config.max_memory_messages,
                use_intelligent_pruning=self.config.use_intelligent_pruning
            )
        else:
            # 如果外部提供了记忆，使用它作为短期记忆
            short_term = self.memory

        # 创建混合记忆
        self.memory = HybridMemory(
            short_term=short_term,
            long_term=long_term
        )

        self.logger.info("Long-term memory enabled with HybridMemory")

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: ReAct 对话
    # ═══════════════════════════════════════════════════════════════

    def chat_with_tools(self, user_input: str, show_reasoning: bool = False) -> str:
        """
        对话（支持工具调用）

        Phase 4: 如果启用了 ReAct，使用 ReAct 推理循环
        否则使用标准工具调用循环

        Args:
            user_input: 用户输入
            show_reasoning: 是否显示 ReAct 思考过程（仅 ReAct 模式有效）

        Returns:
            Agent 响应
        """
        if self.enable_react:
            return self._react_chat(user_input, show_reasoning)
        else:
            return super().chat_with_tools(user_input)

    def _react_chat(self, user_input: str, show_reasoning: bool = False) -> str:
        """
        ReAct 对话循环

        实现 Thought → Action → Observation → Final Answer 循环
        """
        from core.llm.base import Role

        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="react")
        self.logger.info(f"User input (ReAct): {user_input[:100]}...")

        # 初始化 ReAct 轨迹
        trace = ReActTrace(steps=[])

        # 添加用户消息到记忆
        from core.llm.base import Message
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 获取工具定义
        tools = None
        if self.tool_registry.count() > 0:
            tools = self.tool_registry.get_all_definitions()

        # ReAct 循环
        for iteration in range(self.max_react_iterations):
            # 构建 ReAct Prompt
            messages = self._build_react_messages(user_input, trace)

            # 调用 LLM
            response = self.llm.chat_with_tools(messages, tools=tools)
            content = response.content

            self.logger.debug(f"ReAct iteration {iteration + 1}: {content[:200]}...")

            # 解析响应
            thought, action_name, action_input = self._parse_react_response(content)

            # 记录 Thought
            if thought:
                trace.steps.append(ThoughtStep(content=thought))
                self.logger.debug(f"Thought: {thought[:100]}...")

            # 检查是否完成
            if "Final Answer:" in content or not action_name:
                final_answer = content.split("Final Answer:")[-1].strip() if "Final Answer:" in content else content.strip()

                # 添加到记忆
                self.memory.add(Message(role=Role.ASSISTANT, content=final_answer))

                self.logger.info(f"ReAct final answer: {final_answer[:100]}...")
                self._emit(AgentEvent.AFTER_CHAT, response=final_answer, mode="react")

                if show_reasoning:
                    return f"{trace.to_prompt()}\n\nFinal Answer: {final_answer}"
                return final_answer

            # 执行 Action
            if action_name and action_name in self.tool_registry.list_tools():
                action_step = ActionStep(
                    tool_name=action_name,
                    arguments=action_input or {},
                    thought=ThoughtStep(content=thought) if thought else ThoughtStep(content="")
                )
                trace.steps.append(action_step)

                self.logger.info(f"Action: {action_name}({action_input})")
                self._emit(AgentEvent.TOOL_CALL, name=action_name, arguments=action_input)

                # 执行工具
                result = self.tool_registry.execute(action_name, **action_input)

                self.logger.info(f"Observation: {result.content[:100] if result.content else 'empty'}...")
                self._emit(AgentEvent.TOOL_RESULT, name=action_name, result=result.content, is_error=result.is_error)

                # 记录 Observation
                obs_step = ObservationStep(
                    result=result.content if not result.is_error else f"Error: {result.error_message}",
                    is_error=result.is_error,
                    action=action_step
                )
                trace.steps.append(obs_step)

                # 将工具结果添加到记忆（供下一轮使用）
                self.memory.add(Message(
                    role=Role.TOOL,
                    content=result.content if not result.is_error else f"Error: {result.error_message}",
                    name=action_name
                ))

        # 超过最大迭代次数
        error_msg = "抱歉，思考过程太长，请简化问题。"
        self.memory.add(Message(role=Role.ASSISTANT, content=error_msg))
        self._emit(AgentEvent.AFTER_CHAT, response=error_msg, mode="react")
        return error_msg

    def _build_react_messages(self, question: str, trace: ReActTrace) -> list:
        """构建 ReAct Prompt"""
        from core.llm.base import Message, Role

        messages = []

        # 添加系统提示
        system_content = self.system_prompt or self.DEFAULT_SYSTEM_PROMPT
        system_content += "\n\n" + self._get_react_instructions()
        messages.append(Message(role=Role.SYSTEM, content=system_content))

        # 添加工具描述
        if self.tool_registry.count() > 0:
            tools_desc = self._format_tools_for_react()
            messages.append(Message(role=Role.SYSTEM, content=f"可用工具：\n{tools_desc}"))

        # 添加历史执行轨迹
        if trace.steps:
            trace_prompt = trace.to_prompt()
            messages.append(Message(role=Role.USER, content=f"之前的执行过程：\n{trace_prompt}\n\n继续回答：{question}"))
        else:
            messages.append(Message(role=Role.USER, content=question))

        return messages

    def _get_react_instructions(self) -> str:
        """获取 ReAct 指令"""
        return """请按照以下 ReAct 格式思考和行动：

1. Thought: 分析当前情况，思考下一步行动
2. Action: 选择要使用的工具（如果需要）
3. Action Input: 提供工具参数（JSON 格式）
4. Observation: 观察工具返回的结果
5. 重复 Thought-Action-Observation 直到问题解决
6. Final Answer: 给出最终答案

格式示例：
Thought: 用户询问北京天气，我需要查询天气信息。
Action: weather
Action Input: {"city": "北京"}
Observation: {"temperature": 25, "condition": "晴天"}
Thought: 已经获取天气信息，可以回答用户了。
Final Answer: 北京今天晴天，气温25°C。"""

    def _format_tools_for_react(self) -> str:
        """格式化工具描述供 ReAct 使用"""
        definitions = self.tool_registry.get_all_definitions()
        lines = []
        for d in definitions:
            lines.append(f"- {d.name}: {d.description}")
            if d.parameters:
                lines.append(f"  参数: {d.parameters}")
        return "\n".join(lines)

    def _parse_react_response(self, response: str) -> tuple:
        """解析 ReAct 响应，提取 Thought、Action 和 Action Input"""
        import re

        thought = ""
        action_name = None
        action_input = {}

        # 解析 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?:\nAction:|\nFinal Answer:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 解析 Action
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            action_name = action_match.group(1).strip()

        # 解析 Action Input
        input_match = re.search(r'Action Input:\s*(\{[^}]*\}|[^\n]+)', response)
        if input_match:
            input_str = input_match.group(1).strip()
            try:
                action_input = json.loads(input_str)
            except json.JSONDecodeError:
                # 如果不是 JSON，作为字符串参数
                action_input = {"query": input_str}

        return thought, action_name, action_input

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

    def get_capabilities(self) -> dict[str, Any]:
        """
        获取所有能力状态

        Returns:
            能力状态字典
        """
        return {
            "rag": self.enable_rag,
            "browser": self.enable_browser,
            "react": self.enable_react,
            "long_term_memory": self.enable_long_term_memory,
        }

    def __repr__(self) -> str:
        tools = self.list_tools()
        capabilities = []
        if self.enable_rag:
            capabilities.append("RAG")
        if self.enable_browser:
            capabilities.append("Browser")
        if self.enable_react:
            capabilities.append("ReAct")
        if self.enable_long_term_memory:
            capabilities.append("LongTermMemory")

        cap_str = f" capabilities=[{', '.join(capabilities)}]" if capabilities else ""
        return f"<MainAgent llm={self.llm} tools={len(tools)}{cap_str}>"