"""
MiloAgent - 统一的主 Agent

使用组合模式封装各种能力：
- ToolManager: 工具管理
- RAGManager: 知识库（可选）
- BrowserManager: 浏览器自动化（可选）
- ReActReasoner: 统一推理引擎

记忆系统统一使用 HybridMemory，长期记忆可选。
"""

from pathlib import Path
from typing import Any, Optional, Union

from agents.agent_config import AgentConfig
from agents.base import AgentEvent, BaseAgent
from agents.managers import BrowserManager, RAGManager, ReActReasoner, ToolManager
from core.browser import BrowserConfig
from core.llm.base import BaseLLM
from core.logger import get_logger
from core.memory.hybrid import HybridMemory
from core.memory.long_term import LongTermMemory
from core.memory.short_term import ShortTermMemory
from core.rag.embeddings import BaseEmbedding
from core.rag.text_splitter import SplitConfig


class MiloAgent(BaseAgent):
    """
    统一的主 Agent

    使用组合模式封装各种能力，通过可选的管理器实现功能扩展。

    使用示例:
        from core.llm.factory import LLMFactory
        from core.rag.embeddings import create_embedding

        llm = LLMFactory.create("qwen", api_key="...")

        # 基础用法（仅内置工具）
        agent = MiloAgent(llm=llm)

        # 启用 RAG
        embedding = create_embedding("ollama")
        agent = MiloAgent(llm=llm, embedding_model=embedding)
        agent.add_document("guide.pdf")

        # 启用 Browser
        agent = MiloAgent(llm=llm, enable_browser=True)
        await agent.initialize()

        # 完整功能
        agent = MiloAgent(
            llm=llm,
            embedding_model=embedding,
            enable_browser=True
        )

        # 对话
        response = agent.chat("帮我查一下今天的天气")
    """

    DEFAULT_SYSTEM_PROMPT = """你是 Milo，一个智能 AI 助手。你的目标是通过思考、使用工具和执行多步骤推理来帮助用户高效完成任务。

## 核心能力

### 1. 推理与规划（ReAct 模式）
遵循 Thought → Action → Observation 循环：
- **Thought**: 分析当前状态，规划下一步行动
- **Action**: 选择合适的工具执行
- **Observation**: 评估结果，决定继续或完成

### 2. 工具使用
根据需求选择最佳工具：

**内置工具**（始终可用）：
- `calculator` - 数学计算、表达式求值
- `datetime` - 获取当前时间、日期计算
- `web_search` - 从互联网搜索最新信息
- `file_ops` - 读写本地文件
- `code_execute` - 执行 Python 代码

**知识库检索**（RAG 启用时）：
- `knowledge_base_search` - 检索用户上传的文档
- `add_document` / `add_text` - 向知识库添加内容
- 优先使用知识库回答用户关于已上传文档的问题

**浏览器自动化**（Browser 启用时）：
- `browser_navigate` - 打开指定 URL
- `browser_click` / `browser_input` - 与页面交互
- `browser_get_text` - 提取页面文本内容
- `browser_screenshot` - 截取页面截图

### 3. 记忆系统
- **短期记忆**: 维护对话上下文，保持连贯性
- **长期记忆**: 存储重要信息（如用户偏好、历史结论），跨会话保留
- 主动使用记忆：在回答前检查是否有相关历史信息

## 工作原则

1. **先思考后行动**：复杂任务先制定计划，再逐步执行
2. **工具组合**：单工具不足时，组合多个工具（如搜索→浏览→提取）
3. **信息优先级**：
   - 知识库中的文档 > 网络搜索
   - 当前浏览器页面内容（browser_get_text）> 新搜索
4. **结果验证**：检查工具输出，如有错误尝试修正或换工具
5. **清晰总结**：完成任务后用简洁语言总结关键结果

## 工具使用指南

**搜索 vs 浏览**：
- 需要*新信息* → 使用 `web_search`
- 用户已在浏览器打开页面 → 使用 `browser_get_text` 读取
- 需要与网页交互（点击、输入）→ 使用浏览器工具链

**知识库查询**：
- 用户询问上传的文档 → 优先用 `knowledge_base_search`
- 文档未找到答案 → 可结合网络搜索补充

**计算与代码**：
- 简单计算 → `calculator`
- 复杂数据处理、图表生成 → `code_execute`"""

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        max_iterations: int = 10,
        # 能力开关
        enable_builtin_tools: bool = True,
        embedding_model: Optional[BaseEmbedding] = None,
        enable_browser: bool = False,
        browser_config: Optional[BrowserConfig] = None,
        # RAG 配置
        knowledge_base_name: str = "milo_kb",
        persist_directory: Optional[str] = None,
        retriever_type: str = "similarity",
        splitter_config: Optional[SplitConfig] = None,
        # 记忆配置
        session_id: Optional[str] = None,
        memory_persist_directory: Optional[str] = None,
    ):
        """
        初始化 MiloAgent

        Args:
            llm: LLM 实例（必需）
            memory: 自定义记忆系统（可选，默认使用 HybridMemory）
            system_prompt: 系统提示词
            config: Agent 配置
            max_iterations: 最大推理迭代次数
            enable_builtin_tools: 是否启用内置工具
            embedding_model: Embedding 模型（启用 RAG/长期记忆时必需）
            enable_browser: 是否启用浏览器
            browser_config: 浏览器配置
            knowledge_base_name: 知识库名称
            persist_directory: 知识库持久化目录
            retriever_type: 检索器类型
            splitter_config: 文本切分配置
            session_id: 会话 ID
            memory_persist_directory: 长期记忆持久化目录
        """
        self.logger = get_logger(self.__class__.__name__)
        effective_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        effective_config = config or AgentConfig()

        # 确定是否启用 RAG
        enable_rag = embedding_model is not None

        # 设置记忆系统（HybridMemory，长期记忆可选）
        if memory is None:
            memory = self._create_hybrid_memory(
                session_id=session_id,
                embedding_model=embedding_model if enable_rag else None,
                memory_persist_directory=memory_persist_directory,
                max_messages=effective_config.max_memory_messages
            )

        # 初始化基类
        super().__init__(
            llm=llm,
            memory=memory,
            tools=None,
            system_prompt=effective_prompt,
            config=effective_config,
            max_tool_iterations=max_iterations
        )

        # 初始化工具管理器
        self.tool_manager = ToolManager()

        # 注册内置工具
        if enable_builtin_tools:
            self.tool_manager.register_builtin_tools()

        # 初始化 RAG 管理器（如果提供了 embedding_model）
        self.rag_manager: Optional[RAGManager] = None
        if enable_rag:
            self.rag_manager = RAGManager(
                embedding_model=embedding_model,
                knowledge_base_name=knowledge_base_name,
                persist_directory=persist_directory,
                retriever_type=retriever_type,
                splitter_config=splitter_config
            )
            # 注册 RAG 工具
            self.tool_manager.register_rag_tools(
                retriever=self.rag_manager.retriever,
                vector_store=self.rag_manager.knowledge_base,
                splitter=self.rag_manager.splitter,
                document_loader=self.rag_manager.document_loader
            )

        # 初始化浏览器管理器
        self.browser_manager: Optional[BrowserManager] = None
        if enable_browser:
            self.browser_manager = BrowserManager(config=browser_config)
            self.tool_manager.register_browser_tools(
                browser_controller=self.browser_manager.controller
            )

        # 初始化 ReAct 推理引擎
        self.reasoner = ReActReasoner(
            llm=llm,
            memory=self.memory,
            tool_registry=self.tool_manager.registry,
            system_prompt=effective_prompt,
            max_iterations=max_iterations
        )

        self.logger.info(
            f"MiloAgent initialized: "
            f"tools={self.tool_manager.count()}, "
            f"rag={'on' if enable_rag else 'off'}, "
            f"browser={'on' if enable_browser else 'off'}"
        )

    def _create_hybrid_memory(
        self,
        session_id: Optional[str],
        embedding_model: Optional[BaseEmbedding],
        memory_persist_directory: Optional[str],
        max_messages: int = 20
    ) -> HybridMemory:
        """
        创建 HybridMemory，长期记忆可选

        Args:
            session_id: 会话 ID
            embedding_model: Embedding 模型（用于长期记忆）
            memory_persist_directory: 持久化目录
            max_messages: 短期记忆最大消息数

        Returns:
            HybridMemory 实例
        """
        # 短期记忆（启用持久化）
        short_term = ShortTermMemory(
            max_messages=max_messages,
            use_intelligent_pruning=True,
            persist=True,
            session_id=session_id
        )

        # 长期记忆（如果提供了 embedding_model）
        long_term = None
        if embedding_model:
            long_term = LongTermMemory(
                embedding_model=embedding_model,
                session_id=session_id,
                persist_directory=memory_persist_directory
            )
            self.logger.info("Long-term memory enabled")

        return HybridMemory(
            session_id=session_id,
            short_term=short_term,
            long_term=long_term
        )

    # ------------------------------------------------------------------
    # 对话接口（统一使用 ReAct）
    # ------------------------------------------------------------------

    def chat(self, user_input: str, show_reasoning: bool = False) -> str:
        """
        同步对话（统一使用 ReAct 推理）

        Args:
            user_input: 用户输入
            show_reasoning: 是否显示推理过程

        Returns:
            Agent 响应
        """
        return self.reasoner.run(
            user_input=user_input,
            show_reasoning=show_reasoning,
            event_emitter=self._emit
        )

    def chat_with_tools(self, user_input: str, show_reasoning: bool = False) -> str:
        """兼容旧接口，统一使用 ReAct"""
        return self.chat(user_input, show_reasoning)

    # ------------------------------------------------------------------
    # 知识库管理 API（代理到 RAGManager）
    # ------------------------------------------------------------------

    def add_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[dict[str, Any]] = None
    ) -> int:
        """添加文档到知识库"""
        if not self.rag_manager:
            raise RuntimeError("RAG not enabled. Provide embedding_model to enable.")
        return self.rag_manager.add_document(file_path, metadata)

    def add_text(
        self,
        text: str,
        source: str = "user_input",
        metadata: Optional[dict[str, Any]] = None
    ) -> int:
        """添加文本到知识库"""
        if not self.rag_manager:
            raise RuntimeError("RAG not enabled. Provide embedding_model to enable.")
        return self.rag_manager.add_text(text, source, metadata)

    def add_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        extensions: Optional[list[str]] = None
    ) -> dict[str, int]:
        """批量添加目录下的文档"""
        if not self.rag_manager:
            raise RuntimeError("RAG not enabled. Provide embedding_model to enable.")
        return self.rag_manager.add_directory(directory, recursive, extensions)

    def list_sources(self) -> list[str]:
        """列出知识库中的所有文档来源"""
        if not self.rag_manager:
            raise RuntimeError("RAG not enabled. Provide embedding_model to enable.")
        return self.rag_manager.list_sources()

    def remove_document(self, source: str) -> int:
        """移除指定来源的文档"""
        if not self.rag_manager:
            raise RuntimeError("RAG not enabled. Provide embedding_model to enable.")
        return self.rag_manager.remove_document(source)

    def get_knowledge_base_stats(self) -> dict[str, Any]:
        """获取知识库统计信息"""
        if not self.rag_manager:
            return {"enabled": False}
        return {"enabled": True, **self.rag_manager.get_stats()}

    # ------------------------------------------------------------------
    # 浏览器管理（代理到 BrowserManager）
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """初始化异步资源（浏览器）"""
        if self.browser_manager:
            await self.browser_manager.initialize()

    async def close(self) -> None:
        """清理资源（浏览器）"""
        if self.browser_manager:
            await self.browser_manager.close()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_tool_info(self) -> dict[str, Any]:
        """获取工具信息"""
        return self.tool_manager.get_tool_info()

    def get_capabilities(self) -> dict[str, Any]:
        """获取所有能力状态"""
        return {
            "rag": self.rag_manager is not None,
            "browser": self.browser_manager is not None,
            "long_term_memory": self.memory.long_term is not None if hasattr(self.memory, 'long_term') else False,
        }

    def __repr__(self) -> str:
        caps = []
        if self.rag_manager:
            caps.append("RAG")
        if self.browser_manager:
            caps.append("Browser")
        if hasattr(self.memory, 'long_term') and self.memory.long_term:
            caps.append("LongTermMemory")

        cap_str = f" [{', '.join(caps)}]" if caps else ""
        return f"<MiloAgent tools={self.tool_manager.count()}{cap_str}>"
