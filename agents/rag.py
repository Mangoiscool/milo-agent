"""RAG Agent - 检索增强生成 Agent

结合知识库检索能力，实现基于文档的问答。

现在继承自 BaseAgent，支持工具调用，可以与其他能力组合。
"""

from pathlib import Path
from typing import Any, Optional

from agents.agent_config import AgentConfig
from agents.base import BaseAgent
from core.llm.base import BaseLLM, Message, Role
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.rag import (
    SearchResult,
    create_retriever,
    create_splitter,
)
from core.rag.base import Document
from core.rag.document_loader import create_default_registry
from core.rag.embeddings import BaseEmbedding
from core.rag.text_splitter import SplitConfig
from core.rag.tools import (
    RAGAddDocumentTool,
    RAGListSourcesTool,
    RAGRemoveSourceTool,
    RAGSearchTool,
)
from core.rag.vector_store import KnowledgeBase, VectorStore


class RAGAgent(BaseAgent):
    """
    RAG Agent - 检索增强生成 Agent

    特性：
    - 基于知识库的问答
    - 支持多种文档格式
    - 多知识库管理
    - 增量更新
    - 来源引用
    - 支持工具调用（新增）

    使用示例：
        llm = create_llm("qwen", api_key="sk-xxx")
        embedding = create_embedding("ollama", model="nomic-embed-text")

        agent = RAGAgent(
            llm=llm,
            embedding_model=embedding,
            persist_directory="./knowledge_base"
        )

        # 添加知识
        agent.add_document("guide.pdf")

        # 问答（现在支持工具调用）
        response = agent.chat("什么是 RAG？")
        response = agent.chat_with_tools("帮我查一下相关资料")
    """

    DEFAULT_SYSTEM_PROMPT = """你是一个知识库问答助手。请基于检索到的参考内容回答用户问题。

回答要求：
1. 优先使用参考内容中的信息
2. 如果参考内容中没有相关信息，请明确说明"参考内容中没有相关信息"
3. 引用信息来源，格式为 [来源: 文档名]
4. 回答要准确、简洁、有条理

你可以使用 knowledge_search 工具来检索知识库中的相关信息。

注意：不要编造信息，所有回答都应基于参考内容。"""

    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        memory: Optional[BaseMemory] = None,
        persist_directory: str | Path = "./chroma_db",
        knowledge_base_name: str = "default",
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
        retriever_type: str = "similarity",
        top_k: int = 5,
        splitter_config: Optional[SplitConfig] = None
    ):
        """
        初始化 RAG Agent

        Args:
            llm: LLM 实例
            embedding_model: Embedding 模型
            memory: 记忆系统
            persist_directory: 向量数据库持久化目录
            knowledge_base_name: 知识库名称
            config: Agent 配置
            system_prompt: 自定义系统提示词
            retriever_type: 检索器类型 (similarity, mmr, hybrid)
            top_k: 检索返回的文档数量
            splitter_config: 文本切分配置
        """
        # 保存 RAG 特有属性
        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory)
        self.knowledge_base_name = knowledge_base_name
        self.top_k = top_k

        # 使用默认系统提示词
        effective_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # 初始化基类
        super().__init__(
            llm=llm,
            memory=memory,
            tools=None,  # 稍后注册 RAG 工具
            system_prompt=effective_prompt,
            config=config or AgentConfig()
        )

        # 文档加载器
        self.document_loader = create_default_registry()

        # 文本切分器
        self.splitter = create_splitter(
            "recursive",
            config=splitter_config or SplitConfig()
        )

        # 向量存储
        self.vector_store = VectorStore(
            collection_name=knowledge_base_name,
            persist_directory=self.persist_directory,
            embedding_model=embedding_model
        )

        # 检索器
        self.retriever = create_retriever(
            self.vector_store,
            embedding_model,
            retriever_type
        )

        # 注册 RAG 工具
        self.register_tools([
            RAGSearchTool(self.retriever),
            RAGAddDocumentTool(self.vector_store, self.splitter, self.document_loader),
            RAGListSourcesTool(self.vector_store),
            RAGRemoveSourceTool(self.vector_store),
        ])

        # 对话历史（用于 RAG 特有的上下文构建）
        self.conversation_history: list[Message] = []

    # ═══════════════════════════════════════════════════════════════
    # 知识库管理
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
        path = Path(file_path)
        self.logger.info(f"Loading document: {path}")

        # 1. 加载文档
        documents = self.document_loader.load(path)

        # 2. 添加元数据
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        # 3. 切分文档
        chunks = self.splitter.split_documents(documents)
        self.logger.info(f"Split into {len(chunks)} chunks")

        # 4. 存储到向量数据库
        ids = self.vector_store.add_chunks(chunks)

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
        results = {}
        documents = self.document_loader.load_directory(
            directory,
            recursive=recursive,
            extensions=extensions
        )

        # 按来源分组
        by_source: dict[str, list[Document]] = {}
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

    def remove_document(self, source: str) -> int:
        """
        移除指定来源的文档

        Args:
            source: 文档来源标识

        Returns:
            移除前的文档数量
        """
        count_before = self.vector_store.count()
        self.vector_store.delete(where={"source": source})
        count_after = self.vector_store.count()
        return count_before - count_after

    def clear_knowledge_base(self):
        """清空知识库"""
        self.vector_store.delete_collection()
        self.logger.warning(f"Knowledge base '{self.knowledge_base_name}' cleared")

    def get_document_count(self) -> int:
        """获取知识库中的文档数量"""
        return self.vector_store.count()

    def list_sources(self) -> list[str]:
        """
        列出知识库中的所有文档来源

        Returns:
            来源列表
        """
        # 获取所有文档
        all_docs = self.vector_store.get(limit=10000)

        # 提取来源
        sources = set()
        for doc in all_docs:
            source = doc.get("metadata", {}).get("source", "")
            if source:
                sources.add(source)

        return sorted(list(sources))

    # ═══════════════════════════════════════════════════════════════
    # 检索与问答
    # ═══════════════════════════════════════════════════════════════

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[SearchResult]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        k = top_k or self.top_k
        return self.retriever.retrieve(query, top_k=k)

    def _build_context(self, results: list[SearchResult]) -> str:
        """
        构建上下文字符串

        Args:
            results: 检索结果列表

        Returns:
            格式化的上下文字符串
        """
        if not results:
            return "（无相关参考内容）"

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", "未知来源")
            page = result.metadata.get("page")
            source_info = source
            if page:
                source_info += f", 第{page}页"

            context_parts.append(
                f"[{i}] 来源: {source_info}\n{result.text}"
            )

        return "\n\n".join(context_parts)

    def _format_sources(self, results: list[SearchResult]) -> str:
        """格式化来源引用"""
        sources = []
        for result in results:
            source = result.metadata.get("source", "")
            page = result.metadata.get("page")
            if source:
                ref = source
                if page:
                    ref += f", 第{page}页"
                if ref not in sources:
                    sources.append(ref)
        return "、".join(sources[:3]) if sources else "知识库"

    def chat(self, query: str) -> str:
        """
        RAG 对话（自动检索 + 回答）

        Args:
            query: 用户问题

        Returns:
            回答
        """
        # 1. 检索相关文档
        self.logger.info(f"Retrieving for query: {query[:50]}...")
        results = self.retrieve(query)

        # 2. 构建上下文
        context = self._build_context(results)

        # 3. 构建消息
        messages = []

        # 系统提示词
        messages.append(Message(
            role=Role.SYSTEM,
            content=self.system_prompt
        ))

        # 对话历史
        messages.extend(self.conversation_history)

        # 用户问题（带上下文）
        user_message = f"""参考内容：
{context}

---
用户问题：{query}"""

        messages.append(Message(role=Role.USER, content=user_message))

        # 4. 调用 LLM
        self.logger.info(f"Generating response with {len(results)} references")
        response = self.llm.chat(messages)

        # 5. 更新对话历史
        self.conversation_history.append(Message(role=Role.USER, content=query))
        self.conversation_history.append(Message(role=Role.ASSISTANT, content=response.content))

        # 限制历史长度
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return response.content

    async def achat(self, query: str) -> str:
        """异步 RAG 对话"""
        # 1. 检索
        results = self.retrieve(query)
        context = self._build_context(results)

        # 2. 构建消息
        messages = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        messages.extend(self.conversation_history)

        user_message = f"""参考内容：
{context}

---
用户问题：{query}"""
        messages.append(Message(role=Role.USER, content=user_message))

        # 3. 调用 LLM
        response = await self.llm.achat(messages)

        # 4. 更新历史
        self.conversation_history.append(Message(role=Role.USER, content=query))
        self.conversation_history.append(Message(role=Role.ASSISTANT, content=response.content))

        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return response.content

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        super().clear_history()

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息字典
        """
        return {
            "knowledge_base_name": self.knowledge_base_name,
            "document_count": self.get_document_count(),
            "sources": self.list_sources(),
            "persist_directory": str(self.persist_directory),
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimension": self.embedding_model.dimension,
        }

    def __repr__(self) -> str:
        return f"<RAGAgent llm={self.llm} docs={self.get_document_count()} tools={self.tool_registry.count()}>"


class MultiKnowledgeBaseManager:
    """
    多知识库管理器

    管理多个独立的知识库，支持按领域/项目隔离。
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_model: BaseEmbedding,
        llm: Optional[BaseLLM] = None
    ):
        """
        初始化管理器

        Args:
            persist_directory: 持久化目录
            embedding_model: Embedding 模型
            llm: LLM 实例（可选，用于创建 RAG Agent）
        """
        self.persist_directory = Path(persist_directory)
        self.embedding_model = embedding_model
        self.llm = llm

        self.knowledge_bases: dict[str, KnowledgeBase] = {}
        self.agents: dict[str, RAGAgent] = {}
        self.logger = get_logger(self.__class__.__name__)

    def create_knowledge_base(
        self,
        name: str,
        description: str = ""
    ) -> KnowledgeBase:
        """
        创建知识库

        Args:
            name: 知识库名称
            description: 描述

        Returns:
            KnowledgeBase 实例
        """
        if name in self.knowledge_bases:
            raise ValueError(f"Knowledge base '{name}' already exists")

        kb = KnowledgeBase(
            name=name,
            persist_directory=self.persist_directory,
            embedding_model=self.embedding_model
        )

        self.knowledge_bases[name] = kb
        self.logger.info(f"Created knowledge base: {name}")

        return kb

    def get_knowledge_base(self, name: str) -> Optional[KnowledgeBase]:
        """获取知识库"""
        return self.knowledge_bases.get(name)

    def create_agent(
        self,
        name: str,
        system_prompt: Optional[str] = None
    ) -> RAGAgent:
        """
        创建 RAG Agent

        Args:
            name: 知识库名称
            system_prompt: 自定义系统提示词

        Returns:
            RAGAgent 实例
        """
        if not self.llm:
            raise ValueError("LLM is required to create RAG Agent")

        agent = RAGAgent(
            llm=self.llm,
            embedding_model=self.embedding_model,
            persist_directory=self.persist_directory,
            knowledge_base_name=name,
            system_prompt=system_prompt
        )

        self.agents[name] = agent
        return agent

    def get_agent(self, name: str) -> Optional[RAGAgent]:
        """获取 RAG Agent"""
        return self.agents.get(name)

    def list_knowledge_bases(self) -> list[str]:
        """列出所有知识库"""
        return list(self.knowledge_bases.keys())

    def delete_knowledge_base(self, name: str):
        """删除知识库"""
        if name in self.knowledge_bases:
            kb = self.knowledge_bases[name]
            kb.clear()
            del self.knowledge_bases[name]
            self.logger.info(f"Deleted knowledge base: {name}")

        if name in self.agents:
            del self.agents[name]

    def get_all_stats(self) -> dict[str, dict]:
        """获取所有知识库的统计信息"""
        stats = {}
        for name, kb in self.knowledge_bases.items():
            stats[name] = {
                "document_count": kb.document_count,
            }
        return stats


__all__ = ["RAGAgent", "MultiKnowledgeBaseManager"]