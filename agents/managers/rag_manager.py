"""
RAGManager - RAG 知识库管理器

管理知识库的文档加载、切分、存储和检索。
"""

from pathlib import Path
from typing import Any, Optional, Union

from core.logger import get_logger
from core.rag import create_retriever, create_splitter
from core.rag.document_loader import create_default_registry
from core.rag.embeddings import BaseEmbedding
from core.rag.text_splitter import SplitConfig
from core.rag.vector_store import KnowledgeBase


class RAGManager:
    """
    RAG 知识库管理器

    封装知识库的所有操作：
    - 文档加载和切分
    - 向量存储管理
    - 文档增删改查

    使用示例:
        embedding = create_embedding("ollama")
        rag = RAGManager(
            embedding_model=embedding,
            knowledge_base_name="my_kb",
            persist_directory="./workspace/kb"
        )

        # 添加文档
        rag.add_document("guide.pdf")

        # 添加文本
        rag.add_text("这是一些知识内容", source="manual")

        # 获取统计
        stats = rag.get_stats()
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        knowledge_base_name: str = "milo_kb",
        persist_directory: Optional[str] = None,
        retriever_type: str = "similarity",
        splitter_config: Optional[SplitConfig] = None
    ):
        """
        初始化 RAG 管理器

        Args:
            embedding_model: Embedding 模型（必需）
            knowledge_base_name: 知识库名称
            persist_directory: 持久化目录
            retriever_type: 检索器类型
            splitter_config: 文本切分配置
        """
        self.embedding_model = embedding_model
        self.knowledge_base_name = knowledge_base_name
        self.persist_directory = persist_directory
        self.retriever_type = retriever_type

        self.logger = get_logger(self.__class__.__name__)

        # 初始化知识库（自动使用默认路径）
        self.knowledge_base = KnowledgeBase(
            name=knowledge_base_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        # 保持兼容：对外暴露为 vector_store
        self.vector_store = self.knowledge_base

        # 初始化文本切分器
        self.splitter = create_splitter(
            "recursive",
            config=splitter_config or SplitConfig()
        )

        # 初始化文档加载器
        self.document_loader = create_default_registry()

        # 初始化检索器
        self.retriever = create_retriever(
            self.vector_store,
            embedding_model,
            retriever_type
        )

        self.logger.info(f"RAGManager initialized: {knowledge_base_name}")

    def add_document(
        self,
        file_path: Union[str, Path],
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
        directory: Union[str, Path],
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
        count_before = self.vector_store.count()
        self.vector_store.delete(where={"source": source})
        count_after = self.vector_store.count()

        removed = count_before - count_after
        self.logger.info(f"Removed {removed} chunks from source: {source}")
        return removed

    def search(self, query: str, top_k: int = 5) -> list:
        """
        搜索知识库

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            匹配的文档块列表
        """
        return self.retriever.retrieve(query, top_k=top_k)

    def get_stats(self) -> dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息字典
        """
        return {
            "knowledge_base_name": self.knowledge_base_name,
            "document_count": self.vector_store.count(),
            "sources": self.list_sources(),
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model.model_name if hasattr(self.embedding_model, 'model_name') else str(self.embedding_model),
        }

    def clear(self) -> int:
        """
        清空知识库

        Returns:
            删除的文档数量
        """
        count = self.vector_store.count()
        self.vector_store.clear()
        self.logger.info(f"Cleared {count} documents from knowledge base")
        return count
