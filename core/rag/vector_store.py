"""向量存储

使用 ChromaDB 作为向量数据库，支持：
- 文档嵌入和存储
- 相似度检索
- 持久化存储
- 增量更新
"""

from pathlib import Path
from typing import Any

from .base import Chunk


class VectorStore:
    """
    向量存储

    基于 ChromaDB 的向量存储实现。
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str | Path | None = None,
        embedding_model: Any = None
    ):
        """
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录，None 表示内存模式
            embedding_model: Embedding 模型实例
        """
        self._collection_name = collection_name
        self._persist_directory = Path(persist_directory) if persist_directory else None
        self._embedding_model = embedding_model
        self._client = None
        self._collection = None

    def _ensure_client(self):
        """延迟初始化 ChromaDB 客户端"""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                if self._persist_directory:
                    self._persist_directory.mkdir(parents=True, exist_ok=True)
                    self._client = chromadb.PersistentClient(
                        path=str(self._persist_directory),
                        settings=Settings(anonymized_telemetry=False)
                    )
                else:
                    self._client = chromadb.Client(
                        Settings(anonymized_telemetry=False)
                    )
            except ImportError:
                raise ImportError(
                    "chromadb is required. "
                    "Install with: pip install chromadb"
                )
        return self._client

    def _ensure_collection(self):
        """确保集合已初始化"""
        if self._collection is None:
            client = self._ensure_client()

            # 获取或创建集合
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
        return self._collection

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """获取文本的向量"""
        if self._embedding_model is None:
            raise ValueError("Embedding model is required")

        return self._embedding_model.embed_batch(texts)

    def add_chunks(
        self,
        chunks: list[Chunk],
        ids: list[str] | None = None
    ) -> list[str]:
        """
        添加文本块到向量存储

        Args:
            chunks: 文本块列表
            ids: 可选的 ID 列表，不提供则自动生成

        Returns:
            添加的文本块 ID 列表
        """
        collection = self._ensure_collection()

        if not chunks:
            return []

        # 提取文本
        texts = [chunk.text for chunk in chunks]

        # 获取向量
        embeddings = self._get_embeddings(texts)

        # 提取元数据
        metadatas = [chunk.metadata for chunk in chunks]

        # 生成 ID
        if ids is None:
            # 使用现有的文档数量作为基础
            existing_count = collection.count()
            ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]

        # 添加到集合
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # 更新 chunk 的 ID
        for i, chunk in enumerate(chunks):
            chunk.id = ids[i]

        return ids

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None
    ) -> list[str]:
        """
        添加文本到向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表
            ids: ID 列表

        Returns:
            ID 列表
        """
        chunks = [
            Chunk(
                text=text,
                metadata=metadatas[i] if metadatas else {}
            )
            for i, text in enumerate(texts)
        ]
        return self.add_chunks(chunks, ids)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        查询相似文本

        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件

        Returns:
            查询结果列表，每个结果包含 text, metadata, distance
        """
        collection = self._ensure_collection()

        # 获取查询向量
        query_embedding = self._get_embeddings([query_text])[0]

        # 查询
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        # 整理结果
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0
            })

        return formatted_results

    def query_by_embedding(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        通过向量查询

        Args:
            query_embedding: 查询向量
            n_results: 返回结果数量
            where: 元数据过滤条件

        Returns:
            查询结果列表
        """
        collection = self._ensure_collection()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0
            })

        return formatted_results

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None
    ):
        """
        删除文档

        Args:
            ids: 要删除的 ID 列表
            where: 元数据过滤条件
        """
        collection = self._ensure_collection()
        collection.delete(ids=ids, where=where)

    def delete_collection(self):
        """删除整个集合"""
        client = self._ensure_client()
        client.delete_collection(self._collection_name)
        self._collection = None

    def count(self) -> int:
        """获取文档数量"""
        collection = self._ensure_collection()
        return collection.count()

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        获取文档

        Args:
            ids: ID 列表
            where: 元数据过滤条件
            limit: 限制数量

        Returns:
            文档列表
        """
        collection = self._ensure_collection()

        results = collection.get(
            ids=ids,
            where=where,
            limit=limit
        )

        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append({
                "id": results["ids"][i],
                "text": results["documents"][i] if results["documents"] else "",
                "metadata": results["metadatas"][i] if results["metadatas"] else {}
            })

        return formatted_results

    def update(
        self,
        ids: list[str],
        texts: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None
    ):
        """
        更新文档

        Args:
            ids: ID 列表
            texts: 新文本列表
            metadatas: 新元数据列表
        """
        collection = self._ensure_collection()

        update_kwargs: dict[str, Any] = {"ids": ids}

        if texts:
            update_kwargs["documents"] = texts
            update_kwargs["embeddings"] = self._get_embeddings(texts)

        if metadatas:
            update_kwargs["metadatas"] = metadatas

        collection.update(**update_kwargs)

    @property
    def is_persistent(self) -> bool:
        """是否持久化存储"""
        return self._persist_directory is not None


class KnowledgeBase:
    """
    知识库

    封装向量存储，提供更高级的知识库管理功能。
    """

    def __init__(
        self,
        name: str,
        persist_directory: Path | str,
        embedding_model: Any = None
    ):
        """
        Args:
            name: 知识库名称
            persist_directory: 持久化目录
            embedding_model: Embedding 模型
        """
        self.name = name
        self.persist_directory = Path(persist_directory)

        self.vector_store = VectorStore(
            collection_name=name,
            persist_directory=self.persist_directory,
            embedding_model=embedding_model
        )

    def add_documents(self, chunks: list[Chunk]) -> int:
        """
        添加文档

        Returns:
            添加的文档数量
        """
        ids = self.vector_store.add_chunks(chunks)
        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        搜索

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件

        Returns:
            搜索结果
        """
        return self.vector_store.query(
            query_text=query,
            n_results=top_k,
            where=filters
        )

    def delete_by_source(self, source: str):
        """删除指定来源的所有文档"""
        self.vector_store.delete(where={"source": source})

    def clear(self):
        """清空知识库"""
        self.vector_store.delete_collection()

    @property
    def document_count(self) -> int:
        """文档数量"""
        return self.vector_store.count()