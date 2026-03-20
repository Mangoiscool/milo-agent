from __future__ import annotations
"""检索器

提供多种检索策略：
- 相似度检索
- MMR 多样性检索
- 混合检索（关键词 + 向量）
"""

from abc import ABC, abstractmethod
from typing import Any

from .base import Chunk, SearchResult
from .vector_store import VectorStore


class BaseRetriever(ABC):
    """检索器基类"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """检索相关文档"""
        pass


class SimilarityRetriever(BaseRetriever):
    """
    相似度检索器

    基于向量相似度进行检索。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        filters: dict[str, Any] | None = None
    ):
        """
        Args:
            vector_store: 向量存储
            filters: 默认过滤条件
        """
        self.vector_store = vector_store
        self.filters = filters

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """检索相似文档"""
        results = self.vector_store.query(
            query_text=query,
            n_results=top_k,
            where=self.filters
        )

        search_results = []
        for result in results:
            chunk = Chunk(
                text=result["text"],
                metadata=result["metadata"],
                id=result["id"]
            )
            # ChromaDB 返回的是距离，转换为相似度分数
            # 余弦距离范围是 0-2，越小越相似
            # 转换为 0-1 的相似度分数
            distance = result.get("distance", 0)
            score = 1 - (distance / 2)  # 转换为相似度

            search_results.append(SearchResult(
                chunk=chunk,
                score=score
            ))

        return search_results


class MMRRetriever(BaseRetriever):
    """
    MMR 检索器

    使用 Maximal Marginal Relevance 进行多样性检索，
    平衡相关性和多样性，减少重复内容。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: Any,
        lambda_param: float = 0.5,
        filters: dict[str, Any] | None = None
    ):
        """
        Args:
            vector_store: 向量存储
            embedding_model: Embedding 模型
            lambda_param: MMR 参数，0 表示最大化多样性，1 表示最大化相关性
            filters: 过滤条件
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.lambda_param = lambda_param
        self.filters = filters

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """使用 MMR 检索"""
        # 首先获取更多候选结果
        fetch_k = min(top_k * 4, 100)  # 获取 4 倍的候选
        results = self.vector_store.query(
            query_text=query,
            n_results=fetch_k,
            where=self.filters
        )

        if len(results) <= top_k:
            # 候选不足，直接返回
            return self._to_search_results(results[:top_k])

        # MMR 选择
        query_embedding = self.embedding_model.embed(query)
        selected_indices = self._mmr_select(query_embedding, results, top_k)

        return self._to_search_results([results[i] for i in selected_indices])

    def _mmr_select(
        self,
        query_embedding: list[float],
        results: list[dict],
        top_k: int
    ) -> list[int]:
        """
        MMR 选择算法

        Args:
            query_embedding: 查询向量
            results: 候选结果
            top_k: 选择数量

        Returns:
            选中的索引列表
        """
        import numpy as np

        # 获取候选文档的向量
        candidate_embeddings = []
        for result in results:
            # 需要重新获取向量（ChromaDB 查询结果不包含向量）
            embedding = self.embedding_model.embed(result["text"])
            candidate_embeddings.append(embedding)

        embeddings_matrix = np.array(candidate_embeddings)
        query_vec = np.array(query_embedding)

        # 计算与查询的相似度
        query_similarity = self._cosine_similarity_matrix(
            query_vec.reshape(1, -1),
            embeddings_matrix
        )[0]

        selected = []
        selected_embeddings = []

        for _ in range(top_k):
            if not selected:
                # 第一个选择相似度最高的
                best_idx = int(np.argmax(query_similarity))
            else:
                # MMR 分数
                mmr_scores = []
                selected_matrix = np.array(selected_embeddings)

                for i in range(len(results)):
                    if i in selected:
                        mmr_scores.append(-float("inf"))
                        continue

                    # 相关性分数
                    relevance = query_similarity[i]

                    # 与已选文档的最大相似度
                    if len(selected) > 0:
                        similarity_to_selected = self._cosine_similarity_matrix(
                            embeddings_matrix[i:i + 1],
                            selected_matrix
                        )[0]
                        max_similarity = np.max(similarity_to_selected)
                    else:
                        max_similarity = 0

                    # MMR 分数
                    mmr_score = (
                        self.lambda_param * relevance
                        - (1 - self.lambda_param) * max_similarity
                    )
                    mmr_scores.append(mmr_score)

                best_idx = int(np.argmax(mmr_scores))

            selected.append(best_idx)
            selected_embeddings.append(candidate_embeddings[best_idx])

        return selected

    def _cosine_similarity_matrix(
        self,
        a: "np.ndarray",
        b: "np.ndarray"
    ) -> "np.ndarray":
        """计算余弦相似度矩阵"""
        import numpy as np

        # 归一化
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

        return np.dot(a_norm, b_norm.T)

    def _to_search_results(self, results: list[dict]) -> list[SearchResult]:
        """转换为 SearchResult"""
        search_results = []
        for result in results:
            chunk = Chunk(
                text=result["text"],
                metadata=result["metadata"],
                id=result.get("id")
            )
            distance = result.get("distance", 0)
            score = 1 - (distance / 2)

            search_results.append(SearchResult(chunk=chunk, score=score))
        return search_results


class HybridRetriever(BaseRetriever):
    """
    混合检索器

    结合关键词检索和向量检索，提高召回率。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: Any,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None
    ):
        """
        Args:
            vector_store: 向量存储
            embedding_model: Embedding 模型
            alpha: 向量检索权重 (0-1)，1 表示纯向量检索
            filters: 过滤条件
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.filters = filters

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """混合检索"""
        # 1. 向量检索
        vector_results = self.vector_store.query(
            query_text=query,
            n_results=top_k * 2,  # 获取更多候选
            where=self.filters
        )

        # 2. 关键词检索（简化版，基于文本匹配）
        keyword_results = self._keyword_search(query, top_k * 2)

        # 3. 合并结果
        merged = self._merge_results(vector_results, keyword_results, top_k)

        return merged

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """简单的关键词搜索"""
        # 使用 ChromaDB 的 where_document 过滤
        # 注意：这是一个简化的实现
        # 生产环境建议使用 Elasticsearch 等专业搜索引擎

        keywords = query.lower().split()
        results = []

        # 获取所有文档
        all_docs = self.vector_store.get(limit=1000)

        for doc in all_docs:
            text_lower = doc["text"].lower()
            # 计算关键词匹配分数
            score = sum(1 for kw in keywords if kw in text_lower) / len(keywords)

            if score > 0:
                results.append({
                    **doc,
                    "keyword_score": score
                })

        # 按关键词分数排序
        results.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
        return results[:top_k]

    def _merge_results(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        top_k: int
    ) -> list[SearchResult]:
        """合并向量检索和关键词检索结果"""
        # 构建 ID -> 分数字典
        scores: dict[str, dict[str, float]] = {}

        # 向量分数
        for result in vector_results:
            doc_id = result.get("id", "")
            distance = result.get("distance", 0)
            vector_score = 1 - (distance / 2)

            scores[doc_id] = {
                "vector": vector_score,
                "keyword": 0,
                "text": result["text"],
                "metadata": result["metadata"]
            }

        # 关键词分数
        for result in keyword_results:
            doc_id = result.get("id", "")
            keyword_score = result.get("keyword_score", 0)

            if doc_id in scores:
                scores[doc_id]["keyword"] = keyword_score
            else:
                scores[doc_id] = {
                    "vector": 0,
                    "keyword": keyword_score,
                    "text": result["text"],
                    "metadata": result["metadata"]
                }

        # 计算混合分数
        final_results = []
        for doc_id, score_data in scores.items():
            hybrid_score = (
                self.alpha * score_data["vector"]
                + (1 - self.alpha) * score_data["keyword"]
            )

            chunk = Chunk(
                text=score_data["text"],
                metadata=score_data["metadata"],
                id=doc_id
            )

            final_results.append(SearchResult(
                chunk=chunk,
                score=hybrid_score
            ))

        # 按分数排序
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]


def create_retriever(
    vector_store: VectorStore,
    embedding_model: Any,
    retriever_type: str = "similarity",
    **kwargs
) -> BaseRetriever:
    """
    工厂方法：创建检索器

    Args:
        vector_store: 向量存储
        embedding_model: Embedding 模型
        retriever_type: 检索器类型 (similarity, mmr, hybrid)
        **kwargs: 其他参数

    Returns:
        检索器实例
    """
    if retriever_type == "similarity":
        return SimilarityRetriever(vector_store, **kwargs)
    elif retriever_type == "mmr":
        return MMRRetriever(vector_store, embedding_model, **kwargs)
    elif retriever_type == "hybrid":
        return HybridRetriever(vector_store, embedding_model, **kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")