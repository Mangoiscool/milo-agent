"""RAG 工具集

提供可被 Agent 调用的 RAG 相关工具：
- RAGSearchTool: 检索知识库
- RAGAddDocumentTool: 添加文档到知识库
- RAGListSourcesTool: 列出知识库中的文档来源
"""

from pathlib import Path
from typing import Any, Optional

from core.tools.base import BaseTool, ToolResult
from core.logger import get_logger

from .base import Document
from .document_loader import DocumentLoaderRegistry, create_default_registry
from .retriever import BaseRetriever
from .text_splitter import BaseTextSplitter, SplitConfig, create_splitter
from .vector_store import VectorStore


class RAGSearchTool(BaseTool):
    """
    RAG 检索工具

    让 Agent 可以从知识库中检索相关信息。
    """

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return """从知识库中检索相关信息。

适用场景：
- 需要查找文档、资料或专业知识
- 用户询问公司政策、产品说明、技术文档等
- 需要引用已有资料回答问题

参数说明：
- query: 检索查询，描述要查找的内容
- top_k: 返回的相关文档数量（默认5个）"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询，描述要查找的内容"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的相关文档数量（默认5个）",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    def __init__(self, retriever: BaseRetriever):
        """
        初始化检索工具

        Args:
            retriever: 检索器实例
        """
        self.retriever = retriever
        self.logger = get_logger(self.__class__.__name__)

    def execute(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """
        执行检索

        Args:
            query: 检索查询
            top_k: 返回数量

        Returns:
            检索结果
        """
        try:
            self.logger.info(f"Searching knowledge base: {query[:50]}...")

            # 检索
            results = self.retriever.retrieve(query, top_k=top_k)

            if not results:
                return ToolResult(
                    content="知识库中未找到相关信息。",
                    is_error=False
                )

            # 格式化结果
            formatted = self._format_results(results)

            self.logger.info(f"Found {len(results)} results")
            return ToolResult(content=formatted)

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"检索失败: {str(e)}"
            )

    def _format_results(self, results: list) -> str:
        """格式化检索结果"""
        parts = []

        for i, result in enumerate(results, 1):
            source = result.chunk.metadata.get("source", "未知来源")
            page = result.chunk.metadata.get("page")

            source_info = source
            if page:
                source_info += f", 第{page}页"

            parts.append(
                f"[{i}] 来源: {source_info} (相关度: {result.score:.2f})\n"
                f"{result.chunk.text}"
            )

        return "\n\n---\n\n".join(parts)


class RAGAddDocumentTool(BaseTool):
    """
    RAG 添加文档工具

    让 Agent 可以自主学习，将新知识添加到知识库。
    """

    @property
    def name(self) -> str:
        return "knowledge_add"

    @property
    def description(self) -> str:
        return """将文档或文本添加到知识库中，用于后续检索。

适用场景：
- 用户要求"记住这个文档"
- 用户要求"学习这个内容"
- 发现需要保存的重要信息

参数说明：
- file_path: 文档文件路径（可选）
- text: 直接添加的文本内容（可选）
- source: 来源标识（可选）

注意：file_path 和 text 至少提供一个。"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文档文件路径（如 /path/to/document.pdf）"
                },
                "text": {
                    "type": "string",
                    "description": "直接添加的文本内容"
                },
                "source": {
                    "type": "string",
                    "description": "来源标识（默认为 'agent_added'）"
                }
            },
            "required": []  # file_path 或 text 至少一个
        }

    def __init__(
        self,
        vector_store: VectorStore,
        splitter: Optional[BaseTextSplitter] = None,
        document_loader: Optional[DocumentLoaderRegistry] = None
    ):
        """
        初始化添加文档工具

        Args:
            vector_store: 向量存储
            splitter: 文本切分器
            document_loader: 文档加载器
        """
        self.vector_store = vector_store
        self.splitter = splitter or create_splitter("recursive", config=SplitConfig())
        self.document_loader = document_loader or create_default_registry()
        self.logger = get_logger(self.__class__.__name__)

    def execute(
        self,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        source: str = "agent_added",
        **kwargs
    ) -> ToolResult:
        """
        执行添加文档

        Args:
            file_path: 文档路径
            text: 文本内容
            source: 来源标识

        Returns:
            添加结果
        """
        try:
            if not file_path and not text:
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message="请提供 file_path 或 text 参数"
                )

            chunks_added = 0

            # 从文件添加
            if file_path:
                path = Path(file_path)
                if not path.exists():
                    return ToolResult(
                        content="",
                        is_error=True,
                        error_message=f"文件不存在: {file_path}"
                    )

                self.logger.info(f"Loading document: {file_path}")

                # 加载文档
                documents = self.document_loader.load(path)

                # 切分
                chunks = self.splitter.split_documents(documents)

                # 添加元数据
                for chunk in chunks:
                    chunk.metadata["added_by"] = "agent"

                # 存储
                ids = self.vector_store.add_chunks(chunks)
                chunks_added = len(ids)

                self.logger.info(f"Added {chunks_added} chunks from file")

            # 从文本添加
            elif text:
                self.logger.info(f"Adding text: {text[:50]}...")

                # 创建文档
                doc = Document.from_text(text, source=source)

                # 切分
                chunks = self.splitter.split_document(doc)

                # 添加元数据
                for chunk in chunks:
                    chunk.metadata["added_by"] = "agent"

                # 存储
                ids = self.vector_store.add_chunks(chunks)
                chunks_added = len(ids)

                self.logger.info(f"Added {chunks_added} chunks from text")

            return ToolResult(
                content=f"成功添加到知识库：{chunks_added} 个文本片段。"
            )

        except Exception as e:
            self.logger.error(f"Add document failed: {e}")
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"添加失败: {str(e)}"
            )


class RAGListSourcesTool(BaseTool):
    """
    RAG 列出文档来源工具

    列出知识库中已有的所有文档来源。
    """

    @property
    def name(self) -> str:
        return "knowledge_list"

    @property
    def description(self) -> str:
        return """列出知识库中已有的所有文档来源。

适用场景：
- 用户想知道知识库中有哪些资料
- 确认文档是否已添加到知识库
- 了解知识库的内容覆盖范围"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {}
        }

    def __init__(self, vector_store: VectorStore):
        """
        初始化列出来源工具

        Args:
            vector_store: 向量存储
        """
        self.vector_store = vector_store
        self.logger = get_logger(self.__class__.__name__)

    def execute(self, **kwargs) -> ToolResult:
        """
        列出所有文档来源

        Returns:
            文档来源列表
        """
        try:
            # 获取所有文档
            all_docs = self.vector_store.get(limit=10000)

            if not all_docs:
                return ToolResult(content="知识库为空。")

            # 提取来源
            sources = {}
            for doc in all_docs:
                source = doc.get("metadata", {}).get("source", "未知")
                if source not in sources:
                    sources[source] = 0
                sources[source] += 1

            # 格式化输出
            lines = ["知识库中的文档来源：\n"]
            for source, count in sorted(sources.items()):
                lines.append(f"- {source}: {count} 个片段")

            lines.append(f"\n总计: {len(sources)} 个来源, {len(all_docs)} 个片段")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            self.logger.error(f"List sources failed: {e}")
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"列出来源失败: {str(e)}"
            )


class RAGRemoveSourceTool(BaseTool):
    """
    RAG 移除文档来源工具

    从知识库中移除指定来源的文档。
    """

    @property
    def name(self) -> str:
        return "knowledge_remove"

    @property
    def description(self) -> str:
        return """从知识库中移除指定来源的文档。

适用场景：
- 用户要求删除某个文档
- 文档内容过时需要更新
- 清理不需要的知识

注意：此操作不可逆，请谨慎使用。"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "要移除的文档来源标识"
                }
            },
            "required": ["source"]
        }

    def __init__(self, vector_store: VectorStore):
        """
        初始化移除来源工具

        Args:
            vector_store: 向量存储
        """
        self.vector_store = vector_store
        self.logger = get_logger(self.__class__.__name__)

    def execute(self, source: str, **kwargs) -> ToolResult:
        """
        移除文档

        Args:
            source: 文档来源标识

        Returns:
            移除结果
        """
        try:
            # 获取移除前的数量
            count_before = self.vector_store.count()

            # 删除
            self.vector_store.delete(where={"source": source})

            # 获取移除后的数量
            count_after = self.vector_store.count()
            removed = count_before - count_after

            if removed == 0:
                return ToolResult(
                    content=f"未找到来源为 '{source}' 的文档。"
                )

            self.logger.info(f"Removed {removed} chunks from source: {source}")

            return ToolResult(
                content=f"成功移除来源 '{source}'：删除了 {removed} 个文本片段。"
            )

        except Exception as e:
            self.logger.error(f"Remove source failed: {e}")
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"移除失败: {str(e)}"
            )


__all__ = [
    "RAGSearchTool",
    "RAGAddDocumentTool",
    "RAGListSourcesTool",
    "RAGRemoveSourceTool",
]