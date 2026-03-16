"""RAG 基础类型定义"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DocumentType(Enum):
    """支持的文档类型"""
    PDF = "pdf"
    MARKDOWN = "markdown"
    WORD = "word"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    TEXT = "text"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """文档对象"""
    content: str  # 文档内容
    metadata: dict[str, Any] = field(default_factory=dict)  # 元数据

    # 元数据字段（可选）
    source: str = ""  # 来源文件路径
    doc_type: DocumentType = DocumentType.UNKNOWN
    page: int | None = None  # 页码（PDF）
    chunk_index: int | None = None  # 切片索引

    def __post_init__(self):
        """自动填充元数据"""
        if self.source and "source" not in self.metadata:
            self.metadata["source"] = self.source
        if self.doc_type != DocumentType.UNKNOWN and "doc_type" not in self.metadata:
            self.metadata["doc_type"] = self.doc_type.value
        if self.page is not None and "page" not in self.metadata:
            self.metadata["page"] = self.page
        if self.chunk_index is not None and "chunk_index" not in self.metadata:
            self.metadata["chunk_index"] = self.chunk_index

    @classmethod
    def from_text(
        cls,
        text: str,
        source: str = "",
        doc_type: DocumentType = DocumentType.TEXT,
        **metadata
    ) -> "Document":
        """从文本创建文档"""
        return cls(
            content=text,
            source=source,
            doc_type=doc_type,
            metadata=metadata
        )


@dataclass
class Chunk:
    """文本切片"""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    id: str | None = None  # 向量存储中的 ID

    @classmethod
    def from_document(
        cls,
        doc: Document,
        text: str,
        chunk_index: int
    ) -> "Chunk":
        """从文档创建切片"""
        metadata = {**doc.metadata, "chunk_index": chunk_index}
        return cls(text=text, metadata=metadata)


@dataclass
class SearchResult:
    """检索结果"""
    chunk: Chunk
    score: float  # 相似度分数

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata


def detect_document_type(file_path: Path | str) -> DocumentType:
    """根据文件扩展名检测文档类型"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    type_map = {
        ".pdf": DocumentType.PDF,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
        ".doc": DocumentType.WORD,
        ".docx": DocumentType.WORD,
        ".xls": DocumentType.EXCEL,
        ".xlsx": DocumentType.EXCEL,
        ".ppt": DocumentType.POWERPOINT,
        ".pptx": DocumentType.POWERPOINT,
        ".txt": DocumentType.TEXT,
        ".png": DocumentType.IMAGE,
        ".jpg": DocumentType.IMAGE,
        ".jpeg": DocumentType.IMAGE,
        ".gif": DocumentType.IMAGE,
        ".webp": DocumentType.IMAGE,
    }

    return type_map.get(suffix, DocumentType.UNKNOWN)