"""RAG 模块

检索增强生成 (Retrieval-Augmented Generation) 模块，提供：
- 文档加载（PDF、Markdown、Word、Excel、图像）
- 文本切分（递归、Markdown、代码）
- 向量嵌入（本地模型、Ollama、API）
- 向量存储（ChromaDB）
- 检索器（相似度、MMR、混合检索）
"""

from .base import (
    Chunk,
    Document,
    DocumentType,
    SearchResult,
    detect_document_type,
)
from .document_loader import (
    DocumentLoaderRegistry,
    ExcelLoader,
    ImageLoader,
    MarkdownLoader,
    PDFLoader,
    PowerPointLoader,
    TextLoader,
    WordLoader,
    create_default_registry,
)
from .embeddings import (
    BailianEmbedding,
    BaseEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    SentenceTransformersEmbedding,
    create_embedding,
)
from .retriever import (
    BaseRetriever,
    HybridRetriever,
    MMRRetriever,
    SimilarityRetriever,
    create_retriever,
)
from .text_splitter import (
    BaseTextSplitter,
    CodeTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    SplitConfig,
    create_splitter,
)
from .vector_store import KnowledgeBase, VectorStore

__all__ = [
    # Base
    "Document",
    "DocumentType",
    "Chunk",
    "SearchResult",
    "detect_document_type",
    # Document Loaders
    "BaseDocumentLoader",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "WordLoader",
    "ExcelLoader",
    "PowerPointLoader",
    "ImageLoader",
    "DocumentLoaderRegistry",
    "create_default_registry",
    # Text Splitters
    "SplitConfig",
    "BaseTextSplitter",
    "RecursiveCharacterTextSplitter",
    "MarkdownTextSplitter",
    "CodeTextSplitter",
    "create_splitter",
    # Embeddings
    "BaseEmbedding",
    "SentenceTransformersEmbedding",
    "OllamaEmbedding",
    "OpenAIEmbedding",
    "BailianEmbedding",
    "create_embedding",
    # Vector Store
    "VectorStore",
    "KnowledgeBase",
    # Retriever
    "BaseRetriever",
    "SimilarityRetriever",
    "MMRRetriever",
    "HybridRetriever",
    "create_retriever",
]