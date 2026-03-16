"""Embedding 抽象层

支持多种 Embedding 提供者：
- 本地模型（Sentence Transformers）
- Ollama 本地服务
- API 提供者（阿里云百炼、OpenAI 等）
"""

from abc import ABC, abstractmethod
from typing import Any

import httpx


class BaseEmbedding(ABC):
    """Embedding 基类"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """将文本转换为向量"""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量转换文本为向量"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        pass


class SentenceTransformersEmbedding(BaseEmbedding):
    """
    本地 Sentence Transformers Embedding

    使用本地模型生成向量，无需 API 调用。
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None
    ):
        """
        Args:
            model_name: 模型名称或路径
            device: 设备 (cuda, cpu, mps)，None 自动检测
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dimension = None

    def _ensure_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name, device=self._device)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._ensure_model()
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        embeddings = model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


class OllamaEmbedding(BaseEmbedding):
    """
    Ollama Embedding

    通过 Ollama 服务生成向量，支持本地部署的模型。
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ):
        """
        Args:
            model: Ollama 模型名称
            base_url: Ollama 服务地址
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimension = None

    def embed(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        embedding = result["embedding"]

        # 缓存维度
        if self._dimension is None:
            self._dimension = len(embedding)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama 不支持批量，逐个处理
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # 用一个测试文本获取维度
            test = self.embed("test")
            self._dimension = len(test)
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI API Embedding

    使用 OpenAI 或兼容 API 生成向量。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int | None = None
    ):
        """
        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL（兼容其他服务）
            dimension: 向量维度（某些模型支持调整）
        """
        self._model = model
        self._api_key = api_key
        self._base_url = base_url or "https://api.openai.com/v1"
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }
        if self._dimension:
            payload["dimensions"] = self._dimension

        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()

        embeddings = [item["embedding"] for item in result["data"]]

        # 缓存维度
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        return embeddings

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # 根据模型推断维度
            dimension_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            self._dimension = dimension_map.get(self._model, 1536)
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model


class BailianEmbedding(BaseEmbedding):
    """
    阿里云百炼 Embedding

    使用阿里云百炼 API 生成向量。
    """

    def __init__(
        self,
        model: str = "text-embedding-v3",
        api_key: str | None = None,
        dimension: int = 1024
    ):
        """
        Args:
            model: 模型名称 (text-embedding-v3, text-embedding-v2 等)
            api_key: 百炼 API Key
            dimension: 向量维度 (1024, 768 等)
        """
        self._model = model
        self._api_key = api_key
        self._base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dimension,
            "encoding_format": "float"
        }

        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()

        embeddings = [item["embedding"] for item in result["data"]]
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model


def create_embedding(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs
) -> BaseEmbedding:
    """
    工厂方法：创建 Embedding 实例

    Args:
        provider: 提供者 (sentence-transformers, ollama, openai, bailian)
        model: 模型名称
        api_key: API 密钥
        base_url: 服务地址
        **kwargs: 其他参数

    Returns:
        Embedding 实例
    """
    provider = provider.lower()

    if provider in ("sentence-transformers", "local", "st"):
        model = model or "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformersEmbedding(model_name=model, **kwargs)

    elif provider == "ollama":
        model = model or "nomic-embed-text"
        return OllamaEmbedding(model=model, base_url=base_url, **kwargs)

    elif provider in ("openai", "openai-compatible"):
        model = model or "text-embedding-3-small"
        return OpenAIEmbedding(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )

    elif provider in ("bailian", "aliyun", "dashscope"):
        model = model or "text-embedding-v3"
        return BailianEmbedding(
            model=model,
            api_key=api_key,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")