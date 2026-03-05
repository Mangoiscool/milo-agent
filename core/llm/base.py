"""
LLM 抽象基类
定义统一的模型调用接口

学习重点：
- 为什么需要抽象层？→ 解耦业务逻辑和具体模型实现
- Message 的设计：角色（system/user/assistant）+ 内容
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncIterator, List, Optional
from pydantic import BaseModel

from ..logger import get_logger


class Role(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    # Function Calling 时会用到
    FUNCTION = "function"
    TOOL = "tool"


class Message(BaseModel):
    """
    消息结构
    
    为什么用 Pydantic？
    - 数据验证：防止传入错误类型
    - 序列化：方便转为 dict/JSON 传给 API
    - 可读性：模型定义即文档
    """
    role: Role
    content: str
    # 某些模型支持 name 字段（多角色对话）
    name: Optional[str] = None
    
    def to_api_format(self) -> dict:
        """转换为 API 调用格式"""
        result = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


class LLMResponse(BaseModel):
    """模型响应"""
    content: str
    # Token 使用量（用于成本追踪）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # 原始响应（调试用）
    raw_response: Optional[dict] = None


class BaseLLM(ABC):
    """
    LLM 抽象基类
    
    设计原则：
    1. 统一接口 → 不同模型用相同方式调用
    2. 支持同步/异步 → 异步更适合 Agent 场景（并发调用工具）
    3. 支持流式 → 长回复时提升用户体验
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def chat(self, messages: List[Message]) -> LLMResponse:
        """
        同步对话
        
        Args:
            messages: 对话历史（包含 system prompt）
        
        Returns:
            模型响应
        """
        pass
    
    @abstractmethod
    async def achat(self, messages: List[Message]) -> LLMResponse:
        """异步对话"""
        pass
    
    @abstractmethod
    async def astream(self, messages: List[Message]) -> AsyncIterator[str]:
        """
        异步流式输出
        
        用法：
            async for chunk in llm.astream(messages):
                print(chunk, end="", flush=True)
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model}>"
