"""
LLM 抽象基类
定义统一的模型调用接口

学习重点：
- 为什么需要抽象层？→ 解耦业务逻辑和具体模型实现
- Message 的设计：角色（system/user/assistant/tool）+ 内容
- Function Calling：ToolDefinition, ToolCall, ToolResult
"""

import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional
from pydantic import BaseModel

from ..logger import get_logger


# ═══════════════════════════════════════════════════════════════
# 消息角色
# ═══════════════════════════════════════════════════════════════

class Role(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"           # 工具执行结果
    FUNCTION = "function"   # 兼容旧版 OpenAI 格式


# ═══════════════════════════════════════════════════════════════
# 工具调用相关数据结构
# ═══════════════════════════════════════════════════════════════

class ToolDefinition(BaseModel):
    """
    工具定义 - 传递给 LLM 的工具描述
    
    对应 OpenAI API 的 tools[].function 格式
    """
    name: str                       # 工具名称（唯一标识）
    description: str                # 工具描述（LLM 根据这个决定是否调用）
    parameters: Dict[str, Any]      # JSON Schema 格式的参数定义


class ToolCall(BaseModel):
    """
    工具调用 - LLM 返回的调用请求
    
    当 LLM 决定调用工具时，会返回这个结构
    """
    id: str                         # 调用 ID（用于结果匹配）
    name: str                       # 工具名称
    arguments: Dict[str, Any]       # 解析后的参数（dict 格式）


class ToolResult(BaseModel):
    """
    工具执行结果
    
    执行工具后返回给 Agent 的结果
    """
    call_id: str                    # 对应的 ToolCall.id
    name: str                       # 工具名称
    content: str                    # 执行结果（字符串形式）
    is_error: bool = False          # 是否执行失败


# ═══════════════════════════════════════════════════════════════
# 消息结构
# ═══════════════════════════════════════════════════════════════

class Message(BaseModel):
    """
    消息结构 - 支持工具调用
    
    为什么用 Pydantic？
    - 数据验证：防止传入错误类型
    - 序列化：方便转为 dict/JSON 传给 API
    - 可读性：模型定义即文档
    """
    role: Role
    content: Optional[str] = None   # 可选（assistant + tool_calls 时可能为空）
    name: Optional[str] = None      # tool 消息需要这个字段
    tool_calls: Optional[List[ToolCall]] = None  # LLM 返回的工具调用
    tool_call_id: Optional[str] = None  # tool 消息的调用 ID
    
    def to_api_format(self) -> dict:
        """
        转换为 API 调用格式（OpenAI 兼容）
        
        不同角色的格式：
        - system/user: {role, content}
        - assistant (带工具调用): {role, content, tool_calls}
        - tool: {role: "tool", content, tool_call_id}
        """
        result = {"role": self.role.value}
        
        # content 可能为空（assistant 只返回 tool_calls 时）
        if self.content is not None:
            result["content"] = self.content
        
        # tool 消息需要 name 字段
        if self.name:
            result["name"] = self.name
        
        # assistant 消息可能包含 tool_calls
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                    }
                }
                for tc in self.tool_calls
            ]
        
        # tool 消息需要 tool_call_id
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        
        return result


# ═══════════════════════════════════════════════════════════════
# LLM 响应
# ═══════════════════════════════════════════════════════════════

class LLMResponse(BaseModel):
    """
    模型响应 - 支持工具调用
    
    finish_reason 取值：
    - "stop": 正常结束，content 有内容
    - "tool_calls": LLM 要调用工具，tool_calls 有内容
    """
    content: str
    tool_calls: List[ToolCall] = []         # 工具调用列表
    finish_reason: str = "stop"             # stop | tool_calls
    # Token 使用量（用于成本追踪）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # 原始响应（调试用）
    raw_response: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════
# LLM 抽象基类
# ═══════════════════════════════════════════════════════════════

class BaseLLM(ABC):
    """
    LLM 抽象基类
    
    设计原则：
    1. 统一接口 → 不同模型用相同方式调用
    2. 支持同步/异步 → 异步更适合 Agent 场景（并发调用工具）
    3. 支持流式 → 长回复时提升用户体验
    4. 支持工具调用 → Function Calling
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
    
    # ═══════════════════════════════════════════════════════════════
    # 基础对话接口
    # ═══════════════════════════════════════════════════════════════
    
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
    
    # ═══════════════════════════════════════════════════════════════
    # 工具调用接口
    # ═══════════════════════════════════════════════════════════════
    
    def chat_with_tools(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None
    ) -> LLMResponse:
        """
        同步对话（支持工具调用）
        
        Args:
            messages: 对话历史
            tools: 可用工具列表
        
        Returns:
            LLM 响应（可能包含 tool_calls）
        
        默认实现：如果没有 tools，直接调用 chat
        子类可以重写以支持工具调用
        """
        if not tools:
            return self.chat(messages)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tool calling. "
            "Please use a provider that supports Function Calling."
        )
    
    async def achat_with_tools(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None
    ) -> LLMResponse:
        """
        异步对话（支持工具调用）
        
        默认实现：如果没有 tools，直接调用 achat
        """
        if not tools:
            return await self.achat(messages)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tool calling. "
            "Please use a provider that supports Function Calling."
        )
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model}>"
