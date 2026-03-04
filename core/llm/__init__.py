"""
LLM 抽象层
统一的模型调用接口，支持 API 和本地推理
"""

from .base import BaseLLM, Message, Role, LLMResponse

__all__ = ["BaseLLM", "Message", "Role", "LLMResponse"]
