"""
LLM 工厂方法
统一创建不同类型的 LLM 实例
"""

from typing import Optional
from .base import BaseLLM
from .providers.api import OpenAICompatibleLLM, create_qwen_llm, create_glm_llm, MODEL_CONFIGS
from .providers.ollama import OllamaLLM, create_ollama_llm


def create_llm(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> BaseLLM:
    """
    创建 LLM 实例的工厂方法
    
    Args:
        provider: 提供者类型
            - "qwen": 通义千问（需要 api_key）
            - "glm": 智谱 GLM（需要 api_key）
            - "deepseek": DeepSeek（需要 api_key）
            - "ollama": 本地 Ollama（无需 api_key）
        api_key: API 密钥（API 提供者必需）
        model: 模型名称（可选，使用默认值）
        base_url: 自定义 endpoint（可选）
        **kwargs: 其他参数（temperature, max_tokens 等）
    
    Returns:
        LLM 实例
    
    Examples:
        # Qwen API
        llm = create_llm("qwen", api_key="sk-xxx")
        
        # GLM API
        llm = create_llm("glm", api_key="xxx.xxx", model="glm-4-flash")
        
        # Ollama 本地
        llm = create_llm("ollama", model="qwen2:7b")
    """
    provider = provider.lower()
    
    if provider == "qwen":
        return create_qwen_llm(api_key=api_key, model=model, **kwargs)
    
    elif provider == "glm":
        return create_glm_llm(api_key=api_key, model=model, **kwargs)
    
    elif provider == "deepseek":
        config = MODEL_CONFIGS["deepseek"]
        return OpenAICompatibleLLM(
            model=model or config["default_model"],
            base_url=base_url or config["base_url"],
            api_key=api_key,
            **kwargs
        )
    
    elif provider == "ollama":
        return create_ollama_llm(model=model, base_url=base_url, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: qwen, glm, deepseek, ollama"
        )
