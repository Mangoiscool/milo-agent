"""
OpenAI 兼容的 API 提供者
支持：Qwen（通义千问）、GLM（智谱）、DeepSeek 等

为什么大多数国产模型都用 OpenAI 兼容接口？
- 降低迁移成本：开发者只需改 base_url 和 api_key
- 标准化：Function Calling、流式输出等格式统一
"""

import json
from typing import AsyncIterator, List, Optional
import httpx

from ..base import BaseLLM, Message, LLMResponse


class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI 兼容的 API 调用
    
    支持的模型：
    - Qwen: https://dashscope.aliyuncs.com/compatible-mode/v1
    - GLM: https://open.bigmodel.cn/api/paas/v4
    - DeepSeek: https://api.deepseek.com/v1
    """
    
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
    
    def _build_request_body(self, messages: List[Message], stream: bool = False) -> dict:
        """
        构建请求体
        
        学习点：
        - temperature: 控制随机性（0=确定性，1=最随机）
        - max_tokens: 限制回复长度（成本控制）
        - stream: 流式输出开关
        """
        return {
            "model": self.model,
            "messages": [m.to_api_format() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
    
    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def chat(self, messages: List[Message]) -> LLMResponse:
        """同步对话"""
        import httpx
        
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(messages, stream=False)
        
        with httpx.Client() as client:
            response = client.post(
                url,
                headers=self._get_headers(),
                json=body,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
        
        return self._parse_response(data)
    
    async def achat(self, messages: List[Message]) -> LLMResponse:
        """异步对话"""
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(messages, stream=False)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=self._get_headers(),
                json=body,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
        
        return self._parse_response(data)
    
    def _parse_response(self, data: dict) -> LLMResponse:
        """
        解析 API 响应
        
        OpenAI 格式的响应结构：
        {
            "choices": [{
                "message": {"role": "assistant", "content": "..."},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        """
        choice = data["choices"][0]
        usage = data.get("usage", {})
        
        return LLMResponse(
            content=choice["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw_response=data,
        )
    
    async def astream(self, messages: List[Message]) -> AsyncIterator[str]:
        """
        流式输出
        
        SSE (Server-Sent Events) 格式：
        data: {"choices":[{"delta":{"content":"你"}}]}
        data: {"choices":[{"delta":{"content":"好"}}]}
        data: [DONE]
        """
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(messages, stream=True)
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                headers=self._get_headers(),
                json=body,
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]  # 去掉 "data: "
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


# 预设配置：常用模型的 endpoint
MODEL_CONFIGS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
}


def create_qwen_llm(api_key: str, model: str = None, **kwargs) -> OpenAICompatibleLLM:
    """创建 Qwen LLM 实例"""
    config = MODEL_CONFIGS["qwen"]
    return OpenAICompatibleLLM(
        model=model or config["default_model"],
        base_url=config["base_url"],
        api_key=api_key,
        **kwargs
    )


def create_glm_llm(api_key: str, model: str = None, **kwargs) -> OpenAICompatibleLLM:
    """创建 GLM LLM 实例"""
    config = MODEL_CONFIGS["glm"]
    return OpenAICompatibleLLM(
        model=model or config["default_model"],
        base_url=config["base_url"],
        api_key=api_key,
        **kwargs
    )
