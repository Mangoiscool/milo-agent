"""
Ollama 本地推理
支持本地运行的模型，如 Qwen、GLM、Llama 等

优势：
- 数据不出本地（隐私）
- 无 API 成本
- 可离线使用

前提：需要先安装 Ollama 并下载模型
- 安装：https://ollama.ai
- 下载模型：ollama pull qwen2:7b
"""

import json
from typing import AsyncIterator, List

from ..base import BaseLLM, Message, LLMResponse


class OllamaLLM(BaseLLM):
    """
    Ollama 本地推理
    
    Ollama API 文档：https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url.rstrip("/")
    
    def _build_request_body(self, messages: List[Message], stream: bool = False) -> dict:
        """
        Ollama 请求格式（与 OpenAI 略有不同）
        
        {
            "model": "qwen2:7b",
            "messages": [...],
            "stream": false,
            "options": {
                "temperature": 0.7,
                "num_predict": 2048
            }
        }
        """
        return {
            "model": self.model,
            "messages": [m.to_api_format() for m in messages],
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
    
    def chat(self, messages: List[Message]) -> LLMResponse:
        """同步对话"""
        import httpx
        
        url = f"{self.base_url}/api/chat"
        body = self._build_request_body(messages, stream=False)
        
        with httpx.Client() as client:
            response = client.post(url, json=body, timeout=120.0)
            response.raise_for_status()
            data = response.json()
        
        return self._parse_response(data)
    
    async def achat(self, messages: List[Message]) -> LLMResponse:
        """异步对话"""
        import httpx
        
        url = f"{self.base_url}/api/chat"
        body = self._build_request_body(messages, stream=False)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=body, timeout=120.0)
            response.raise_for_status()
            data = response.json()
        
        return self._parse_response(data)
    
    def _parse_response(self, data: dict) -> LLMResponse:
        """
        Ollama 响应格式：
        {
            "model": "qwen2:7b",
            "message": {"role": "assistant", "content": "..."},
            "done": true,
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        
        Qwen3 等思考模型格式：
        {
            "message": {"role": "assistant", "content": "", "thinking": "..."}
        }
        """
        message = data.get("message", {})
        content = message.get("content", "") if message else ""
        
        # 支持思考模型：如果 content 为空但有 thinking，使用 thinking
        if not content and message and "thinking" in message:
            content = message["thinking"]
        
        return LLMResponse(
            content=content,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            raw_response=data,
        )
    
    async def astream(self, messages: List[Message]) -> AsyncIterator[str]:
        """
        流式输出
        
        Ollama 流式格式（每行一个 JSON）：
        {"model":"qwen2","message":{"role":"assistant","content":"你"},"done":false}
        {"model":"qwen2","message":{"role":"assistant","content":"好"},"done":true}
        """
        import httpx
        
        url = f"{self.base_url}/api/chat"
        body = self._build_request_body(messages, stream=True)
        
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=body, timeout=120.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        continue


def create_ollama_llm(model: str = "qwen2:7b", base_url: str = None, **kwargs) -> OllamaLLM:
    """
    创建 Ollama LLM 实例
    
    常用模型：
    - qwen2:7b      - 通义千问 7B
    - qwen2:14b     - 通义千问 14B（需要更多显存）
    - llama3:8b     - Llama 3 8B
    - glm4:9b       - GLM-4 9B
    """
    return OllamaLLM(
        model=model,
        base_url=base_url or "http://localhost:11434",
        **kwargs
    )
