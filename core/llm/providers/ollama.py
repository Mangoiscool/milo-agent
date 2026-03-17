"""
Ollama 本地推理
支持本地运行的模型，如 Qwen、GLM、Llama 等

优势：
- 数据不出本地（隐私）
- 无 API 成本
- 可离线使用

前提：需要先安装 Ollama 并下载模型
- 安装：https://ollama.ai
- 下载模型：ollama pull qwen3.5:4b
"""

import json
from typing import AsyncIterator, List, Optional
import httpx

from ..base import BaseLLM, Message, LLMResponse, ToolDefinition, ToolCall
from ...logger import get_logger


class OllamaLLM(BaseLLM):
    """
    Ollama 本地推理

    Ollama API 文档：https://github.com/ollama/ollama/blob/main/docs/api.md

    支持功能：
    - 对话（同步/异步/流式）
    - Function Calling（工具调用）- 需要 Ollama 0.3.0+
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        think: bool = True,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.think = think  # 控制思考模式（Qwen3 等模型）
        self.logger = get_logger(self.__class__.__name__)

    def _build_request_body(
        self,
        messages: List[Message],
        stream: bool = False,
        tools: Optional[List[ToolDefinition]] = None
    ) -> dict:
        """
        Ollama 请求格式（与 OpenAI 略有不同）

        {
            "model": "qwen3.5:4b",
            "messages": [...],
            "stream": false,
            "options": {
                "temperature": 0.7,
                "num_predict": 2048
            },
            "think": false,  # 关闭思考模式（Qwen3）
            "tools": [...]   # 可选：工具定义
        }
        """
        body = {
            "model": self.model,
            "messages": [m.to_api_format() for m in messages],
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        # 如果 think=False，添加到请求体（Ollama 会忽略不支持该参数的模型）
        if not self.think:
            body["think"] = False

        # 添加工具定义（Ollama 0.3.0+ 支持）
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                }
                for t in tools
            ]

        return body

    def chat(self, messages: List[Message]) -> LLMResponse:
        """同步对话"""
        self.logger.info(f"Ollama chat with {self.model}, messages: {len(messages)}")
        url = f"{self.base_url}/api/chat"
        body = self._build_request_body(messages, stream=False)

        self.logger.debug(f"Requesting Ollama at {url}")

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=body)
            response.raise_for_status()
            data = response.json()

        self.logger.debug(f"Ollama response: tokens={data.get('prompt_eval_count', 0)}+{data.get('eval_count', 0)}")
        return self._parse_response(data)

    async def achat(self, messages: List[Message]) -> LLMResponse:
        """异步对话"""
        self.logger.info(f"Ollama async chat with {self.model}, messages: {len(messages)}")
        url = f"{self.base_url}/api/chat"
        body = self._build_request_body(messages, stream=False)

        self.logger.debug(f"Async requesting Ollama at {url}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=body)
            response.raise_for_status()
            data = response.json()

        self.logger.debug(f"Ollama async response: tokens={data.get('prompt_eval_count', 0)}+{data.get('eval_count', 0)}")
        return self._parse_response(data)

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
        """
        if not tools:
            return self.chat(messages)

        self.logger.info(f"Ollama chat with tools, model={self.model}, messages={len(messages)}, tools={len(tools)}")
        url = f"{self.base_url}/api/chat"
        body = self._build_request_body(messages, stream=False, tools=tools)

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=body)
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data)

    async def achat_with_tools(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None
    ) -> LLMResponse:
        """
        异步对话（支持工具调用）
        """
        if not tools:
            return await self.achat(messages)

        self.logger.info(f"Ollama async chat with tools, model={self.model}, messages={len(messages)}, tools={len(tools)}")
        url = f"{self.base_url}/api/chat"
        body = self._build_request_body(messages, stream=False, tools=tools)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=body)
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data)
    
    def _parse_response(self, data: dict) -> LLMResponse:
        """
        Ollama 响应格式：
        {
            "model": "qwen3.5:4b",
            "message": {"role": "assistant", "content": "..."},
            "done": true,
            "prompt_eval_count": 10,
            "eval_count": 20
        }

        Qwen3 等思考模型格式：
        {
            "message": {"role": "assistant", "content": "", "thinking": "..."}
        }

        工具调用格式（Ollama 0.3.0+）：
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "tool_name",
                            "arguments": {...}
                        }
                    }
                ]
            }
        }
        """
        message = data.get("message", {})
        content = message.get("content", "") if message else ""

        # 支持思考模型：如果 content 为空但有 thinking，使用 thinking
        if not content and message and "thinking" in message:
            content = message["thinking"]

        # 解析 tool_calls
        tool_calls = []
        if message and "tool_calls" in message:
            for tc in message["tool_calls"]:
                try:
                    func = tc.get("function", {})
                    # arguments 可能是字符串或字典
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args) if args else {}

                    tool_calls.append(ToolCall(
                        id=tc.get("id", f"call_{len(tool_calls)}"),
                        name=func.get("name", ""),
                        arguments=args
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to parse tool_call: {e}")

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            raw_response=data,
        )
    
    async def astream(self, messages: List[Message]) -> AsyncIterator[str]:
        """
        流式输出

        Ollama 流式格式（每行一个 JSON）：
        {"model":"qwen3.5","message":{"role":"assistant","content":"你"},"done":false}
        {"model":"qwen3.5","message":{"role":"assistant","content":"好"},"done":true}
        """
        self.logger.info(f"Ollama stream chat with {self.model}, messages: {len(messages)}")
        url = f"{self.base_url}/api/chat"
        body = self._build_request_body(messages, stream=True)

        self.logger.debug(f"Starting Ollama stream at {url}")

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=body, timeout=120.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            if content:  # Only yield if content is not None/empty
                                yield content
                        if data.get("done"):
                            self.logger.debug("Ollama stream completed")
                            break
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"Ollama stream parse error: {e}")
                        continue


def create_ollama_llm(model: str = "qwen3.5:4b", base_url: str = None, think: bool = True, **kwargs) -> OllamaLLM:
    """
    创建 Ollama LLM 实例

    常用模型：
    - qwen3.5:4b    - 通义千问 3.5 4B（推荐，支持思考模式）
    - qwen2.5:7b    - 通义千问 2.5 7B
    - llama3:8b     - Llama 3 8B
    - glm4:9b       - GLM-4 9B

    参数：
    - think: 是否启用思考模式（仅 Qwen3 等支持），默认 True
    """
    return OllamaLLM(
        model=model,
        base_url=base_url or "http://localhost:11434",
        think=think,
        **kwargs
    )
