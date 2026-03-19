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

from ..base import BaseLLM, Message, LLMResponse, ToolDefinition, ToolCall
from ...logger import get_logger


class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI 兼容的 API 调用
    
    支持的模型：
    - Qwen: https://dashscope.aliyuncs.com/compatible-mode/v1
    - GLM: https://open.bigmodel.cn/api/paas/v4
    - DeepSeek: https://api.deepseek.com/v1
    
    支持功能：
    - 对话（同步/异步/流式）
    - Function Calling（工具调用）
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
        self.logger = get_logger(self.__class__.__name__)

    def _make_request(
        self, 
        messages: List[Message], 
        stream: bool = False,
        tools: Optional[List[ToolDefinition]] = None
    ) -> dict:
        """
        执行 HTTP 请求（同步）

        由 chat() 方法调用，避免与 achat() 的代码重复
        """
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(messages, stream=stream, tools=tools)

        self.logger.debug(f"Requesting {url} with model {self.model}")

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                url,
                headers=self._get_headers(),
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        self.logger.debug(f"Response received: {data.get('usage', {})}")
        return data
    
    def _build_request_body(
        self, 
        messages: List[Message], 
        stream: bool = False,
        tools: Optional[List[ToolDefinition]] = None
    ) -> dict:
        """
        构建请求体
        
        学习点：
        - temperature: 控制随机性（0=确定性，1=最随机）
        - max_tokens: 限制回复长度（成本控制）
        - stream: 流式输出开关
        - tools: 工具定义列表（Function Calling）
        """
        body = {
            "model": self.model,
            "messages": [m.to_api_format() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        
        # 添加工具定义
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
    
    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    # ═══════════════════════════════════════════════════════════════
    # 基础对话接口
    # ═══════════════════════════════════════════════════════════════
    
    def chat(self, messages: List[Message]) -> LLMResponse:
        """同步对话"""
        self.logger.info(f"Chat with {self.model}, messages: {len(messages)}")
        data = self._make_request(messages, stream=False)
        return self._parse_response(data)
    
    async def achat(self, messages: List[Message]) -> LLMResponse:
        """异步对话"""
        self.logger.info(f"Async chat with {self.model}, messages: {len(messages)}")
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(messages, stream=False)

        self.logger.debug(f"Async requesting {url} with model {self.model}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers=self._get_headers(),
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        self.logger.debug(f"Async response received: {data.get('usage', {})}")
        return self._parse_response(data)
    
    def _parse_response(self, data: dict) -> LLMResponse:
        """
        解析 API 响应
        
        OpenAI 格式的响应结构：
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "...",  // 可能为空
                    "tool_calls": [    // 可选
                        {
                            "id": "call_xxx",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": "{\"expression\": \"2+3\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "stop" | "tool_calls"
            }],
            "usage": {...}
        }
        """
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})
        finish_reason = choice.get("finish_reason", "stop")
        
        # 解析 tool_calls
        tool_calls = []
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                try:
                    # 解析 arguments JSON
                    args_str = tc["function"]["arguments"]
                    args = json.loads(args_str) if args_str else {}
                    
                    tool_calls.append(ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to parse tool_call: {e}")
        
        # content 可能为空（当 finish_reason 是 tool_calls 时）
        content = message.get("content") or ""
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw_response=data,
        )
    
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
        
        self.logger.info(f"Chat with tools, model={self.model}, messages={len(messages)}, tools={len(tools)}")
        data = self._make_request(messages, stream=False, tools=tools)
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
        
        self.logger.info(f"Async chat with tools, model={self.model}, messages={len(messages)}, tools={len(tools)}")
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(messages, stream=False, tools=tools)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers=self._get_headers(),
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data)
    
    # ═══════════════════════════════════════════════════════════════
    # 流式输出
    # ═══════════════════════════════════════════════════════════════
    
    async def astream(self, messages: List[Message]) -> AsyncIterator[str]:
        """
        流式输出

        SSE (Server-Sent Events) 格式：
        data: {"choices":[{"delta":{"content":"你"}}]}
        data: {"choices":[{"delta":{"content":"好"}}]}
        data: [DONE]
        
        注意：流式模式不支持工具调用
        """
        self.logger.info(f"Stream chat with {self.model}, messages: {len(messages)}")
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(messages, stream=True)

        self.logger.debug(f"Starting stream to {url}")

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
                        self.logger.debug("Stream completed")
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            content = delta["content"]
                            if content:  # Only yield if content is not None/empty
                                yield content
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        self.logger.debug(f"Stream parse error: {e}")
                        continue


# ═══════════════════════════════════════════════════════════════
# 预设配置：常用模型的 endpoint
# ═══════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "MiniMax-M2.1",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4.5-air",
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
