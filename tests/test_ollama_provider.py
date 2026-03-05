"""
测试 Ollama 提供者

测试内容：
- OllamaLLM 的请求构建
- 响应解析（包括思考模式）
- 流式输出
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from core.llm.base import Message, Role
from core.llm.providers.ollama import OllamaLLM, create_ollama_llm


class TestOllamaLLM:
    """测试 Ollama 提供者"""

    @pytest.fixture
    def llm(self):
        """创建测试用的 LLM 实例"""
        return OllamaLLM(
            model="qwen3.5:4b",
            base_url="http://localhost:11434"
        )

    @pytest.fixture
    def sample_messages(self):
        """示例消息列表"""
        return [
            Message(role=Role.SYSTEM, content="你是一个助手"),
            Message(role=Role.USER, content="你好"),
        ]

    def test_init(self, llm):
        """测试初始化"""
        assert llm.model == "qwen3.5:4b"
        assert llm.base_url == "http://localhost:11434"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2048
        assert llm.think is True  # 默认启用思考模式

    def test_init_with_think_disabled(self):
        """测试禁用思考模式"""
        llm = OllamaLLM(
            model="qwen3.5:4b",
            think=False
        )
        assert llm.think is False

    def test_base_url_trailing_slash(self):
        """测试 base_url 去除尾部斜杠"""
        llm = OllamaLLM(
            model="test",
            base_url="http://localhost:11434/"
        )
        assert llm.base_url == "http://localhost:11434"

    def test_build_request_body(self, llm, sample_messages):
        """测试请求体构建"""
        body = llm._build_request_body(sample_messages, stream=False)
        assert body["model"] == "qwen3.5:4b"
        assert len(body["messages"]) == 2
        assert body["stream"] is False
        assert "options" in body
        assert body["options"]["temperature"] == 0.7
        assert body["options"]["num_predict"] == 2048

    def test_build_request_body_without_think(self, sample_messages):
        """测试禁用思考模式的请求体"""
        llm = OllamaLLM(model="test", think=False)
        body = llm._build_request_body(sample_messages, stream=False)
        assert body["think"] is False

    def test_parse_response_normal(self, llm):
        """测试解析正常响应"""
        data = {
            "model": "qwen3.5:4b",
            "message": {"role": "assistant", "content": "你好！"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        response = llm._parse_response(data)
        assert response.content == "你好！"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20
        assert response.total_tokens == 30

    def test_parse_response_thinking_mode(self, llm):
        """测试解析思考模式响应"""
        data = {
            "model": "qwen3.5:4b",
            "message": {"role": "assistant", "content": "", "thinking": "正在思考..."},
            "done": True
        }
        response = llm._parse_response(data)
        # 当 content 为空但有 thinking 时，使用 thinking
        assert response.content == "正在思考..."

    def test_parse_response_content_takes_precedence(self, llm):
        """测试 content 优先于 thinking"""
        data = {
            "model": "qwen3.5:4b",
            "message": {"role": "assistant", "content": "实际回复", "thinking": "思考内容"},
            "done": True
        }
        response = llm._parse_response(data)
        # 当 content 不为空时，使用 content
        assert response.content == "实际回复"

    def test_parse_response_empty_message(self, llm):
        """测试空消息响应"""
        data = {
            "model": "qwen3.5:4b",
            "done": True
        }
        response = llm._parse_response(data)
        assert response.content == ""

    @patch('httpx.Client')
    def test_chat(self, mock_client_class, llm, sample_messages):
        """测试同步对话"""
        # Mock HTTP 响应
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "model": "qwen3.5:4b",
            "message": {"role": "assistant", "content": "测试回复"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 10
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        response = llm.chat(sample_messages)

        assert response.content == "测试回复"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/api/chat" in call_args[0][0]

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_achat(self, mock_client_class, llm, sample_messages):
        """测试异步对话"""
        # Mock HTTP 响应
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "model": "qwen3.5:4b",
            "message": {"role": "assistant", "content": "异步回复"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 10
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        response = await llm.achat(sample_messages)

        assert response.content == "异步回复"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_astream(self, mock_client_class, llm, sample_messages):
        """测试流式输出"""
        # Mock 流式响应
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        # 创建异步生成器函数
        async def async_line_iterator():
            lines = [
                '{"model":"qwen3.5","message":{"role":"assistant","content":"你"}}\n',
                '{"model":"qwen3.5","message":{"role":"assistant","content":"好"}}\n',
                '{"model":"qwen3.5","done":true}\n',
            ]
            for line in lines:
                yield line

        mock_response.aiter_lines = Mock(return_value=async_line_iterator())

        # Mock stream context
        mock_stream_context = Mock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_stream_context)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        chunks = []
        async for chunk in llm.astream(sample_messages):
            chunks.append(chunk)

        # 应该至少收到一些内容
        assert len(chunks) > 0


class TestCreateOllamaLLM:
    """测试 Ollama LLM 工厂函数"""

    def test_default_params(self):
        """测试默认参数"""
        llm = create_ollama_llm()
        assert llm.model == "qwen3.5:4b"
        assert llm.base_url == "http://localhost:11434"
        assert llm.think is True

    def test_custom_model(self):
        """测试自定义模型"""
        llm = create_ollama_llm(model="llama3:8b")
        assert llm.model == "llama3:8b"

    def test_custom_base_url(self):
        """测试自定义 base_url"""
        llm = create_ollama_llm(base_url="http://custom:8080")
        assert llm.base_url == "http://custom:8080"

    def test_think_disabled(self):
        """测试禁用思考模式"""
        llm = create_ollama_llm(think=False)
        assert llm.think is False

    def test_extra_kwargs(self):
        """测试额外参数"""
        llm = create_ollama_llm(temperature=0.5, max_tokens=1000)
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1000
