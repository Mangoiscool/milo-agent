"""
测试 API 提供者

测试内容：
- OpenAICompatibleLLM 的请求构建
- 响应解析
- 流式输出
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from core.llm.base import Message, Role
from core.llm.providers.api import OpenAICompatibleLLM, MODEL_CONFIGS


class TestOpenAICompatibleLLM:
    """测试 OpenAI 兼容 API 提供者"""

    @pytest.fixture
    def llm(self):
        """创建测试用的 LLM 实例"""
        return OpenAICompatibleLLM(
            model="test-model",
            base_url="https://api.test.com/v1",
            api_key="test-key"
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
        assert llm.model == "test-model"
        assert llm.base_url == "https://api.test.com/v1"
        assert llm.api_key == "test-key"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2048

    def test_build_request_body(self, llm, sample_messages):
        """测试请求体构建"""
        body = llm._build_request_body(sample_messages, stream=False)
        assert body["model"] == "test-model"
        assert len(body["messages"]) == 2
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 2048
        assert body["stream"] is False

    def test_build_request_body_stream(self, llm, sample_messages):
        """测试流式请求体构建"""
        body = llm._build_request_body(sample_messages, stream=True)
        assert body["stream"] is True

    def test_get_headers(self, llm):
        """测试请求头"""
        headers = llm._get_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_parse_response(self, llm):
        """测试响应解析"""
        api_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "你好！有什么我可以帮助你的？"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        response = llm._parse_response(api_response)
        assert response.content == "你好！有什么我可以帮助你的？"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20
        assert response.total_tokens == 30
        assert response.raw_response == api_response

    def test_parse_response_without_usage(self, llm):
        """测试没有 usage 字段的响应解析"""
        api_response = {
            "choices": [{
                "message": {"role": "assistant", "content": "回复"}
            }]
        }
        response = llm._parse_response(api_response)
        assert response.content == "回复"
        assert response.total_tokens == 0

    @patch('httpx.Client')
    def test_chat(self, mock_client_class, llm, sample_messages):
        """测试同步对话"""
        # Mock HTTP 响应
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "测试回复"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
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
        assert "/chat/completions" in call_args[0][0]

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_achat(self, mock_client_class, llm, sample_messages):
        """测试异步对话"""
        # Mock HTTP 响应
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "choices": [{"message": {"content": "异步回复"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
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
                'data: {"choices":[{"delta":{"content":"你"}}]}\n',
                'data: {"choices":[{"delta":{"content":"好"}}]}\n',
                'data: [DONE]\n',
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

        assert len(chunks) > 0


class TestModelConfigs:
    """测试模型配置"""

    def test_qwen_config(self):
        """测试 Qwen 配置"""
        config = MODEL_CONFIGS["qwen"]
        assert "dashscope.aliyuncs.com" in config["base_url"]
        assert config["default_model"] == "MiniMax-M2.1"

    def test_glm_config(self):
        """测试 GLM 配置"""
        config = MODEL_CONFIGS["glm"]
        assert "bigmodel.cn" in config["base_url"]
        assert config["default_model"] == "glm-4"

    def test_deepseek_config(self):
        """测试 DeepSeek 配置"""
        config = MODEL_CONFIGS["deepseek"]
        assert "deepseek.com" in config["base_url"]
        assert config["default_model"] == "deepseek-chat"
