"""
测试 LLM 基础模块

测试内容：
- Message 类的功能
- Role 枚举
- LLMResponse 类
"""

import pytest
from core.llm.base import Message, Role, LLMResponse


class TestMessage:
    """测试 Message 类"""

    def test_message_creation(self):
        """测试创建消息"""
        msg = Message(role=Role.USER, content="你好")
        assert msg.role == Role.USER
        assert msg.content == "你好"
        assert msg.name is None

    def test_message_with_name(self):
        """测试带 name 的消息"""
        msg = Message(role=Role.USER, content="你好", name="Alice")
        assert msg.name == "Alice"

    def test_to_api_format(self):
        """测试转换为 API 格式"""
        msg = Message(role=Role.USER, content="你好")
        api_format = msg.to_api_format()
        assert api_format == {"role": "user", "content": "你好"}

    def test_to_api_format_with_name(self):
        """测试带 name 的消息转换为 API 格式"""
        msg = Message(role=Role.USER, content="你好", name="Alice")
        api_format = msg.to_api_format()
        assert api_format == {"role": "user", "content": "你好", "name": "Alice"}

    def test_message_pydantic_validation(self):
        """测试 Pydantic 验证"""
        with pytest.raises(Exception):  # Pydantic 会抛出 ValidationError
            Message(content="缺少 role")

    def test_message_invalid_role(self):
        """测试无效的角色值"""
        with pytest.raises(Exception):
            Message(role="invalid", content="你好")


class TestRole:
    """测试 Role 枚举"""

    def test_role_values(self):
        """测试角色枚举值"""
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.FUNCTION == "function"
        assert Role.TOOL == "tool"

    def test_role_str_comparison(self):
        """测试角色字符串比较"""
        msg = Message(role="user", content="test")
        assert msg.role == Role.USER


class TestLLMResponse:
    """测试 LLMResponse 类"""

    def test_response_creation(self):
        """测试创建响应"""
        response = LLMResponse(content="回答内容")
        assert response.content == "回答内容"
        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0
        assert response.total_tokens == 0

    def test_response_with_tokens(self):
        """测试带 token 统计的响应"""
        response = LLMResponse(
            content="回答",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20
        assert response.total_tokens == 30

    def test_response_with_raw(self):
        """测试带原始响应数据的响应"""
        raw_data = {"id": "test-123"}
        response = LLMResponse(content="回答", raw_response=raw_data)
        assert response.raw_response == raw_data
