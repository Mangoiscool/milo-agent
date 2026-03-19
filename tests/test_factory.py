"""
测试 LLM 工厂方法

测试内容：
- create_llm 函数
- 各种提供者的创建
"""

import pytest
from core.llm.factory import create_llm
from core.llm.providers.api import OpenAICompatibleLLM
from core.llm.providers.ollama import OllamaLLM


class TestCreateLLM:
    """测试 LLM 工厂函数"""

    def test_create_qwen(self):
        """测试创建 Qwen LLM"""
        llm = create_llm("qwen", api_key="test-key")
        assert isinstance(llm, OpenAICompatibleLLM)
        assert llm.model == "MiniMax-M2.1"
        assert llm.api_key == "test-key"

    def test_create_qwen_custom_model(self):
        """测试创建 Qwen LLM（自定义模型）"""
        llm = create_llm("qwen", api_key="test-key", model="qwen-max")
        assert llm.model == "qwen-max"

    def test_create_glm(self):
        """测试创建 GLM LLM"""
        llm = create_llm("glm", api_key="test-key")
        assert isinstance(llm, OpenAICompatibleLLM)
        assert llm.model == "glm-4"
        assert llm.api_key == "test-key"

    def test_create_glm_custom_model(self):
        """测试创建 GLM LLM（自定义模型）"""
        llm = create_llm("glm", api_key="test-key", model="glm-4-flash")
        assert llm.model == "glm-4-flash"

    def test_create_deepseek(self):
        """测试创建 DeepSeek LLM"""
        llm = create_llm("deepseek", api_key="test-key")
        assert isinstance(llm, OpenAICompatibleLLM)
        assert llm.model == "deepseek-chat"
        assert llm.api_key == "test-key"

    def test_create_ollama(self):
        """测试创建 Ollama LLM"""
        llm = create_llm("ollama")
        assert isinstance(llm, OllamaLLM)
        assert llm.model == "qwen3.5:4b"

    def test_create_ollama_custom_model(self):
        """测试创建 Ollama LLM（自定义模型）"""
        llm = create_llm("ollama", model="llama3:8b")
        assert llm.model == "llama3:8b"

    def test_create_ollama_custom_base_url(self):
        """测试创建 Ollama LLM（自定义 URL）"""
        llm = create_llm("ollama", base_url="http://custom:8080")
        assert llm.base_url == "http://custom:8080"

    def test_case_insensitive_provider(self):
        """测试提供者名称不区分大小写"""
        llm = create_llm("OLLAMA", model="test")
        assert isinstance(llm, OllamaLLM)

    def test_unknown_provider(self):
        """测试未知的提供者"""
        with pytest.raises(ValueError) as exc_info:
            create_llm("unknown", api_key="test")
        assert "Unknown provider" in str(exc_info.value)
        assert "unknown" in str(exc_info.value)

    def test_extra_kwargs_passed_through(self):
        """测试额外参数传递"""
        llm = create_llm("ollama", temperature=0.5, max_tokens=500)
        assert llm.temperature == 0.5
        assert llm.max_tokens == 500

    def test_think_parameter(self):
        """测试 think 参数"""
        llm_with_think = create_llm("ollama", think=True)
        assert llm_with_think.think is True

        llm_no_think = create_llm("ollama", think=False)
        assert llm_no_think.think is False
