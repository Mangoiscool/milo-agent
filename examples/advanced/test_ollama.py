#!/usr/bin/env python3
"""
测试 Ollama 相关功能
"""

from core.llm.factory import create_llm
from agents.simple import SimpleAgent, AgentConfig
from core.tools import ToolRegistry
from core.tools.builtin import CalculatorTool

def test_ollama():
    """测试 Ollama 连接"""
    try:
        print("测试连接 Ollama...")
        llm = create_llm("ollama", model="qwen3.5:4b")
        print("✓ Ollama 连接成功")
        print(f"模型：{llm.model}")
        return llm
    except Exception as e:
        print(f"✗ Ollama 连接失败：{e}")
        print("请确保：")
        print("1. Ollama 已安装：https://ollama.ai")
        print("2. 已下载模型：ollama pull qwen3.5:4b")
        print("3. Ollama 服务正在运行：ollama serve")
        return None

def test_ollama_with_tools():
    """测试 Ollama（普通对话模式）"""
    llm = test_ollama()
    if not llm:
        return

    print("\n测试普通对话模式...")
    config = AgentConfig(system_prompt="你是一个有用的助手")
    agent = SimpleAgent(llm, config=config)

    # 测试对话（不使用工具）
    response = agent.chat("你好，请介绍一下自己")
    print(f"✓ 对话成功：{response}")

def test_api_provider():
    """测试 API 提供者配置"""
    print("\n测试 API 提供者配置...")

    providers = ["qwen", "glm", "deepseek"]
    for provider in providers:
        api_key = os.environ.get(f"{provider.upper()}_API_KEY")
        if api_key:
            print(f"✓ {provider.upper()}_API_KEY 已设置")
        else:
            print(f"✗ {provider.upper()}_API_KEY 未设置")
            print(f"  请运行：export {provider.upper()}_API_KEY='your-api-key'")

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("Ollama 配置测试")
    print("=" * 60)

    # 测试 Ollama
    test_ollama()

    # 测试工具调用（不适用）
    print("\n" + "-" * 60)
    print("注意：Ollama 不支持工具调用，将使用普通对话模式")
    test_ollama_with_tools()

    # 测试 API 提供者
    print("\n" + "-" * 60)
    print("API 提供者配置检查")
    test_api_provider()

    print("\n" + "=" * 60)
    print("测试完成")