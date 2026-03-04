"""
LLM 模块测试脚本

用法：
    # 测试 GLM API（需要设置 GLM_API_KEY 环境变量）
    python tests/test_llm.py --provider glm
    
    # 测试 Qwen API
    python tests/test_llm.py --provider qwen
    
    # 测试 Ollama 本地
    python tests/test_llm.py --provider ollama --model qwen2:7b
    
    # 流式输出测试
    python tests/test_llm.py --provider glm --stream
"""

import sys
import os
import asyncio
import argparse

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import Message, Role
from core.llm.factory import create_llm


def test_sync_chat(llm):
    """测试同步对话"""
    print("\n" + "="*50)
    print("📌 测试同步对话")
    print("="*50)
    
    messages = [
        Message(role=Role.SYSTEM, content="你是一个有帮助的AI助手。"),
        Message(role=Role.USER, content="用一句话介绍你自己。"),
    ]
    
    print(f"📤 发送消息: {messages[-1].content}")
    
    response = llm.chat(messages)
    
    print(f"📥 收到回复: {response.content}")
    print(f"📊 Token 使用: {response.total_tokens} (prompt: {response.prompt_tokens}, completion: {response.completion_tokens})")


async def test_async_chat(llm):
    """测试异步对话"""
    print("\n" + "="*50)
    print("📌 测试异步对话")
    print("="*50)
    
    messages = [
        Message(role=Role.USER, content="1+1等于几？只回答数字。"),
    ]
    
    print(f"📤 发送消息: {messages[-1].content}")
    
    response = await llm.achat(messages)
    
    print(f"📥 收到回复: {response.content}")


async def test_stream_chat(llm):
    """测试流式输出"""
    print("\n" + "="*50)
    print("📌 测试流式输出")
    print("="*50)
    
    messages = [
        Message(role=Role.USER, content="用三句话介绍一下 Python 语言。"),
    ]
    
    print(f"📤 发送消息: {messages[-1].content}")
    print("📥 流式回复: ", end="", flush=True)
    
    async for chunk in llm.astream(messages):
        print(chunk, end="", flush=True)
    
    print()  # 换行


def main():
    parser = argparse.ArgumentParser(description="测试 LLM 模块")
    parser.add_argument(
        "--provider",
        choices=["qwen", "glm", "deepseek", "ollama"],
        required=True,
        help="LLM 提供者",
    )
    parser.add_argument(
        "--model",
        help="模型名称（可选）",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="测试流式输出",
    )
    parser.add_argument(
        "--api-key",
        help="API Key（也可通过环境变量设置）",
    )
    
    args = parser.parse_args()
    
    # 获取 API Key
    api_key = args.api_key
    if not api_key and args.provider != "ollama":
        env_key = f"{args.provider.upper()}_API_KEY"
        api_key = os.environ.get(env_key)
        if not api_key:
            print(f"❌ 错误: 请设置环境变量 {env_key} 或使用 --api-key 参数")
            sys.exit(1)
    
    # 创建 LLM 实例
    kwargs = {"temperature": 0.7, "max_tokens": 100}
    if args.model:
        kwargs["model"] = args.model
    
    print(f"🚀 创建 {args.provider} LLM 实例...")
    llm = create_llm(
        provider=args.provider,
        api_key=api_key,
        **kwargs
    )
    print(f"✅ LLM 实例: {llm}")
    
    # 运行测试
    if args.stream:
        asyncio.run(test_stream_chat(llm))
    else:
        test_sync_chat(llm)
        asyncio.run(test_async_chat(llm))
    
    print("\n✨ 测试完成！")


if __name__ == "__main__":
    main()
