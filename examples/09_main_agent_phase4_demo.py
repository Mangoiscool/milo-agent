"""MainAgent Phase 4 演示

展示 ReAct 和长期记忆功能
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm.factory import create_llm
from core.rag.embeddings import create_embedding
from agents.main import MainAgent


def demo_react():
    """演示 ReAct 推理模式"""
    print("=" * 60)
    print("演示 1: ReAct 推理模式")
    print("=" * 60)

    # 创建 LLM
    llm = create_llm("ollama", model="qwen3.5:4b", think=False)

    # 创建启用 ReAct 的 Agent
    agent = MainAgent(
        llm=llm,
        enable_builtin_tools=True,
        enable_react=True,
        max_react_iterations=5
    )

    print("\n能力状态:", agent.get_capabilities())
    print("工具列表:", agent.list_tools())

    # 测试 ReAct（不显示思考过程）
    print("\n--- 测试 1: 简单计算 ---")
    response = agent.chat_with_tools("15 + 25 等于多少？")
    print(f"问题: 15 + 25 等于多少？")
    print(f"回答: {response}")

    # 测试 ReAct（显示思考过程）
    print("\n--- 测试 2: 天气 + 计算（显示思考过程） ---")
    response = agent.chat_with_tools(
        "北京今天气温多少？如果明天降温5度，温度是多少？",
        show_reasoning=True
    )
    print(f"问题: 北京今天气温多少？如果明天降温5度，温度是多少？")
    print(f"\n完整响应（包含思考过程）:\n{response}")

    print("\n" + "=" * 60)
    print("ReAct 演示完成!")
    print("=" * 60)


def demo_long_term_memory():
    """演示长期记忆功能"""
    print("\n" + "=" * 60)
    print("演示 2: 长期记忆")
    print("=" * 60)

    # 创建 Embedding
    embedding = create_embedding("ollama")

    # 创建 LLM
    llm = create_llm("ollama", model="qwen3.5:4b", think=False)

    # 创建启用长期记忆的 Agent
    print("\n--- 会话 1: 记住用户信息 ---")
    agent1 = MainAgent(
        llm=llm,
        embedding_model=embedding,
        enable_long_term_memory=True,
        memory_session_id="demo_session_001"
    )

    print("能力状态:", agent1.get_capabilities())

    # 记住一些信息
    agent1.chat_with_tools("我叫 Mango，是一名系统架构师")
    agent1.chat_with_tools("我喜欢 Python 和 AI 技术")
    agent1.chat_with_tools("【重要】明天下午 3 点有个关键项目评审会")

    print("\n已记录信息:")
    print("- 名字: Mango")
    print("- 职业: 系统架构师")
    print("- 爱好: Python、AI")
    print("- 重要事件: 明天下午 3 点项目评审会")

    # 创建新会话（使用相同的长期记忆存储）
    print("\n--- 会话 2: 跨会话检索（新会话） ---")
    agent2 = MainAgent(
        llm=llm,
        embedding_model=embedding,
        enable_long_term_memory=True,
        memory_session_id="demo_session_002"  # 新会话 ID
    )

    # 测试跨会话记忆
    response = agent2.chat_with_tools("还记得我是谁吗？")
    print(f"问题: 还记得我是谁吗？")
    print(f"回答: {response}")

    response = agent2.chat_with_tools("我明天有什么重要的事情？")
    print(f"\n问题: 我明天有什么重要的事情？")
    print(f"回答: {response}")

    print("\n" + "=" * 60)
    print("长期记忆演示完成!")
    print("=" * 60)


def demo_combined():
    """演示 ReAct + 长期记忆组合"""
    print("\n" + "=" * 60)
    print("演示 3: ReAct + 长期记忆组合")
    print("=" * 60)

    # 创建 Embedding 和 LLM
    embedding = create_embedding("ollama")
    llm = create_llm("ollama", model="qwen3.5:4b", think=False)

    # 创建同时启用 ReAct 和长期记忆的 Agent
    agent = MainAgent(
        llm=llm,
        embedding_model=embedding,
        enable_builtin_tools=True,
        enable_react=True,
        enable_long_term_memory=True,
        memory_session_id="combined_demo"
    )

    print("\n能力状态:", agent.get_capabilities())

    # 先记住一些信息
    print("\n--- 步骤 1: 记住用户偏好 ---")
    agent.chat_with_tools("我喜欢 JavaScript，对 Python 不太熟悉")

    # 然后使用 ReAct + 长期记忆进行复杂任务
    print("\n--- 步骤 2: 复杂任务（ReAct + 长期记忆） ---")
    response = agent.chat_with_tools(
        "根据我的技术背景，帮我计算 (15 + 25) * 2 并用我熟悉的编程语言风格解释结果",
        show_reasoning=True
    )
    print(f"问题: 根据我的技术背景，帮我计算 (15 + 25) * 2 并用我熟悉的编程语言风格解释结果")
    print(f"\n完整响应:\n{response}")

    print("\n" + "=" * 60)
    print("组合演示完成!")
    print("=" * 60)


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           MainAgent Phase 4 功能演示                         ║
║                                                              ║
║  1. ReAct 推理模式 - 显式思考过程                             ║
║  2. 长期记忆 - 跨会话语义检索                                 ║
║  3. 组合使用 - ReAct + 长期记忆                               ║
╚══════════════════════════════════════════════════════════════╝
    """)

    try:
        # 演示 1: ReAct
        demo_react()

        # 演示 2: 长期记忆
        demo_long_term_memory()

        # 演示 3: 组合
        demo_combined()

        print("\n" + "=" * 60)
        print("所有演示完成!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n用户中断演示")
    except Exception as e:
        print(f"\n演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
