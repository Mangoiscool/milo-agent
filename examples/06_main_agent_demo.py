"""
MainAgent 演示

展示 MainAgent 如何统一使用内置工具、RAG 和 Browser 能力。

运行前请设置环境变量：
    export QWEN_API_KEY="your-api-key"
"""

import asyncio
import os
from pathlib import Path

from core.llm.factory import create_llm
from core.rag import create_embedding
from agents import MainAgent


def get_llm():
    """获取 LLM 实例"""
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("⚠️  请先设置环境变量: export QWEN_API_KEY='your-api-key'")
        print("   或者使用 Ollama 本地模型（确保 Ollama 服务正在运行）")
        return None

    return create_llm(provider="qwen", api_key=api_key)


def get_ollama_llm():
    """获取 Ollama LLM（本地）"""
    try:
        return create_llm(provider="ollama", model="qwen3.5:4b")
    except Exception as e:
        print(f"⚠️  无法连接 Ollama: {e}")
        return None


def demo_builtin_tools():
    """演示内置工具"""
    print("\n" + "=" * 60)
    print("演示 1: 内置工具（默认启用）")
    print("=" * 60)

    # 尝试获取 LLM
    llm = get_llm() or get_ollama_llm()
    if not llm:
        return

    # 创建 MainAgent（默认启用内置工具）
    agent = MainAgent(llm=llm)

    print(f"\n已注册工具: {agent.list_tools()}")
    print(f"\n工具分类: {agent.get_tool_info()}")

    # 使用内置工具
    print("\n" + "-" * 40)
    print("问题: 今天是什么日期？帮我算一下 123 * 456 等于多少")
    print("-" * 40)

    response = agent.chat_with_tools("今天是什么日期？帮我算一下 123 * 456 等于多少")
    print(f"\n回答: {response}")


def demo_rag_capability():
    """演示 RAG 能力"""
    print("\n" + "=" * 60)
    print("演示 2: RAG 能力")
    print("=" * 60)

    # 尝试获取 LLM
    llm = get_llm() or get_ollama_llm()
    if not llm:
        return

    # 尝试创建 Embedding（使用已安装的 qwen3-embedding:0.6b）
    try:
        embedding = create_embedding(provider="ollama", model="qwen3-embedding:0.6b")
    except Exception as e:
        print(f"⚠️  无法创建 Embedding: {e}")
        print("   请确保 Ollama 服务正在运行，并已下载 embedding 模型:")
        print("   ollama pull qwen3-embedding:0.6b")
        return

    # 创建 MainAgent（启用 RAG）
    agent = MainAgent(
        llm=llm,
        enable_rag=True,
        embedding_model=embedding,
        persist_directory="./knowledge_base"
    )

    # 添加知识
    print("\n添加知识到知识库...")

    agent.add_text(
        "公司请假制度：员工每年有 15 天年假。请假需要提前 3 天申请，经直属领导批准。",
        source="company_policy"
    )

    agent.add_text(
        "报销流程：1. 填写报销单 2. 附上发票 3. 提交给财务部 4. 等待审批 5. 收到报销款",
        source="finance_guide"
    )

    print(f"\n知识库统计: {agent.get_knowledge_base_stats()}")

    # 查询知识库
    print("\n" + "-" * 40)
    print("问题: 公司的请假制度是什么？")
    print("-" * 40)

    response = agent.chat_with_tools("公司的请假制度是什么？")
    print(f"\n回答: {response}")


async def demo_browser_capability():
    """演示 Browser 能力"""
    print("\n" + "=" * 60)
    print("演示 3: Browser 能力")
    print("=" * 60)

    # 尝试获取 LLM
    llm = get_llm() or get_ollama_llm()
    if not llm:
        return

    # 创建 MainAgent（启用 Browser）
    agent = MainAgent(llm=llm, enable_browser=True)

    print(f"\n已注册工具: {agent.list_tools()}")

    # 初始化浏览器
    print("\n正在初始化浏览器...")
    try:
        await agent.initialize()
    except Exception as e:
        print(f"⚠️  无法初始化浏览器: {e}")
        print("   请确保已安装 playwright: pip install playwright && playwright install chromium")
        return

    try:
        # 执行浏览器任务
        print("\n" + "-" * 40)
        print("任务: 打开百度搜索 Python")
        print("-" * 40)

        response = agent.chat_with_tools("打开百度搜索 Python")
        print(f"\n回答: {response}")

    finally:
        # 清理
        await agent.close()


async def demo_full_featured():
    """演示完整功能"""
    print("\n" + "=" * 60)
    print("演示 4: 完整功能（内置工具 + RAG + Browser）")
    print("=" * 60)

    # 尝试获取 LLM
    llm = get_llm() or get_ollama_llm()
    if not llm:
        return

    # 尝试创建 Embedding
    try:
        embedding = create_embedding(provider="ollama", model="nomic-embed-text")
    except Exception as e:
        print(f"⚠️  无法创建 Embedding: {e}")
        return

    # 创建 MainAgent（启用所有能力）
    agent = MainAgent(
        llm=llm,
        enable_builtin_tools=True,
        enable_rag=True,
        embedding_model=embedding,
        persist_directory="./knowledge_base",
        enable_browser=True
    )

    print(f"\n工具信息: {agent.get_tool_info()}")
    print(f"知识库统计: {agent.get_knowledge_base_stats()}")

    # 添加知识
    agent.add_text(
        "API 文档：我们的 API 基础地址是 https://api.example.com，"
        "认证方式使用 Bearer Token。",
        source="api_docs"
    )

    # 初始化浏览器
    print("\n正在初始化浏览器...")
    try:
        await agent.initialize()
    except Exception as e:
        print(f"⚠️  无法初始化浏览器: {e}")
        return

    try:
        # 综合任务
        print("\n" + "-" * 40)
        print("任务: 帮我查一下公司文档中关于 API 的说明")
        print("-" * 40)

        response = agent.chat_with_tools("帮我查一下公司文档中关于 API 的说明")
        print(f"\n回答: {response}")

    finally:
        await agent.close()


def demo_event_system():
    """演示事件系统"""
    print("\n" + "=" * 60)
    print("演示 5: 事件系统")
    print("=" * 60)

    from agents.base import AgentEvent

    # 尝试获取 LLM
    llm = get_llm() or get_ollama_llm()
    if not llm:
        return

    # 创建 MainAgent
    agent = MainAgent(llm=llm)

    # 注册事件处理器
    def on_tool_call(name, arguments):
        print(f"  🔧 [工具调用] {name}({arguments})")

    def on_tool_result(name, result, is_error):
        status = "❌" if is_error else "✅"
        preview = result[:50] + "..." if len(result) > 50 else result
        print(f"  {status} [工具结果] {name}: {preview}")

    agent.on(AgentEvent.TOOL_CALL, on_tool_call)
    agent.on(AgentEvent.TOOL_RESULT, on_tool_result)

    # 执行任务
    print("\n" + "-" * 40)
    print("问题: 今天是星期几？")
    print("-" * 40)

    response = agent.chat_with_tools("今天是星期几？")
    print(f"\n回答: {response}")


def main():
    """主函数"""
    print("=" * 60)
    print("MainAgent 演示")
    print("=" * 60)
    print("""
MainAgent 是统一的 Agent 实现，可以组合多种能力：

1. 内置工具（默认启用）: calculator, datetime, web_search 等
2. RAG 能力（可选）: knowledge_search, knowledge_add 等
3. Browser 能力（可选）: browser_navigate, browser_click 等

运行前请确保：
- 设置 QWEN_API_KEY 环境变量，或
- 启动 Ollama 本地服务
""")

    print("请选择演示：")
    print("1. 内置工具")
    print("2. RAG 能力")
    print("3. Browser 能力")
    print("4. 完整功能")
    print("5. 事件系统")
    print("0. 退出")

    choice = input("\n请输入选择 (0-5): ").strip()

    if choice == "1":
        demo_builtin_tools()
    elif choice == "2":
        demo_rag_capability()
    elif choice == "3":
        asyncio.run(demo_browser_capability())
    elif choice == "4":
        asyncio.run(demo_full_featured())
    elif choice == "5":
        demo_event_system()
    elif choice == "0":
        print("再见！")
    else:
        print("无效选择")


if __name__ == "__main__":
    main()