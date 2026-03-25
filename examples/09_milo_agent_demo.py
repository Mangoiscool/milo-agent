#!/usr/bin/env python3
"""
MiloAgent 综合测试脚本

演示优化后的 MiloAgent 能力：
- ReAct 推理模式（Thought → Action → Observation）
- 内置工具组合使用
- RAG 知识库检索
- 浏览器自动化
- 记忆系统

运行方式：
    python examples/09_milo_agent_demo.py

环境要求：
    - 设置 QWEN_API_KEY 或启动 Ollama 服务
    - 可选：安装 playwright 以使用浏览器功能
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from agents.milo_agent import MiloAgent
from core.llm.factory import create_llm
from core.rag.embeddings import create_embedding


console = Console()


def get_llm():
    """获取 LLM 实例"""
    # 优先使用环境变量配置的 API Key
    api_key = os.getenv("QWEN_API_KEY")
    if api_key:
        try:
            llm = create_llm(provider="qwen", model="qwen3.5-flash-2026-02-23", api_key=api_key)
            console.print("[green]✓[/green] 使用通义千问 API")
            return llm
        except Exception as e:
            console.print(f"[yellow]⚠ 通义千问初始化失败: {e}[/yellow]")

    # 尝试 Ollama 本地模型
    try:
        llm = create_llm(provider="ollama", model="qwen2.5:7b")
        console.print("[green]✓[/green] 使用 Ollama 本地模型")
        return llm
    except Exception as e:
        console.print(f"[yellow]⚠ Ollama 连接失败: {e}[/yellow]")

    return None


def demo_basic_tools():
    """演示基础工具使用（计算器、日期时间）"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]演示 1: 基础工具使用[/bold cyan]\n"
        "测试 calculator 和 datetime 工具",
        border_style="cyan"
    ))

    llm = get_llm()
    if not llm:
        console.print("[red]✗ 无法创建 LLM，跳过演示[/red]")
        return

    # 创建基础 Agent（仅内置工具）
    agent = MiloAgent(llm=llm)

    # 显示工具信息
    console.print(f"\n[blue]已注册工具:[/blue] {agent.get_tool_info()}")

    # 测试计算
    question = "计算 12345 × 67890 等于多少？"
    console.print(f"\n[cyan]用户:[/cyan] {question}")

    response = agent.chat(question, show_reasoning=True)

    console.print(Panel(
        Markdown(response),
        title="[bold green]MiloAgent 回答[/bold green]",
        border_style="green"
    ))


def demo_web_search():
    """演示网络搜索工具"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]演示 2: 网络搜索[/bold cyan]\n"
        "测试 web_search 工具获取最新信息",
        border_style="cyan"
    ))

    llm = get_llm()
    if not llm:
        console.print("[red]✗ 无法创建 LLM，跳过演示[/red]")
        return

    agent = MiloAgent(llm=llm)

    question = "2025年 AI 领域有哪些重要进展？请搜索最新信息"
    console.print(f"\n[cyan]用户:[/cyan] {question}")

    response = agent.chat(question, show_reasoning=True)

    console.print(Panel(
        Markdown(response),
        title="[bold green]MiloAgent 回答[/bold green]",
        border_style="green"
    ))


def demo_rag_knowledge():
    """演示 RAG 知识库功能"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]演示 3: RAG 知识库[/bold cyan]\n"
        "测试知识库添加和检索",
        border_style="cyan"
    ))

    llm = get_llm()
    if not llm:
        console.print("[red]✗ 无法创建 LLM，跳过演示[/red]")
        return

    # 尝试创建 Embedding 模型
    try:
        embedding = create_embedding(provider="ollama", model="qwen3-embedding:0.6b")
        console.print("[green]✓[/green] Embedding 模型就绪 (qwen3-embedding:0.6b)")
    except Exception as e:
        console.print(f"[red]✗ 无法创建 Embedding: {e}[/red]")
        console.print("[yellow]提示: 请确保 Ollama 服务正在运行，并已下载 embedding 模型:[/yellow]")
        console.print("[yellow]       ollama pull qwen3-embedding:0.6b[/yellow]")
        return

    # 创建带 RAG 的 Agent
    agent = MiloAgent(
        llm=llm,
        embedding_model=embedding,
        persist_directory="./workspace/knowledge_base"
    )

    # 添加知识
    console.print("\n[blue]添加知识到知识库...[/blue]")

    agent.add_text(
        text="""Milo Agent 产品手册 v1.0

产品概述：
Milo Agent 是一个智能 AI 助手，支持多种工具和能力组合。

核心功能：
1. 工具调用：计算器、搜索、文件操作等
2. 知识库：支持文档检索和问答
3. 浏览器：网页自动化操作
4. 记忆：短期记忆和长期记忆

定价：
- 免费版：基础工具使用
- 专业版：¥99/月，包含所有功能
- 企业版：定制报价""",
        source="product_manual"
    )

    agent.add_text(
        text="""技术支持联系方式：
- 邮箱：support@milo-agent.com
- 电话：400-123-4567
- 在线客服：工作日 9:00-18:00
- 技术文档：https://docs.milo-agent.com""",
        source="support_info"
    )

    # 显示知识库统计
    stats = agent.get_knowledge_base_stats()
    console.print(f"[green]✓[/green] 知识库状态: {stats}")

    # 查询知识库
    questions = [
        "Milo Agent 有哪些核心功能？",
        "如何联系技术支持？",
        "专业版定价是多少？"
    ]

    for question in questions:
        console.print(f"\n[cyan]用户:[/cyan] {question}")
        response = agent.chat(question, show_reasoning=True)
        console.print(Panel(
            Markdown(response),
            title="[bold green]MiloAgent 回答[/bold green]",
            border_style="green"
        ))


async def demo_browser():
    """演示浏览器自动化"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]演示 4: 浏览器自动化[/bold cyan]\n"
        "测试 browser 工具链",
        border_style="cyan"
    ))

    llm = get_llm()
    if not llm:
        console.print("[red]✗ 无法创建 LLM，跳过演示[/red]")
        return

    # 创建带 Browser 的 Agent
    agent = MiloAgent(llm=llm, enable_browser=True)

    console.print("[blue]正在初始化浏览器...[/blue]")
    try:
        await agent.initialize()
        console.print("[green]✓[/green] 浏览器初始化完成")
    except Exception as e:
        console.print(f"[red]✗ 浏览器初始化失败: {e}[/red]")
        console.print("[yellow]提示: 请运行 'playwright install chromium'[/yellow]")
        return

    try:
        # 浏览器任务
        question = "打开 https://www.baidu.com，搜索 'Python 编程'，告诉我搜索结果页面的标题"
        console.print(f"\n[cyan]用户:[/cyan] {question}")

        response = agent.chat(question, show_reasoning=True)

        console.print(Panel(
            Markdown(response),
            title="[bold green]MiloAgent 回答[/bold green]",
            border_style="green"
        ))

    finally:
        await agent.close()
        console.print("[blue]浏览器已关闭[/blue]")


def demo_memory():
    """演示记忆系统"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]演示 5: 记忆系统[/bold cyan]\n"
        "测试多轮对话中的记忆保持（短期记忆 + 长期记忆）",
        border_style="cyan"
    ))

    llm = get_llm()
    if not llm:
        console.print("[red]✗ 无法创建 LLM，跳过演示[/red]")
        return

    # 尝试创建 Embedding 模型以启用长期记忆
    embedding = None
    try:
        embedding = create_embedding(provider="ollama", model="qwen3-embedding:0.6b")
        console.print("[green]✓[/green] 长期记忆已启用 (qwen3-embedding:0.6b)")
    except Exception as e:
        console.print(f"[yellow]⚠ 长期记忆未启用: {e}[/yellow]")
        console.print("[yellow]  将仅使用短期记忆[/yellow]")

    agent = MiloAgent(
        llm=llm,
        embedding_model=embedding,
        session_id="demo_session_001"
    )

    # 显示记忆系统状态
    capabilities = agent.get_capabilities()
    console.print(f"\n[blue]记忆系统状态:[/blue]")
    console.print(f"  - 长期记忆: {'启用' if capabilities.get('long_term_memory') else '禁用'}")

    # 多轮对话
    conversations = [
        "我叫张三，是一名软件工程师",
        "我喜欢用 Python 编程",
        "请推荐一些适合我的学习资源",
        "你记得我叫什么名字吗？",
        "根据我之前说的，给我一些建议"
    ]

    for i, user_input in enumerate(conversations, 1):
        console.print(f"\n[cyan]用户 ({i}/5):[/cyan] {user_input}")
        response = agent.chat(user_input, show_reasoning=False)
        console.print(Panel(
            Markdown(response),
            title="[bold green]MiloAgent[/bold green]",
            border_style="green"
        ))


def demo_complex_task():
    """演示复杂任务（工具组合）"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]演示 6: 复杂任务 - 工具组合[/bold cyan]\n"
        "测试多步骤推理和工具组合",
        border_style="cyan"
    ))

    llm = get_llm()
    if not llm:
        console.print("[red]✗ 无法创建 LLM，跳过演示[/red]")
        return

    agent = MiloAgent(llm=llm)

    # 复杂任务
    question = """帮我完成以下任务：
1. 计算 (123 + 456) × 78 等于多少
2. 告诉我现在的时间
3. 搜索 'Python 最新版本' 的最新信息
请分步骤完成并给出总结"""

    console.print(f"\n[cyan]用户:[/cyan] {question}")

    response = agent.chat(question, show_reasoning=True)

    console.print(Panel(
        Markdown(response),
        title="[bold green]MiloAgent 回答[/bold green]",
        border_style="green"
    ))


def show_system_prompt():
    """显示当前使用的 system prompt"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]当前 System Prompt[/bold cyan]\n"
        "MiloAgent 使用的默认系统提示词",
        border_style="cyan"
    ))

    # 直接访问类属性
    prompt = MiloAgent.DEFAULT_SYSTEM_PROMPT

    console.print(Panel(
        Markdown(prompt),
        title="[bold]DEFAULT_SYSTEM_PROMPT[/bold]",
        border_style="blue"
    ))


def interactive_mode():
    """交互式对话模式"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]交互式对话模式[/bold cyan]\n"
        "输入问题与 MiloAgent 对话\n"
        "命令: [yellow]/reasoning[/yellow] 切换推理显示 | [yellow]/quit[/yellow] 退出",
        border_style="cyan"
    ))

    llm = get_llm()
    if not llm:
        console.print("[red]✗ 无法创建 LLM[/red]")
        return

    agent = MiloAgent(llm=llm)
    show_reasoning = False

    console.print(f"\n[dim]工具数量: {agent.get_tool_info()}[/dim]")

    while True:
        console.print()
        user_input = console.input("[cyan]你:[/cyan] ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
            console.print("[yellow]再见！[/yellow]")
            break

        if user_input.lower() == '/reasoning':
            show_reasoning = not show_reasoning
            console.print(f"[dim]推理显示: {'开启' if show_reasoning else '关闭'}[/dim]")
            continue

        try:
            response = agent.chat(user_input, show_reasoning=show_reasoning)
            console.print(Panel(
                Markdown(response),
                title="[bold green]MiloAgent[/bold green]",
                border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold]MiloAgent 综合测试脚本[/bold]\n\n"
        "测试优化后的 System Prompt 和各种能力组合\n"
        "[dim]Version: 2.0 | Updated System Prompt[/dim]",
        border_style="blue"
    ))

    # 检查环境
    if not os.getenv("QWEN_API_KEY"):
        console.print("[yellow]⚠ 未设置 QWEN_API_KEY，将尝试使用 Ollama 本地模型[/yellow]")

    # 显示菜单
    console.print("\n[bold]选择测试项目:[/bold]")

    menu = Table(show_header=False, border_style="dim")
    menu.add_column("选项", style="cyan", justify="right")
    menu.add_column("描述")

    menu.add_row("1", "基础工具（计算器、日期时间）")
    menu.add_row("2", "网络搜索")
    menu.add_row("3", "RAG 知识库")
    menu.add_row("4", "浏览器自动化")
    menu.add_row("5", "记忆系统")
    menu.add_row("6", "复杂任务（工具组合）")
    menu.add_row("7", "查看当前 System Prompt")
    menu.add_row("8", "交互式对话模式")
    menu.add_row("0", "退出")

    console.print(menu)

    choice = console.input("\n[bold]请输入选择 (0-8):[/bold] ").strip()

    if choice == "1":
        demo_basic_tools()
    elif choice == "2":
        demo_web_search()
    elif choice == "3":
        demo_rag_knowledge()
    elif choice == "4":
        asyncio.run(demo_browser())
    elif choice == "5":
        demo_memory()
    elif choice == "6":
        demo_complex_task()
    elif choice == "7":
        show_system_prompt()
    elif choice == "8":
        interactive_mode()
    elif choice == "0":
        console.print("[yellow]退出测试[/yellow]")
    else:
        console.print("[red]无效选择，运行默认演示[/red]")
        demo_basic_tools()


if __name__ == "__main__":
    main()
