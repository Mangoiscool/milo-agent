#!/usr/bin/env python3
"""
ReAct Agent 演示

演示 ReAct (Reasoning + Acting) Agent 的能力：
- 显式思考过程
- 工具调用
- 多步骤推理

运行方式：
    python examples/07_react_agent_demo.py
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

from agents import ReActAgent
from core.llm.factory import create_llm
from core.tools import (
    CalculatorTool,
    DateTimeTool,
    WeatherTool,
    WebSearchTool,
)


console = Console()


def create_react_agent():
    """创建 ReAct Agent"""
    # 创建 LLM
    # 支持多种 LLM 提供商，根据环境变量选择
    provider = os.getenv("LLM_PROVIDER", "qwen")
    model = os.getenv("LLM_MODEL", "qwen-plus")
    api_key = os.getenv("QWEN_API_KEY")

    try:
        if provider == "qwen":
            if not api_key:
                raise ValueError("请设置 QWEN_API_KEY 环境变量")
            llm = create_llm(provider=provider, model=model, api_key=api_key)
        else:
            llm = create_llm(provider=provider, model=model)
        console.print(f"[green]✓[/green] 使用 LLM: {provider}/{model}")
    except Exception as e:
        console.print(f"[red]✗[/red] 创建 LLM 失败: {e}")
        console.print("[yellow]提示: 请设置 QWEN_API_KEY 环境变量[/yellow]")
        sys.exit(1)
    
    # 创建工具
    tools = [
        CalculatorTool(),
        DateTimeTool(),
        WeatherTool(),
        WebSearchTool(engine="duckduckgo"),
    ]
    
    # 创建 ReAct Agent
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        max_react_iterations=10
    )
    
    console.print(f"[green]✓[/green] 已注册工具: {', '.join(agent.list_tools())}")
    
    return agent


def demo_basic_chat():
    """演示基本对话"""
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold blue]演示 1: 基本对话（不需要工具）[/bold blue]",
        border_style="blue"
    ))
    
    agent = create_react_agent()
    
    question = "你好，你是谁？"
    console.print(f"\n[cyan]用户:[/cyan] {question}")
    
    response = agent.chat(question)
    
    console.print(Panel(
        Markdown(response),
        title="[bold green]ReAct Agent[/bold green]",
        border_style="green"
    ))


def demo_single_tool():
    """演示单工具调用"""
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold blue]演示 2: 单工具调用 - 天气查询[/bold blue]",
        border_style="blue"
    ))
    
    agent = create_react_agent()
    
    question = "北京今天天气怎么样？"
    console.print(f"\n[cyan]用户:[/cyan] {question}")
    
    # 显示思考过程
    response = agent.chat(question, show_reasoning=True)
    
    console.print(Panel(
        Markdown(response),
        title="[bold green]ReAct Agent (含思考过程)[/bold green]",
        border_style="green"
    ))


def demo_multi_step():
    """演示多步骤推理"""
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold blue]演示 3: 多步骤推理 - 天气 + 计算[/bold blue]",
        border_style="blue"
    ))
    
    agent = create_react_agent()
    
    question = "北京今天气温是多少？如果明天降温5度，温度会是多少？"
    console.print(f"\n[cyan]用户:[/cyan] {question}")
    
    # 显示思考过程
    response = agent.chat(question, show_reasoning=True)
    
    console.print(Panel(
        Markdown(response),
        title="[bold green]ReAct Agent (多步骤推理)[/bold green]",
        border_style="green"
    ))


def demo_complex_reasoning():
    """演示复杂推理"""
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold blue]演示 4: 复杂推理 - 多工具协作[/bold blue]",
        border_style="blue"
    ))
    
    agent = create_react_agent()
    
    question = "现在几点了？帮我算一下 123 * 456 等于多少"
    console.print(f"\n[cyan]用户:[/cyan] {question}")
    
    # 显示思考过程
    response = agent.chat(question, show_reasoning=True)
    
    console.print(Panel(
        Markdown(response),
        title="[bold green]ReAct Agent (复杂推理)[/bold green]",
        border_style="green"
    ))


def demo_without_reasoning():
    """演示不显示思考过程"""
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold blue]演示 5: 不显示思考过程[/bold blue]",
        border_style="blue"
    ))
    
    agent = create_react_agent()
    
    question = "上海今天天气如何？"
    console.print(f"\n[cyan]用户:[/cyan] {question}")
    
    # 不显示思考过程
    response = agent.chat(question, show_reasoning=False)
    
    console.print(Panel(
        Markdown(response),
        title="[bold green]ReAct Agent (最终答案)[/bold green]",
        border_style="green"
    ))


def demo_interactive():
    """交互式对话"""
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold blue]交互式对话模式[/bold blue]\n"
        "输入问题与 ReAct Agent 对话\n"
        "输入 'quit' 退出\n"
        "输入 'trace' 查看最近的思考过程",
        border_style="blue"
    ))
    
    agent = create_react_agent()
    last_trace = None
    
    while True:
        console.print()
        user_input = console.input("[cyan]你:[/cyan] ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            console.print("[yellow]再见！[/yellow]")
            break
        
        if user_input.lower() == 'trace':
            if last_trace:
                console.print(Panel(
                    last_trace,
                    title="[bold yellow]最近的思考过程[/bold yellow]",
                    border_style="yellow"
                ))
            else:
                console.print("[yellow]还没有执行过任何推理[/yellow]")
            continue
        
        # 对话
        try:
            response = agent.chat(user_input, show_reasoning=False)
            last_trace = agent.get_reasoning_summary()
            
            console.print(Panel(
                Markdown(response),
                title="[bold green]Agent[/bold green]",
                border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold]ReAct Agent 演示[/bold]\n\n"
        "ReAct = Reasoning + Acting\n\n"
        "特点:\n"
        "• 显式思考过程 (Thought)\n"
        "• 工具调用 (Action)\n"
        "• 结果观察 (Observation)\n"
        "• 执行轨迹追踪",
        border_style="blue"
    ))
    
    # 检查 API Key
    if not os.getenv("QWEN_API_KEY"):
        console.print("[yellow]警告: 未设置 QWEN_API_KEY 环境变量[/yellow]")
        console.print("[yellow]请设置: export QWEN_API_KEY=your-key-here[/yellow]")
    
    # 选择演示模式
    console.print("\n选择演示模式:")
    console.print("1. 基本对话（不需要工具）")
    console.print("2. 单工具调用 - 天气查询")
    console.print("3. 多步骤推理 - 天气 + 计算")
    console.print("4. 复杂推理 - 多工具协作")
    console.print("5. 不显示思考过程")
    console.print("6. 交互式对话")
    console.print("0. 退出")
    
    choice = console.input("\n[bold]请选择 (1-6, 0退出):[/bold] ").strip()
    
    if choice == "1":
        demo_basic_chat()
    elif choice == "2":
        demo_single_tool()
    elif choice == "3":
        demo_multi_step()
    elif choice == "4":
        demo_complex_reasoning()
    elif choice == "5":
        demo_without_reasoning()
    elif choice == "6":
        demo_interactive()
    elif choice == "0":
        console.print("[yellow]退出演示[/yellow]")
    else:
        console.print("[red]无效选择[/red]")
        # 默认运行演示 3
        console.print("\n运行默认演示: 多步骤推理...")
        demo_multi_step()


if __name__ == "__main__":
    main()