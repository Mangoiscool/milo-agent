#!/usr/bin/env python3
"""
完整 Agent 对话循环演示

展示 Agent 如何使用工具完成任务
"""

import os
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from core.llm.factory import create_llm
from core.tools import ToolRegistry
from core.tools.builtin import (
    CalculatorTool,
    WeatherTool,
    WebSearchTool,
    FileReadTool,
    CodeExecutionTool
)
from agents.simple import SimpleAgent, AgentConfig


console = Console()


def create_demo_agent(llm_provider="qwen"):
    """创建演示 Agent

    Args:
        llm_provider: LLM 提供者，可选 "qwen", "glm", "deepseek", "ollama"
    """
    registry = ToolRegistry()

    # 注册工具
    console.rule("[bold cyan]注册工具...[/bold cyan]")
    registry.register(CalculatorTool())
    console.rule("[dim green]✓ Calculator[/dim green]")
    registry.register(WeatherTool())
    console.rule("[dim green]✓ 天气查询[/dim green]")
    registry.register(WebSearchTool(engine="duckduckgo"))
    console.rule("[dim green]✓ 网络搜索[/dim green]")
    registry.register(FileReadTool())
    console.rule("[dim green]✓ 文件读取[/dim green]")
    registry.register(CodeExecutionTool())
    console.rule("[dim green]✓ 代码执行[/dim green]")

    # 创建 Agent
    config = AgentConfig(
        system_prompt="你是一个有用的助手，可以使用各种工具来帮助用户完成任务。当需要计算时使用计算器，需要查询天气时使用天气工具，需要搜索信息时使用网络搜索工具。",
        enable_stream_fallback=True
    )

    # 根据 provider 创建 LLM
    if llm_provider == "ollama":
        try:
            llm = create_llm("ollama", model="qwen3.5:4b")
            # 不启用工具调用
            agent = SimpleAgent(llm, config=config)
        except Exception as e:
            console.print(f"[bold red]错误：无法连接到 Ollama[/bold red]")
            console.print("请确保：")
            console.print("1. 已安装 Ollama：https://ollama.ai")
            console.print("2. 已下载模型：ollama pull qwen3.5:4b")
            console.print("3. Ollama 服务正在运行：ollama serve")
            return None
    else:
        # API 提供者（支持工具调用）
        api_key = os.environ.get(f"{llm_provider.upper()}_API_KEY")
        if not api_key:
            console.print(f"[bold red]错误：[/bold red]")
            console.print(f"需要设置 {llm_provider.upper()}_API_KEY 环境变量")
            console.print(f"请运行：export {llm_provider.upper()}_API_KEY='your-api-key'")
            return None

        try:
            llm = create_llm(llm_provider, api_key=api_key)
            agent = SimpleAgent(llm, tools=list(registry._tools.values()), config=config)
        except Exception as e:
            console.print(f"[bold red]错误：创建 {llm_provider} LLM 失败[/bold red]")
            console.print(f"请检查 API key 是否正确：{str(e)}")
            return None

    return agent


def demo(llm_provider="qwen", test_inputs=None):
    """演示对话循环

    Args:
        llm_provider: LLM 提供者，可选 "qwen", "glm", "deepseek", "ollama"
        test_inputs: 测试输入列表
    """
    agent = create_demo_agent(llm_provider)
    if agent is None:
        return


    if not test_inputs:
        # 交互模式
        console.print("\n" + "="*60)
        console.print(Panel(
            "[bold cyan]欢迎使用 milo-agent 工具系统演示[/bold cyan]",
            border_style="cyan"
        ))
        console.print("\n可用工具:")

        tools = agent.list_tools() if hasattr(agent, 'list_tools') else []
        for tool_name in tools:
            console.print(f"  - {tool_name}")

        console.print("\n" + "="*60)
        console.print(Panel(
            Markdown("**使用方法：**\n- 直接输入问题，如：'帮我计算 2+2'\n- 询问天气：'北京今天天气怎么样？'\n- 网络搜索：'最新人工智能新闻'\n- 输入 'exit' 退出"),
            title="[cyan]使用说明[/cyan]",
            border_style="cyan"
        ))

        while True:
            try:
                user_input = input("\n消息: ").strip()
                if not user_input:
                    continue
                elif user_input.lower() in ['exit', 'quit']:
                    break

                console.print(f"\n[bold yellow]用户输入：{user_input}[/bold yellow]")

                # 运行对话
                response = agent.chat_with_tools(user_input) if hasattr(agent, 'chat_with_tools') else agent.chat(user_input)
                console.print(f"\n[bold green]Agent 回复：[/bold green]")
                console.print(Markdown(response))

            except KeyboardInterrupt:
                console.print("\n\n[bold yellow]已退出[/bold yellow]")
                break
            except Exception as e:
                console.print(f"\n[bold red]错误：{str(e)}[/bold red]")
    else:
        # 批量测试模式
        console.print(f"\n[bold cyan]开始测试 {llm_provider} 提供者[/bold cyan]")

        for user_input in test_inputs:
            if user_input.lower() in ['exit', 'quit']:
                continue

            console.print(f"\n[bold yellow]测试输入：{user_input}[/bold yellow]")

            try:
                if hasattr(agent, 'chat_with_tools'):
                    response = agent.chat_with_tools(user_input)
                else:
                    response = agent.chat(user_input)

                console.print(f"\n[bold green]Agent 回复：[/bold green]")
                console.print(Markdown(response))
                console.print("\n" + "-"*60)
            except Exception as e:
                console.print(f"\n[bold red]错误：{str(e)}[/bold red]")
                console.print("\n" + "-"*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milo Agent 完整演示")
    parser.add_argument("--provider", "-p", choices=["qwen", "glm", "deepseek", "ollama"],
                       default="qwen", help="LLM 提供者")
    parser.add_argument("inputs", nargs="*", help="测试输入（可选）")

    args = parser.parse_args()
    demo(args.provider, args.inputs)