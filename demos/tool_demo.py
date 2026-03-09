#!/usr/bin/env python3
"""
工具调用演示

演示 SimpleAgent 的 Function Calling 能力

使用方法：
1. 设置环境变量：
   export QWEN_API_KEY="your-api-key"
   
2. 运行演示：
   python demos/tool_demo.py
"""

import asyncio
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm.factory import create_llm
from core.tools import CalculatorTool, DateTimeTool, RandomTool
from agents.simple import SimpleAgent, AgentEvent


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'-'*60}\n")


def demo_basic_tools():
    """演示基础工具调用"""
    print_separator("工具调用演示 - 计算器、日期时间、随机数")
    
    # 1. 创建 LLM
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("❌ 请设置 QWEN_API_KEY 环境变量")
        return
    
    llm = create_llm("qwen", api_key=api_key)
    
    # 2. 创建 Agent（带工具）
    agent = SimpleAgent(
        llm=llm,
        system_prompt="你是一个有用的助手。可以使用以下工具：计算器、日期时间、随机数生成器。",
        tools=[
            CalculatorTool(),
            DateTimeTool(),
            RandomTool()
        ]
    )
    
    # 3. 注册事件监听
    def on_tool_call(name, arguments):
        print(f"🔧 调用工具: {name}({arguments})")
    
    def on_tool_result(name, result, is_error):
        status = "❌" if is_error else "✅"
        print(f"📋 工具结果: {status} {result}")
    
    agent.on(AgentEvent.TOOL_CALL, on_tool_call)
    agent.on(AgentEvent.TOOL_RESULT, on_tool_result)
    
    # 4. 测试对话
    test_cases = [
        "你好！",
        "现在几点了？",
        "帮我算一下 123 + 456",
        "给我一个 1 到 100 之间的随机数",
        "今天几号？我想知道现在的日期",
    ]
    
    for user_input in test_cases:
        print(f"👤 用户: {user_input}")
        response = agent.chat_with_tools(user_input)
        print(f"🤖 Agent: {response}")
        print()
        llm=llm,
        system_prompt="你是一个有用的助手。当需要计算时，使用计算器工具。",
        tools=[CalculatorTool()]
    )
    
    # 3. 注册事件监听
    def on_tool_call(name, arguments):
        print(f"🔧 调用工具: {name}({arguments})")
    
    def on_tool_result(name, result, is_error):
        status = "❌" if is_error else "✅"
        print(f"📋 工具结果: {status} {result}")
    
    agent.on(AgentEvent.TOOL_CALL, on_tool_call)
    agent.on(AgentEvent.TOOL_RESULT, on_tool_result)
    
    # 4. 测试对话
    test_cases = [
        "你好！",
        "帮我算一下 123 + 456",
        "计算 (10 - 3) * 4 的结果",
        "2 的 10 次方是多少？",
    ]
    
    for user_input in test_cases:
        print(f"👤 用户: {user_input}")
        response = agent.chat_with_tools(user_input)
        print(f"🤖 Agent: {response}")
        print()


def demo_tool_registration():
    """演示工具注册"""
    print_separator("工具注册演示")
    
    # 创建空 Agent
    llm = create_llm("qwen", api_key=os.environ.get("QWEN_API_KEY"))
    agent = SimpleAgent(llm=llm)
    
    print(f"初始工具数量: {agent.tool_registry.count()}")
    print(f"工具列表: {agent.list_tools()}")
    
    # 注册工具
    calc_tool = CalculatorTool()
    agent.register_tool(calc_tool)
    
    print(f"\n注册后工具数量: {agent.tool_registry.count()}")
    print(f"工具列表: {agent.list_tools()}")
    
    # 查看工具定义
    tool_def = calc_tool.get_definition()
    print(f"\n工具定义:")
    print(f"  名称: {tool_def.name}")
    print(f"  描述: {tool_def.description[:50]}...")
    print(f"  参数: {tool_def.parameters}")
    
    # 注销工具
    agent.unregister_tool("calculator")
    print(f"\n注销后工具数量: {agent.tool_registry.count()}")


async def demo_async_tools():
    """演示异步工具调用"""
    print_separator("异步工具调用演示")
    
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("❌ 请设置 QWEN_API_KEY 环境变量")
        return
    
    llm = create_llm("qwen", api_key=api_key)
    agent = SimpleAgent(
        llm=llm,
        system_prompt="你是一个有用的助手。需要计算时使用计算器。",
        tools=[CalculatorTool()]
    )
    
    user_input = "请帮我计算 (100 + 200) / 3"
    print(f"👤 用户: {user_input}")
    
    response = await agent.achat_with_tools(user_input)
    print(f"🤖 Agent: {response}")


def main():
    """主函数"""
    print("\n" + "🎩" * 30)
    print("  milo-agent 工具调用演示")
    print("🎩" * 30)
    
    # 检查 API Key
    if not os.environ.get("QWEN_API_KEY"):
        print("\n⚠️  未设置 QWEN_API_KEY 环境变量")
        print("请先设置：export QWEN_API_KEY='your-api-key'\n")
        print("以下演示将跳过 LLM 调用部分...")
        demo_tool_registration()
        return
    
    try:
        # 1. 工具注册演示（不需要 API）
        demo_tool_registration()
        
        # 2. 基础工具调用
        demo_basic_tools()
        
        # 3. 异步工具调用
        asyncio.run(demo_async_tools())
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    print_separator("演示结束")


if __name__ == "__main__":
    main()
