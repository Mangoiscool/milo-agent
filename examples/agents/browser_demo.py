"""Browser Agent 演示

展示浏览器自动化的基本用法。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio

from agents.browser import BrowserAgent, browse
from core.browser import BrowserConfig
from core.llm.factory import create_llm


async def demo_browser_agent():
    """Browser Agent 演示"""
    print("=" * 60)
    print("Browser Agent 演示")
    print("=" * 60)

    # 创建 LLM
    llm = create_llm("ollama", model="qwen3.5:4b", think=False)

    # 创建浏览器配置
    browser_config = BrowserConfig(
        headless=False,  # 显示浏览器窗口
        viewport_width=1280,
        viewport_height=720
    )

    # 创建 Agent
    agent = BrowserAgent(llm, browser_config=browser_config)

    try:
        # 初始化
        print("\n[1] 初始化浏览器...")
        await agent.initialize()

        # 获取初始页面状态
        print("[2] 获取页面状态...")
        page_state = await agent.get_page_state()
        print(f"当前页面:\n{page_state[:500]}...")

        # 执行任务
        print("\n[3] 执行任务: 导航到百度...")
        result = await agent.execute_simple("导航到 https://www.baidu.com")
        print(f"结果: {result}")

        # 等待一下
        await asyncio.sleep(2)

        # 获取页面文本
        print("\n[4] 获取页面内容...")
        page_state = await agent.get_page_state()
        print(f"页面状态:\n{page_state[:500]}...")

    finally:
        # 关闭
        print("\n[5] 关闭浏览器...")
        await agent.close()

    print("\n演示完成！")


async def demo_quick_browse():
    """快速浏览演示"""
    print("=" * 60)
    print("快速浏览演示")
    print("=" * 60)

    llm = create_llm("ollama", model="qwen3.5:4b", think=False)

    # 使用便捷函数
    result = await browse(
        llm,
        command="打开 https://example.com 并告诉我页面标题",
        headless=True
    )

    print(f"结果: {result}")


async def demo_search():
    """搜索演示"""
    print("=" * 60)
    print("搜索演示")
    print("=" * 60)

    llm = create_llm("ollama", model="qwen3.5:4b", think=False)

    browser_config = BrowserConfig(headless=False)

    async with BrowserAgent(llm, browser_config=browser_config) as agent:
        # 复杂任务：搜索
        result = await agent.execute("""
        请帮我完成以下任务：
        1. 导航到百度
        2. 在搜索框输入 "Python 编程"
        3. 点击搜索按钮
        4. 告诉我搜索结果页面的标题
        """)

        print(f"结果: {result}")


if __name__ == "__main__":
    # 基础演示
    asyncio.run(demo_browser_agent())

    # 快速浏览（可选）
    # asyncio.run(demo_quick_browse())

    # 搜索演示（可选，需要更复杂的交互）
    # asyncio.run(demo_search())