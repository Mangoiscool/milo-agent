"""Browser Agent - 浏览器自动化 Agent

使用 LLM 驱动的浏览器自动化。

现在继承自 BaseAgent，复用工具调用循环，可以与其他能力组合。
"""

import asyncio
from typing import Optional

from agents.agent_config import AgentConfig
from agents.base import BaseAgent
from core.browser import (
    BrowserConfig,
    BrowserController,
    BrowserNavigateTool,
    BrowserClickTool,
    BrowserTypeTool,
    BrowserScrollTool,
    BrowserGetTextTool,
    BrowserScreenshotTool,
    BrowserWaitTool,
    BrowserBackTool,
)
from core.llm.base import BaseLLM, Message, Role
from core.logger import get_logger
from core.memory.base import BaseMemory


class BrowserAgent(BaseAgent):
    """
    Browser Agent - 浏览器自动化 Agent

    特性：
    - LLM 驱动的网页操作
    - 自动解析页面结构
    - 支持复杂的交互流程
    - 错误恢复和重试

    使用示例：
        llm = create_llm("qwen", api_key="sk-xxx")

        agent = BrowserAgent(llm)
        await agent.initialize()

        # 执行任务
        result = await agent.execute("打开百度并搜索 Python")

        # 清理
        await agent.close()

        # 或使用上下文管理器
        async with BrowserAgent(llm) as agent:
            result = await agent.execute("打开百度并搜索 Python")
    """

    DEFAULT_SYSTEM_PROMPT = """你是一个浏览器自动化助手。
你可以使用浏览器工具来操作网页，完成用户指定的任务。

当前页面状态：
{page_state}

工作原则：
1. 仔细观察页面状态，理解当前页面内容
2. 根据用户指令规划操作步骤
3. 使用合适的工具执行操作
4. 如果遇到问题，尝试其他方法
5. 完成任务后报告结果

注意事项：
- 点击元素前确认元素存在
- 输入文本前先聚焦输入框
- 等待页面加载完成后再进行下一步操作
- 如果操作失败，截图并分析原因"""

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemory] = None,
        browser_config: Optional[BrowserConfig] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    ):
        """
        初始化 Browser Agent

        Args:
            llm: LLM 实例
            memory: 记忆系统
            browser_config: 浏览器配置
            config: Agent 配置
            system_prompt: 自定义系统提示词
            max_iterations: 最大迭代次数
        """
        # 保存 Browser 特有属性
        self.browser_config = browser_config or BrowserConfig()
        self._custom_system_prompt = system_prompt

        # 初始化基类（系统提示词会在 execute 中动态构建）
        super().__init__(
            llm=llm,
            memory=memory,
            tools=None,  # 稍后注册浏览器工具
            system_prompt=None,  # 动态构建
            config=config or AgentConfig(),
            max_tool_iterations=max_iterations
        )

        # 浏览器控制器
        self.controller = BrowserController(self.browser_config)

        # 注册浏览器工具
        self._register_browser_tools()

        # 页面状态缓存
        self._current_page_state: Optional[str] = None

    def _register_browser_tools(self):
        """注册浏览器工具"""
        browser_tools = [
            BrowserNavigateTool(self.controller),
            BrowserClickTool(self.controller),
            BrowserTypeTool(self.controller),
            BrowserScrollTool(self.controller),
            BrowserGetTextTool(self.controller),
            BrowserScreenshotTool(self.controller),
            BrowserWaitTool(self.controller),
            BrowserBackTool(self.controller),
        ]

        for tool in browser_tools:
            self.tool_registry.register(tool)

        self.logger.info(f"Registered {len(browser_tools)} browser tools")

    async def initialize(self):
        """初始化浏览器"""
        self.logger.info("Initializing browser...")
        await self.controller.initialize()
        self.logger.info("Browser initialized")

    async def close(self):
        """关闭浏览器"""
        self.logger.info("Closing browser...")
        await self.controller.close()
        self.logger.info("Browser closed")

    async def get_page_state(self) -> str:
        """获取当前页面状态的描述"""
        page_state = await self.controller.get_page_state()
        return page_state.to_context()

    async def execute(self, command: str) -> str:
        """
        执行浏览器任务

        使用 BaseAgent 的工具调用循环，配合动态页面状态。

        Args:
            command: 用户指令

        Returns:
            执行结果
        """
        self.logger.info(f"Executing command: {command}")

        # 获取初始页面状态
        page_state = await self.get_page_state()
        self._current_page_state = page_state

        # 构建动态系统提示词
        system_prompt = self._build_system_prompt(page_state)

        # 添加用户消息
        user_message = Message(role=Role.USER, content=command)
        self.memory.add(user_message)

        # 获取工具定义
        tools = self.tool_registry.get_all_definitions()

        # 工具调用循环（类似 BaseAgent.chat_with_tools，但更新页面状态）
        last_response = ""

        for iteration in range(self.max_tool_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{self.max_tool_iterations}")

            # 构建消息（包含更新的系统提示词）
            messages = [
                Message(role=Role.SYSTEM, content=system_prompt),
                *self.memory.get_all()
            ]

            # 调用 LLM
            response = await self.llm.achat_with_tools(messages, tools)

            # 检查是否需要工具调用
            if not response.tool_calls:
                # 没有工具调用，任务完成
                last_response = response.content
                # 保存最终响应
                self.memory.add(Message(role=Role.ASSISTANT, content=response.content))
                break

            # 添加 assistant 消息（包含 tool_calls）
            self.memory.add(Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls
            ))

            # 执行所有工具调用
            for tool_call in response.tool_calls:
                self.logger.info(f"Executing tool: {tool_call.name}")
                self._emit("tool_call", name=tool_call.name, arguments=tool_call.arguments)

                # 执行工具
                result = await self.tool_registry.aexecute(
                    tool_call.name,
                    **tool_call.arguments
                )

                self.logger.info(f"Tool result: {result.content[:100] if result.content else 'empty'}...")
                self._emit("tool_result", name=tool_call.name, result=result.content, is_error=result.is_error)

                # 添加工具结果
                self.memory.add(Message(
                    role=Role.TOOL,
                    content=result.content,
                    name=tool_call.name,
                    tool_call_id=tool_call.id
                ))

            # 更新页面状态
            try:
                new_page_state = await self.get_page_state()
                self._current_page_state = new_page_state
                system_prompt = self._build_system_prompt(new_page_state)
            except Exception as e:
                self.logger.warning(f"Failed to get page state: {e}")

        if iteration >= self.max_tool_iterations - 1 and not last_response:
            last_response = f"达到最大迭代次数 ({self.max_iterations})，任务可能未完成。"

        return last_response

    def _build_system_prompt(self, page_state: str) -> str:
        """
        构建包含页面状态的系统提示词

        Args:
            page_state: 当前页面状态描述

        Returns:
            完整的系统提示词
        """
        template = self._custom_system_prompt or self.DEFAULT_SYSTEM_PROMPT
        return template.format(page_state=page_state)

    async def execute_simple(self, command: str) -> str:
        """
        简化版执行（单次 LLM 调用）

        Args:
            command: 用户指令

        Returns:
            执行结果
        """
        page_state = await self.get_page_state()
        system_prompt = self._build_system_prompt(page_state)

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=command)
        ]

        tools = self.tool_registry.get_all_definitions()
        response = await self.llm.achat_with_tools(messages, tools)

        # 执行所有工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                await self.tool_registry.aexecute(
                    tool_call.name,
                    **tool_call.arguments
                )

        return response.content

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    def __repr__(self) -> str:
        return f"<BrowserAgent llm={self.llm} tools={self.tool_registry.count()}>"


# 便捷函数
async def browse(llm: BaseLLM, command: str, headless: bool = True) -> str:
    """
    快速执行浏览器任务

    Args:
        llm: LLM 实例
        command: 指令
        headless: 是否无头模式

    Returns:
        执行结果
    """
    config = BrowserConfig(headless=headless)

    async with BrowserAgent(llm, browser_config=config) as agent:
        return await agent.execute(command)


__all__ = ["BrowserAgent", "browse"]