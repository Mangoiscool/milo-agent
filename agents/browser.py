"""Browser Agent - 浏览器自动化 Agent

使用 LLM 驱动的浏览器自动化。
"""

import asyncio
from typing import Optional

from agents.agent_config import AgentConfig
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
from core.tools.base import BaseTool
from core.tools.registry import ToolRegistry


class BrowserAgent:
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
        browser_config: Optional[BrowserConfig] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    ):
        """
        初始化 Browser Agent

        Args:
            llm: LLM 实例
            browser_config: 浏览器配置
            config: Agent 配置
            system_prompt: 自定义系统提示词
            max_iterations: 最大迭代次数
        """
        self.llm = llm
        self.browser_config = browser_config or BrowserConfig()
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_iterations = max_iterations

        self.logger = get_logger(self.__class__.__name__)

        # 浏览器控制器
        self.controller = BrowserController(self.browser_config)

        # 工具注册
        self.tool_registry = ToolRegistry()
        self._register_tools()

        # 对话历史
        self.conversation_history: list[Message] = []

    def _register_tools(self):
        """注册浏览器工具"""
        tools: list[BaseTool] = [
            BrowserNavigateTool(self.controller),
            BrowserClickTool(self.controller),
            BrowserTypeTool(self.controller),
            BrowserScrollTool(self.controller),
            BrowserGetTextTool(self.controller),
            BrowserScreenshotTool(self.controller),
            BrowserWaitTool(self.controller),
            BrowserBackTool(self.controller),
        ]

        for tool in tools:
            self.tool_registry.register(tool)

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

        Args:
            command: 用户指令

        Returns:
            执行结果
        """
        self.logger.info(f"Executing command: {command}")

        # 获取初始页面状态
        page_state = await self.get_page_state()

        # 构建消息
        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.system_prompt.format(page_state=page_state)
            ),
            Message(role=Role.USER, content=command)
        ]

        iteration = 0
        last_response = ""

        while iteration < self.max_iterations:
            iteration += 1
            self.logger.info(f"Iteration {iteration}/{self.max_iterations}")

            # 调用 LLM
            tools = self.tool_registry.get_all_definitions()
            response = await self.llm.achat_with_tools(messages, tools)

            # 检查是否需要工具调用
            if not response.tool_calls:
                # 没有工具调用，任务完成
                last_response = response.content
                break

            # 添加 assistant 消息
            messages.append(Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls
            ))

            # 执行工具调用
            for tool_call in response.tool_calls:
                self.logger.info(f"Executing tool: {tool_call.name}")

                # 执行工具
                result = await self.tool_registry.aexecute(
                    tool_call.name,
                    **tool_call.arguments
                )

                # 添加工具结果
                messages.append(Message(
                    role=Role.TOOL,
                    content=result.content,
                    name=tool_call.name,
                    tool_call_id=tool_call.id
                ))

                self.logger.info(f"Tool result: {result.content[:100]}...")

            # 更新页面状态
            try:
                new_page_state = await self.get_page_state()
                # 在下一轮对话中更新状态
                messages[0] = Message(
                    role=Role.SYSTEM,
                    content=self.system_prompt.format(page_state=new_page_state)
                )
            except Exception as e:
                self.logger.warning(f"Failed to get page state: {e}")

        if iteration >= self.max_iterations:
            last_response = f"达到最大迭代次数 ({self.max_iterations})，任务可能未完成。最后状态：{last_response}"

        # 保存对话历史
        self.conversation_history.extend(messages)

        return last_response

    async def execute_simple(self, command: str) -> str:
        """
        简化版执行（单次 LLM 调用）

        Args:
            command: 用户指令

        Returns:
            执行结果
        """
        page_state = await self.get_page_state()

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.system_prompt.format(page_state=page_state)
            ),
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

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


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