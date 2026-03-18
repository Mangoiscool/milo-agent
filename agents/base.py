"""
BaseAgent - Agent 抽象基类

提供所有 Agent 的通用功能：
- 对话管理（同步/异步/流式）
- 工具注册与调用
- 事件系统
- 记忆管理
"""

from abc import ABC
from enum import Enum
from typing import AsyncIterator, Callable, Dict, List, Optional

from agents.agent_config import AgentConfig
from core.llm.base import BaseLLM, LLMResponse, Message, Role
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.memory.short_term import ShortTermMemory
from core.tools.base import BaseTool
from core.tools.registry import ToolRegistry


class AgentEvent(str, Enum):
    """Agent 事件类型"""
    BEFORE_CHAT = "before_chat"          # 发送用户输入前
    AFTER_CHAT = "after_chat"            # 收到 LLM 响应后
    STREAM_START = "stream_start"        # 流式输出开始
    STREAM_CHUNK = "stream_chunk"        # 流式输出每个 chunk
    STREAM_END = "stream_end"            # 流式输出结束
    MEMORY_PRUNED = "memory_pruned"      # 记忆被裁剪
    TOOL_CALL = "tool_call"              # 工具被调用
    TOOL_RESULT = "tool_result"          # 工具返回结果


class BaseAgent(ABC):
    """
    Agent 抽象基类

    提供所有 Agent 的核心功能：
    - 对话接口（同步/异步/流式）
    - 工具管理与调用
    - 事件系统
    - 记忆管理

    使用示例：
        class MyAgent(BaseAgent):
            def __init__(self, llm, **kwargs):
                super().__init__(llm, **kwargs)

        # 使用
        agent = MyAgent(llm, system_prompt="You are helpful")
        response = agent.chat("Hello!")
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        max_tool_iterations: int = 5
    ):
        """
        初始化 Agent

        Args:
            llm: LLM 实例
            memory: 记忆系统实例（默认使用 ShortTermMemory）
            tools: 初始工具列表
            system_prompt: 系统提示词
            config: Agent 配置
            max_tool_iterations: 最大工具调用迭代次数
        """
        # 核心组件
        self.llm = llm
        self.config = config or AgentConfig()

        # 记忆系统
        if memory is None:
            self.memory = ShortTermMemory(
                max_messages=self.config.max_memory_messages,
                use_intelligent_pruning=self.config.use_intelligent_pruning
            )
        else:
            self.memory = memory

        # 系统提示词
        self.system_prompt = system_prompt or self.config.system_prompt

        # 工具注册中心
        self.tool_registry = ToolRegistry()
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

        # 工具调用配置
        self.max_tool_iterations = max_tool_iterations

        # 日志
        self.logger = get_logger(self.__class__.__name__)

        # 事件处理器
        self._handlers: Dict[AgentEvent, List[Callable]] = {}

    # ═══════════════════════════════════════════════════════════════
    # 事件系统
    # ═══════════════════════════════════════════════════════════════

    def on(self, event: AgentEvent, handler: Callable) -> None:
        """
        注册事件处理器

        Args:
            event: 事件类型
            handler: 处理函数

        Example:
            agent.on(AgentEvent.AFTER_CHAT, lambda response: print(response))
            agent.on(AgentEvent.TOOL_CALL, lambda name, args: print(f"Calling {name}"))
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def off(self, event: AgentEvent, handler: Callable) -> bool:
        """
        移除事件处理器

        Args:
            event: 事件类型
            handler: 处理函数

        Returns:
            是否成功移除
        """
        if event in self._handlers and handler in self._handlers[event]:
            self._handlers[event].remove(handler)
            return True
        return False

    def _emit(self, event: AgentEvent, **kwargs) -> None:
        """
        触发事件

        Args:
            event: 事件类型
            **kwargs: 传递给处理器的参数
        """
        for handler in self._handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception as e:
                self.logger.warning(f"Event handler failed for {event.value}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 工具管理
    # ═══════════════════════════════════════════════════════════════

    def register_tool(self, tool: BaseTool) -> None:
        """注册单个工具"""
        self.tool_registry.register(tool)

    def register_tools(self, tools: List[BaseTool]) -> None:
        """批量注册工具"""
        for tool in tools:
            self.tool_registry.register(tool)

    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        return self.tool_registry.unregister(name)

    def list_tools(self) -> List[str]:
        """列出所有已注册的工具"""
        return self.tool_registry.list_tools()

    # ═══════════════════════════════════════════════════════════════
    # 消息构建
    # ═══════════════════════════════════════════════════════════════

    def _build_messages(self) -> List[Message]:
        """
        构建发送给 LLM 的消息列表

        组合系统提示词和记忆中的消息

        Returns:
            完整的消息列表
        """
        messages = []

        # 添加系统提示词
        if self.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))

        # 添加记忆中的消息
        messages.extend(self.memory.get_all())

        return messages

    # ═══════════════════════════════════════════════════════════════
    # 基础对话接口
    # ═══════════════════════════════════════════════════════════════

    def chat(self, user_input: str) -> str:
        """
        同步对话

        Args:
            user_input: 用户输入

        Returns:
            Agent 响应
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="sync")

        self.logger.info(f"User input: {user_input[:100]}...")

        # 1. 添加用户消息到记忆
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 2. 构建消息列表
        messages = self._build_messages()

        # 3. 调用 LLM
        response = self.llm.chat(messages)

        # 4. 添加助手消息到记忆
        assistant_message = Message(role=Role.ASSISTANT, content=response.content)
        self.memory.add(assistant_message)

        self.logger.info(f"Agent response: {response.content[:100]}...")

        # 5. 触发事件并返回
        self._emit(AgentEvent.AFTER_CHAT, response=response.content, mode="sync")
        return response.content

    async def achat(self, user_input: str) -> str:
        """
        异步对话

        Args:
            user_input: 用户输入

        Returns:
            Agent 响应
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="async")

        self.logger.info(f"User input (async): {user_input[:100]}...")

        # 1. 添加用户消息
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 2. 构建消息列表
        messages = self._build_messages()

        # 3. 调用 LLM
        response = await self.llm.achat(messages)

        # 4. 添加助手消息
        assistant_message = Message(role=Role.ASSISTANT, content=response.content)
        self.memory.add(assistant_message)

        self.logger.info(f"Agent response (async): {response.content[:100]}...")

        # 5. 触发事件并返回
        self._emit(AgentEvent.AFTER_CHAT, response=response.content, mode="async")
        return response.content

    async def astream(self, user_input: str) -> AsyncIterator[str]:
        """
        异步流式对话

        Args:
            user_input: 用户输入

        Yields:
            响应的每个 chunk
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="stream")
        self.logger.info(f"User input (stream): {user_input[:100]}...")

        # 1. 添加用户消息
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 2. 构建消息列表
        messages = self._build_messages()

        # 3. 流式调用 LLM
        full_response = []

        try:
            self._emit(AgentEvent.STREAM_START, user_input=user_input)
            async for chunk in self.llm.astream(messages):
                full_response.append(chunk)
                self._emit(AgentEvent.STREAM_CHUNK, chunk=chunk)
                yield chunk
            self._emit(AgentEvent.STREAM_END, complete="".join(full_response))
        except Exception as e:
            if self.config.enable_stream_fallback:
                self.logger.warning(f"Streaming failed: {e}, falling back to async chat")
                self._emit(AgentEvent.STREAM_END, complete=None, error=str(e))

                # 回退到异步调用
                response = await self.llm.achat(messages)
                if response.content:
                    yield response.content
                    full_response = [response.content]
                else:
                    self.logger.error("Async chat returned empty response")
                    full_response = []
            else:
                raise

        # 4. 添加助手消息
        complete_response = "".join(full_response)
        assistant_message = Message(role=Role.ASSISTANT, content=complete_response)
        self.memory.add(assistant_message)

        self.logger.info(f"Agent response (stream): {complete_response[:100]}...")
        self._emit(AgentEvent.AFTER_CHAT, response=complete_response, mode="stream")

    # ═══════════════════════════════════════════════════════════════
    # 工具调用接口
    # ═══════════════════════════════════════════════════════════════

    def chat_with_tools(self, user_input: str) -> str:
        """
        对话（支持工具调用）

        自动处理工具调用循环：
        1. LLM 返回 tool_calls
        2. 执行工具
        3. 将结果喂回 LLM
        4. 重复直到 LLM 返回普通响应

        Args:
            user_input: 用户输入

        Returns:
            Agent 响应
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="tools")

        self.logger.info(f"User input (with tools): {user_input[:100]}...")

        # 1. 添加用户消息
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 2. 获取工具定义
        tools = None
        if self.tool_registry.count() > 0:
            tools = self.tool_registry.get_all_definitions()

        # 3. 工具调用循环
        for iteration in range(self.max_tool_iterations):
            messages = self._build_messages()

            # 调用 LLM
            response = self.llm.chat_with_tools(messages, tools=tools)

            self.logger.debug(
                f"LLM response: finish_reason={response.finish_reason}, "
                f"tool_calls={len(response.tool_calls)}"
            )

            # 如果没有工具调用，返回结果
            if not response.tool_calls:
                # 保存 assistant 消息
                self.memory.add(Message(role=Role.ASSISTANT, content=response.content))

                self.logger.info(f"Agent response (final): {response.content[:100]}...")
                self._emit(AgentEvent.AFTER_CHAT, response=response.content, mode="tools")
                return response.content

            # 有工具调用，先保存 assistant 消息（包含 tool_calls）
            self.memory.add(Message(
                role=Role.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls
            ))

            # 执行所有工具调用
            for tool_call in response.tool_calls:
                self.logger.info(f"Tool call: {tool_call.name}({tool_call.arguments})")
                self._emit(AgentEvent.TOOL_CALL, name=tool_call.name, arguments=tool_call.arguments)

                # 执行工具
                result = self.tool_registry.execute(tool_call.name, **tool_call.arguments)

                self.logger.info(
                    f"Tool result: {result.content[:100] if result.content else 'empty'}..."
                )
                self._emit(
                    AgentEvent.TOOL_RESULT,
                    name=tool_call.name,
                    result=result.content,
                    is_error=result.is_error
                )

                # 添加工具结果到记忆
                self.memory.add(Message(
                    role=Role.TOOL,
                    content=result.content if not result.is_error else f"Error: {result.error_message}",
                    name=tool_call.name,
                    tool_call_id=tool_call.id
                ))

        # 超过最大迭代次数
        error_msg = "抱歉，工具调用次数超过限制，请简化您的问题或稍后再试。"
        self.memory.add(Message(role=Role.ASSISTANT, content=error_msg))
        self._emit(AgentEvent.AFTER_CHAT, response=error_msg, mode="tools")
        return error_msg

    async def achat_with_tools(self, user_input: str) -> str:
        """
        异步对话（支持工具调用）
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="tools_async")

        self.logger.info(f"User input (async with tools): {user_input[:100]}...")

        # 1. 添加用户消息
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 2. 获取工具定义
        tools = None
        if self.tool_registry.count() > 0:
            tools = self.tool_registry.get_all_definitions()

        # 3. 工具调用循环
        for iteration in range(self.max_tool_iterations):
            messages = self._build_messages()

            # 调用 LLM
            response = await self.llm.achat_with_tools(messages, tools=tools)

            self.logger.debug(
                f"LLM response: finish_reason={response.finish_reason}, "
                f"tool_calls={len(response.tool_calls)}"
            )

            # 如果没有工具调用，返回结果
            if not response.tool_calls:
                self.memory.add(Message(role=Role.ASSISTANT, content=response.content))

                self.logger.info(f"Agent response (final): {response.content[:100]}...")
                self._emit(AgentEvent.AFTER_CHAT, response=response.content, mode="tools_async")
                return response.content

            # 保存 assistant 消息
            self.memory.add(Message(
                role=Role.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls
            ))

            # 执行所有工具调用
            for tool_call in response.tool_calls:
                self.logger.info(f"Tool call: {tool_call.name}({tool_call.arguments})")
                self._emit(AgentEvent.TOOL_CALL, name=tool_call.name, arguments=tool_call.arguments)

                result = await self.tool_registry.aexecute(tool_call.name, **tool_call.arguments)

                self.logger.info(
                    f"Tool result: {result.content[:100] if result.content else 'empty'}..."
                )
                self._emit(
                    AgentEvent.TOOL_RESULT,
                    name=tool_call.name,
                    result=result.content,
                    is_error=result.is_error
                )

                self.memory.add(Message(
                    role=Role.TOOL,
                    content=result.content if not result.is_error else f"Error: {result.error_message}",
                    name=tool_call.name,
                    tool_call_id=tool_call.id
                ))

        error_msg = "抱歉，工具调用次数超过限制，请简化您的问题或稍后再试。"
        self.memory.add(Message(role=Role.ASSISTANT, content=error_msg))
        self._emit(AgentEvent.AFTER_CHAT, response=error_msg, mode="tools_async")
        return error_msg

    # ═══════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════

    def clear_history(self) -> None:
        """清空对话历史"""
        self.memory.clear()
        self.logger.info("Conversation history cleared")

    def get_history(self) -> List[Message]:
        """获取对话历史"""
        return self.memory.get_all()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} llm={self.llm} memory={self.memory} tools={self.tool_registry.count()}>"