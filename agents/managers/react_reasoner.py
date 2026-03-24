"""
ReActReasoner - ReAct 推理引擎

统一的工作流引擎，实现 Thought → Action → Observation → Final Answer 循环。
标准工具调用可以被视为没有显式 Thought 步骤的 ReAct 特例。
"""

import json
import re
from typing import Callable, List, Optional

from core.llm.base import BaseLLM, Message, Role
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.reasoning.react import ActionStep, ObservationStep, ReActTrace, ThoughtStep
from core.tools.registry import ToolRegistry


class ReActReasoner:
    """
    ReAct 推理引擎

    实现统一的推理工作流：
    1. Thought: 分析当前状态，思考下一步
    2. Action: 选择并执行工具
    3. Observation: 观察执行结果
    4. 循环直到获得 Final Answer

    使用示例:
        reasoner = ReActReasoner(
            llm=llm,
            memory=memory,
            tool_registry=tool_registry,
            system_prompt="You are a helpful assistant"
        )

        # 执行推理
        response = reasoner.run("北京今天天气如何？")

        # 显示推理过程
        response = reasoner.run("查询天气", show_reasoning=True)
    """

    # ReAct 格式指令
    REACT_INSTRUCTIONS = """请按照以下 ReAct 格式思考和行动：

1. Thought: 分析当前情况，思考下一步行动
2. Action: 选择要使用的工具（如果需要）
3. Action Input: 提供工具参数（JSON 格式）
4. Observation: 观察工具返回的结果
5. 重复 Thought-Action-Observation 直到问题解决
6. Final Answer: 给出最终答案

格式示例：
Thought: 用户询问北京天气，我需要查询天气信息。
Action: weather
Action Input: {"city": "北京"}
Observation: {"temperature": 25, "condition": "晴天"}
Thought: 已经获取天气信息，可以回答用户了。
Final Answer: 北京今天晴天，气温25°C。"""

    def __init__(
        self,
        llm: BaseLLM,
        memory: BaseMemory,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    ):
        """
        初始化 ReAct 推理引擎

        Args:
            llm: LLM 实例
            memory: 记忆系统
            tool_registry: 工具注册中心
            system_prompt: 系统提示词
            max_iterations: 最大迭代次数
        """
        self.llm = llm
        self.memory = memory
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt or ""
        self.max_iterations = max_iterations
        self.logger = get_logger(self.__class__.__name__)

    def run(
        self,
        user_input: str,
        show_reasoning: bool = False,
        event_emitter: Optional[Callable] = None
    ) -> str:
        """
        执行 ReAct 推理循环

        Args:
            user_input: 用户输入
            show_reasoning: 是否返回推理过程
            event_emitter: 可选的事件发射函数

        Returns:
            Agent 响应
        """
        self.logger.info(f"ReAct run started: {user_input[:100]}...")

        # 初始化轨迹
        trace = ReActTrace(steps=[])

        # 添加用户消息到记忆
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 触发开始事件
        if event_emitter:
            event_emitter("before_chat", user_input=user_input, mode="react")

        # ReAct 循环
        for iteration in range(self.max_iterations):
            # 构建消息
            messages = self._build_messages(user_input, trace)

            # 获取工具定义
            tools = None
            if self.tool_registry.count() > 0:
                tools = self.tool_registry.get_all_definitions()

            # 调用 LLM
            response = self.llm.chat_with_tools(messages, tools=tools)
            content = response.content or ""

            self.logger.debug(f"Iteration {iteration + 1}: {content[:200]}...")

            # 解析响应
            thought, action_name, action_input = self._parse_response(content)

            # 记录 Thought
            if thought:
                trace.steps.append(ThoughtStep(content=thought))
                self.logger.debug(f"Thought: {thought[:100]}...")

            # 检查是否完成
            if "Final Answer:" in content or not action_name:
                final_answer = content.split("Final Answer:")[-1].strip() if "Final Answer:" in content else content.strip()

                # 添加到记忆
                self.memory.add(Message(role=Role.ASSISTANT, content=final_answer))

                self.logger.info(f"Final answer: {final_answer[:100]}...")

                if event_emitter:
                    event_emitter("after_chat", response=final_answer, mode="react")

                if show_reasoning:
                    return f"{trace.to_prompt()}\n\nFinal Answer: {final_answer}"
                return final_answer

            # 执行 Action
            if action_name and action_name in self.tool_registry.list_tools():
                action_step = ActionStep(
                    tool_name=action_name,
                    arguments=action_input or {},
                    thought=ThoughtStep(content=thought) if thought else ThoughtStep(content="")
                )
                trace.steps.append(action_step)

                self.logger.info(f"Action: {action_name}({action_input})")

                if event_emitter:
                    event_emitter("tool_call", name=action_name, arguments=action_input)

                # 执行工具
                result = self.tool_registry.execute(action_name, **action_input)

                self.logger.info(f"Observation: {result.content[:100] if result.content else 'empty'}...")

                if event_emitter:
                    event_emitter("tool_result", name=action_name, result=result.content, is_error=result.is_error)

                # 记录 Observation
                obs_step = ObservationStep(
                    result=result.content if not result.is_error else f"Error: {result.error_message}",
                    is_error=result.is_error,
                    action=action_step
                )
                trace.steps.append(obs_step)

                # 将工具结果添加到记忆
                self.memory.add(Message(
                    role=Role.TOOL,
                    content=result.content if not result.is_error else f"Error: {result.error_message}",
                    name=action_name
                ))

        # 超过最大迭代次数
        error_msg = "抱歉，思考过程太长，请简化问题。"
        self.memory.add(Message(role=Role.ASSISTANT, content=error_msg))

        if event_emitter:
            event_emitter("after_chat", response=error_msg, mode="react")

        return error_msg

    def _build_messages(self, question: str, trace: ReActTrace) -> List[Message]:
        """构建 ReAct Prompt"""
        messages = []

        # 添加系统提示
        system_content = self.system_prompt + "\n\n" + self.REACT_INSTRUCTIONS
        messages.append(Message(role=Role.SYSTEM, content=system_content))

        # 添加工具描述
        if self.tool_registry.count() > 0:
            tools_desc = self._format_tools()
            messages.append(Message(role=Role.SYSTEM, content=f"可用工具：\n{tools_desc}"))

        # 添加历史执行轨迹
        if trace.steps:
            trace_prompt = trace.to_prompt()
            messages.append(Message(role=Role.USER, content=f"之前的执行过程：\n{trace_prompt}\n\n继续回答：{question}"))
        else:
            messages.append(Message(role=Role.USER, content=question))

        return messages

    def _format_tools(self) -> str:
        """格式化工具描述"""
        definitions = self.tool_registry.get_all_definitions()
        lines = []
        for d in definitions:
            lines.append(f"- {d.name}: {d.description}")
            if d.parameters:
                lines.append(f"  参数: {d.parameters}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> tuple:
        """解析 ReAct 响应，提取 Thought、Action 和 Action Input"""
        thought = ""
        action_name = None
        action_input = {}

        # 解析 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?:\nAction:|\nFinal Answer:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 解析 Action
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            action_name = action_match.group(1).strip()

        # 解析 Action Input
        input_match = re.search(r'Action Input:\s*(\{[^}]*\}|[^\n]+)', response)
        if input_match:
            input_str = input_match.group(1).strip()
            try:
                action_input = json.loads(input_str)
            except json.JSONDecodeError:
                # 如果不是 JSON，作为字符串参数
                action_input = {"query": input_str}

        return thought, action_name, action_input
