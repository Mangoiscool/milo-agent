"""
ReAct Agent - 推理与行动结合

ReAct = Reasoning + Acting

核心特性：
- 显式思考过程（Thought）
- 工具调用（Action）
- 结果观察（Observation）
- 执行轨迹追踪

与 ToolAgent 的区别：
- ToolAgent：直接调用工具，LLM 决定调用什么
- ReActAgent：显式思考步骤，可追踪推理过程

使用示例：
    from agents import ReActAgent
    from core.tools import CalculatorTool, WeatherTool
    
    agent = ReActAgent(
        llm=llm,
        tools=[CalculatorTool(), WeatherTool()]
    )
    
    # 普通对话
    response = agent.chat("帮我算一下 2+3")
    
    # 显示思考过程
    response = agent.chat("北京今天气温多少？明天降温5度后呢？", show_reasoning=True)
    # 输出：
    # Thought: 用户询问北京气温和计算...
    # Action: weather(city="北京")
    # Observation: 晴天，25°C
    # Thought: 今天25度，明天降温5度...
    # Action: calculator(expression="25 - 5")
    # Observation: 20
    # Final Answer: 北京今天25°C，明天降温5度后是20°C
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from agents.base import AgentEvent, BaseAgent
from core.llm.base import BaseLLM, LLMResponse, Message, Role
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.memory.short_term import ShortTermMemory
from core.reasoning.react import (
    ActionStep,
    ObservationStep,
    ReActTrace,
    ThoughtStep,
)
from core.tools.base import BaseTool
from agents.agent_config import AgentConfig


# ═══════════════════════════════════════════════════════════════
# ReAct Prompt 模板
# ═══════════════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = """你是一个智能助手，可以使用工具帮助用户解决问题。

请按照以下格式思考和行动：

1. Thought: 思考当前情况，分析用户问题，规划下一步行动
2. Action: 选择一个工具执行（如果需要）
3. Action Input: 提供工具参数（JSON 格式）
4. Observation: 观察工具返回的结果
5. 重复 Thought-Action-Observation 直到问题解决
6. Final Answer: 给出最终答案

重要规则：
- 每次只能执行一个 Action
- 必须等待 Observation 后再进行下一步思考
- 如果不需要工具，直接给出 Final Answer
- Final Answer 必须清晰、完整地回答用户问题

可用工具：
{tools}

示例 1 - 简单查询：
Question: 北京今天天气怎么样？
Thought: 用户询问北京今天的天气，我需要调用天气工具查询。
Action: weather
Action Input: {{"city": "北京"}}
Observation: {{"temperature": 25, "condition": "晴天"}}
Thought: 已经获取天气信息，可以回答用户了。
Final Answer: 北京今天晴天，气温25°C。

示例 2 - 多步骤推理：
Question: 100的平方根是多少？再加10呢？
Thought: 用户需要计算平方根然后加10，我需要分两步计算。
Action: calculator
Action Input: {{"expression": "sqrt(100)"}}
Observation: 10
Thought: 100的平方根是10，现在需要计算10+10。
Action: calculator
Action Input: {{"expression": "10 + 10"}}
Observation: 20
Thought: 已经完成计算，结果是20。
Final Answer: 100的平方根是10，再加10等于20。

示例 3 - 不需要工具：
Question: 你好，你是谁？
Thought: 这是简单的问候，不需要使用工具。
Final Answer: 你好！我是一个智能助手，可以帮助你解答问题、查询信息、进行计算等。有什么我可以帮你的吗？

记住：遵循 Thought -> Action -> Action Input -> Observation -> ... -> Final Answer 的格式。"""


class ReActAgent(BaseAgent):
    """
    ReAct Agent - 推理与行动结合
    
    特性：
    - 显式思考过程（Thought）
    - 工具调用（Action）
    - 结果观察（Observation）
    - 执行轨迹追踪
    
    使用示例：
        agent = ReActAgent(llm, tools=[weather_tool, search_tool])
        response = agent.chat("帮我查一下北京天气和新闻")
        # 可以看到思考过程：
        # Thought: 用户需要天气和新闻...
        # Action: weather(city="北京")
        # Observation: {...}
        # Thought: 还需要获取新闻...
        # Action: web_search(query="北京新闻")
        # ...
    """
    
    DEFAULT_SYSTEM_PROMPT = REACT_SYSTEM_PROMPT
    
    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        max_react_iterations: int = 10,
        max_tool_iterations: int = 5
    ):
        """
        初始化 ReAct Agent
        
        Args:
            llm: LLM 实例
            memory: 记忆系统实例（默认使用 ShortTermMemory）
            tools: 初始工具列表
            system_prompt: 系统提示词
            config: Agent 配置
            max_react_iterations: 最大 ReAct 循环次数
            max_tool_iterations: 单次工具调用最大迭代次数
        """
        super().__init__(
            llm=llm,
            memory=memory,
            tools=tools,
            system_prompt=system_prompt,
            config=config,
            max_tool_iterations=max_tool_iterations
        )
        
        # ReAct 执行轨迹
        self.trace = ReActTrace(steps=[])
        self.max_react_iterations = max_react_iterations
        
        # 缓存格式化的工具描述
        self._tools_description: Optional[str] = None
    
    # ═══════════════════════════════════════════════════════════════
    # 核心对话接口
    # ═══════════════════════════════════════════════════════════════
    
    def chat(self, user_input: str, show_reasoning: bool = False) -> str:
        """
        ReAct 对话
        
        Args:
            user_input: 用户输入
            show_reasoning: 是否返回思考过程
        
        Returns:
            Agent 响应（可选包含思考过程）
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="react")
        self.logger.info(f"User input (ReAct): {user_input[:100]}...")
        
        # 1. 初始化轨迹
        self.trace = ReActTrace(steps=[])
        iteration = 0
        
        # 2. ReAct 循环
        while iteration < self.max_react_iterations:
            iteration += 1
            
            # 构建 ReAct Prompt
            prompt = self._build_react_prompt(user_input)
            
            # 3. 调用 LLM
            messages = [Message(role=Role.USER, content=prompt)]
            response = self.llm.chat(messages)
            content = response.content
            
            self.logger.debug(f"ReAct iteration {iteration}: {content[:200]}...")
            
            # 4. 解析 Thought 和 Action
            thought, action, action_input = self._parse_thought_action(content)
            
            # 记录 Thought
            if thought:
                thought_step = self.trace.add_thought(thought)
                self.logger.info(f"Thought: {thought[:100]}...")
            
            # 5. 检查是否完成
            final_answer = self._extract_final_answer(content)
            if final_answer:
                self.logger.info(f"Final Answer: {final_answer[:100]}...")
                
                # 保存到记忆
                self.memory.add(Message(role=Role.USER, content=user_input))
                self.memory.add(Message(role=Role.ASSISTANT, content=final_answer))
                
                self._emit(AgentEvent.AFTER_CHAT, response=final_answer, mode="react")
                
                if show_reasoning:
                    reasoning = self.trace.to_prompt()
                    return f"{reasoning}\n\nFinal Answer: {final_answer}"
                return final_answer
            
            # 6. 执行 Action
            if action:
                # 检查工具是否存在
                if not self.tool_registry.has(action):
                    error_msg = f"工具 '{action}' 不存在。可用工具: {', '.join(self.tool_registry.list_tools())}"
                    self.trace.add_observation(error_msg, is_error=True)
                    self.logger.warning(error_msg)
                    continue
                
                # 记录 Action
                action_step = self.trace.add_action(
                    tool_name=action,
                    arguments=action_input or {}
                )
                self.logger.info(f"Action: {action}({action_input})")
                
                # 执行工具
                self._emit(AgentEvent.TOOL_CALL, name=action, arguments=action_input)
                result = self.tool_registry.execute(action, **(action_input or {}))
                
                # 记录 Observation
                obs_result = result.content if not result.is_error else f"Error: {result.error_message}"
                self.trace.add_observation(
                    result=obs_result,
                    is_error=result.is_error,
                    action=action_step
                )
                
                self._emit(
                    AgentEvent.TOOL_RESULT,
                    name=action,
                    result=result.content,
                    is_error=result.is_error
                )
                self.logger.info(f"Observation: {obs_result[:100]}...")
            else:
                # 没有 Action 也没有 Final Answer，可能 LLM 输出格式有问题
                self.logger.warning(f"Unexpected LLM output format: {content[:200]}")
                # 直接返回 LLM 的响应
                self.memory.add(Message(role=Role.USER, content=user_input))
                self.memory.add(Message(role=Role.ASSISTANT, content=content))
                self._emit(AgentEvent.AFTER_CHAT, response=content, mode="react")
                return content
        
        # 超过最大迭代次数
        error_msg = "抱歉，思考过程太长，请简化问题或稍后再试。"
        self.logger.warning(f"ReAct exceeded max iterations ({self.max_react_iterations})")
        
        self.memory.add(Message(role=Role.USER, content=user_input))
        self.memory.add(Message(role=Role.ASSISTANT, content=error_msg))
        
        self._emit(AgentEvent.AFTER_CHAT, response=error_msg, mode="react")
        
        if show_reasoning:
            reasoning = self.trace.to_prompt()
            return f"{reasoning}\n\n{error_msg}"
        return error_msg
    
    # ═══════════════════════════════════════════════════════════════
    # Prompt 构建
    # ═══════════════════════════════════════════════════════════════
    
    def _build_react_prompt(self, question: str) -> str:
        """
        构建 ReAct Prompt
        
        Args:
            question: 用户问题
        
        Returns:
            完整的 ReAct prompt
        """
        # 获取工具描述
        tools_desc = self._format_tools()
        
        # 获取之前的执行轨迹
        trace_prompt = self.trace.to_prompt()
        
        # 构建系统提示
        system_prompt = self.system_prompt or self.DEFAULT_SYSTEM_PROMPT
        system_prompt = system_prompt.format(tools=tools_desc)
        
        if trace_prompt:
            # 有之前的执行记录
            prompt = f"""{system_prompt}

之前的执行过程：
{trace_prompt}

继续回答用户问题：{question}
Thought:"""
        else:
            # 首次执行
            prompt = f"""{system_prompt}

Question: {question}
Thought:"""
        
        return prompt
    
    def _format_tools(self) -> str:
        """
        格式化工具描述
        
        Returns:
            工具描述文本
        """
        if self._tools_description:
            return self._tools_description
        
        definitions = self.tool_registry.get_all_definitions()
        if not definitions:
            return "（无可用工具）"
        
        lines = []
        for d in definitions:
            # 获取参数描述
            params_desc = ""
            if d.parameters and "properties" in d.parameters:
                params = []
                for name, prop in d.parameters["properties"].items():
                    desc = prop.get("description", "")
                    required = name in d.parameters.get("required", [])
                    req_str = " (必需)" if required else ""
                    params.append(f"    - {name}: {desc}{req_str}")
                if params:
                    params_desc = "\n参数:\n" + "\n".join(params)
            
            lines.append(f"- {d.name}: {d.description}{params_desc}")
        
        self._tools_description = "\n".join(lines)
        return self._tools_description
    
    # ═══════════════════════════════════════════════════════════════
    # 响应解析
    # ═══════════════════════════════════════════════════════════════
    
    def _parse_thought_action(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """
        解析 LLM 响应，提取 Thought 和 Action
        
        Args:
            response: LLM 响应文本
        
        Returns:
            (thought, action_name, action_input)
        """
        thought = None
        action = None
        action_input = None
        
        # 解析 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Final)|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # 解析 Action
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            action = action_match.group(1).strip()
            
            # 解析 Action Input
            input_match = re.search(r'Action Input:\s*(\{.+?\}|\[.+?\]|".+?"|\'.+?\'|.+$)', response, re.DOTALL)
            if input_match:
                input_str = input_match.group(1).strip()
                
                # 尝试解析 JSON
                try:
                    # 如果是 JSON 格式
                    if input_str.startswith('{') or input_str.startswith('['):
                        action_input = json.loads(input_str)
                    else:
                        # 如果是字符串，尝试作为 JSON 解析
                        try:
                            action_input = json.loads(input_str)
                        except json.JSONDecodeError:
                            # 不是 JSON，作为单个参数
                            action_input = {"query": input_str.strip('"\'')}
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse Action Input: {input_str}, error: {e}")
                    # 解析失败，作为原始字符串
                    action_input = {"query": input_str.strip('"\'')}
        
        return thought, action, action_input
    
    def _extract_final_answer(self, response: str) -> Optional[str]:
        """
        提取最终答案
        
        Args:
            response: LLM 响应文本
        
        Returns:
            最终答案文本，如果没有则返回 None
        """
        match = re.search(r'Final Answer:\s*(.+?)$', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # 工具管理
    # ═══════════════════════════════════════════════════════════════
    
    def register_tool(self, tool: BaseTool) -> None:
        """注册工具（重写以清除缓存）"""
        super().register_tool(tool)
        self._tools_description = None  # 清除缓存
    
    def unregister_tool(self, name: str) -> bool:
        """注销工具（重写以清除缓存）"""
        result = super().unregister_tool(name)
        self._tools_description = None  # 清除缓存
        return result
    
    # ═══════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════
    
    def get_trace(self) -> ReActTrace:
        """获取当前执行轨迹"""
        return self.trace
    
    def get_reasoning_summary(self) -> str:
        """获取推理过程摘要"""
        return self.trace.to_prompt()
    
    def __repr__(self) -> str:
        return f"<ReActAgent llm={self.llm} tools={self.tool_registry.count()} iterations={self.max_react_iterations}>"