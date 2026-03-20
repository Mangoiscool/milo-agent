"""
ReAct 推理数据结构

ReAct = Reasoning + Acting

核心概念：
- Thought: 思考当前状态，规划下一步
- Action: 执行具体动作（调用工具）
- Observation: 观察执行结果
- 循环直到问题解决

使用示例：
    trace = ReActTrace(steps=[])
    
    # 记录思考
    thought = ThoughtStep(content="用户询问天气，需要调用天气工具")
    trace.steps.append(thought)
    
    # 记录行动
    action = ActionStep(
        tool_name="weather",
        arguments={"city": "北京"},
        thought=thought
    )
    trace.steps.append(action)
    
    # 记录观察
    observation = ObservationStep(
        result="晴天，25°C",
        is_error=False,
        action=action
    )
    trace.steps.append(observation)
    
    # 转换为 prompt
    prompt = trace.to_prompt()
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ThoughtStep:
    """
    思考步骤
    
    记录 Agent 的思考过程
    """
    content: str                          # 思考内容
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Thought: {preview}"
    
    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        return f"Thought: {self.content}"


@dataclass
class ActionStep:
    """
    行动步骤
    
    记录 Agent 执行的工具调用
    """
    tool_name: str                        # 工具名称
    arguments: Dict[str, Any]             # 工具参数
    thought: Optional[ThoughtStep] = None # 关联的思考步骤
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def __repr__(self) -> str:
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"Action: {self.tool_name}({args_str})"
    
    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        import json
        args_json = json.dumps(self.arguments, ensure_ascii=False)
        return f"Action: {self.tool_name}\nAction Input: {args_json}"


@dataclass
class ObservationStep:
    """
    观察步骤
    
    记录工具执行的结果
    """
    result: str                           # 执行结果
    is_error: bool = False                # 是否错误
    action: Optional[ActionStep] = None   # 关联的行动步骤
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def __repr__(self) -> str:
        status = "Error" if self.is_error else "OK"
        preview = self.result[:50] + "..." if len(self.result) > 50 else self.result
        return f"Observation [{status}]: {preview}"
    
    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        if self.is_error:
            return f"Observation: Error: {self.result}"
        return f"Observation: {self.result}"


@dataclass
class ReActTrace:
    """
    ReAct 执行轨迹
    
    记录完整的思考-行动-观察循环
    """
    steps: List[Any] = field(default_factory=list)  # 步骤列表
    
    def add_thought(self, content: str) -> ThoughtStep:
        """添加思考步骤"""
        step = ThoughtStep(content=content)
        self.steps.append(step)
        return step
    
    def add_action(self, tool_name: str, arguments: Dict[str, Any], thought: Optional[ThoughtStep] = None) -> ActionStep:
        """添加行动步骤"""
        step = ActionStep(tool_name=tool_name, arguments=arguments, thought=thought)
        self.steps.append(step)
        return step
    
    def add_observation(self, result: str, is_error: bool = False, action: Optional[ActionStep] = None) -> ObservationStep:
        """添加观察步骤"""
        step = ObservationStep(result=result, is_error=is_error, action=action)
        self.steps.append(step)
        return step
    
    def to_prompt(self) -> str:
        """
        转换为 LLM 可理解的格式
        
        将所有步骤转换为连续的文本，供 LLM 理解之前的执行过程
        """
        lines = []
        for step in self.steps:
            if isinstance(step, ThoughtStep):
                lines.append(step.to_prompt())
            elif isinstance(step, ActionStep):
                lines.append(step.to_prompt())
            elif isinstance(step, ObservationStep):
                lines.append(step.to_prompt())
        return "\n".join(lines)
    
    def get_thoughts(self) -> List[ThoughtStep]:
        """获取所有思考步骤"""
        return [s for s in self.steps if isinstance(s, ThoughtStep)]
    
    def get_actions(self) -> List[ActionStep]:
        """获取所有行动步骤"""
        return [s for s in self.steps if isinstance(s, ActionStep)]
    
    def get_observations(self) -> List[ObservationStep]:
        """获取所有观察步骤"""
        return [s for s in self.steps if isinstance(s, ObservationStep)]
    
    def get_last_action(self) -> Optional[ActionStep]:
        """获取最后一个行动步骤"""
        for step in reversed(self.steps):
            if isinstance(step, ActionStep):
                return step
        return None
    
    def get_last_observation(self) -> Optional[ObservationStep]:
        """获取最后一个观察步骤"""
        for step in reversed(self.steps):
            if isinstance(step, ObservationStep):
                return step
        return None
    
    def count_iterations(self) -> int:
        """
        计算 ReAct 循环迭代次数
        
        每个完整的 Thought-Action-Observation 算一次迭代
        """
        return len(self.get_actions())
    
    def clear(self) -> None:
        """清空轨迹"""
        self.steps.clear()
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __repr__(self) -> str:
        thoughts = len(self.get_thoughts())
        actions = len(self.get_actions())
        observations = len(self.get_observations())
        return f"<ReActTrace iterations={actions} thoughts={thoughts} observations={observations}>"