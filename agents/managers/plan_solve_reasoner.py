"""
PlanSolveReasoner - Plan-and-Solve 推理引擎

先制定完整计划，再按步骤执行的推理范式。
支持：
1. 任务分解与规划
2. 按计划逐步执行
3. 执行中的反思与重新规划
"""

import json
import re
from typing import Callable, Dict, List, Optional, Set

from core.llm.base import BaseLLM, Message, Role
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.reasoning.plan_solve import (
    ExecutionStep,
    Plan,
    PlanSolveTrace,
    PlanStep,
    Review,
    StepStatus,
)
from core.tools.registry import ToolRegistry


class PlanSolveReasoner:
    """
    Plan-and-Solve 推理引擎

    实现统一的工作流：
    1. Plan: 分析任务，制定执行计划
    2. Execute: 按计划逐步执行
    3. Review: 执行中反思，必要时重新规划
    4. Final Answer: 汇总结果

    使用示例:
        reasoner = PlanSolveReasoner(
            llm=llm,
            memory=memory,
            tool_registry=tool_registry,
            system_prompt="You are a helpful assistant"
        )

        # 执行推理
        response = reasoner.run("比较北京和上海今天的天气")

        # 显示执行过程
        response = reasoner.run("查询天气", show_planning=True)
    """

    # Plan-and-Solve 格式指令
    PLAN_INSTRUCTIONS = """请按照以下 Plan-and-Solve 格式解决问题：

第一步 - 制定计划：
1. 分析任务，将其分解为清晰的步骤
2. 每个步骤应该是原子的、可执行的
3. 标注步骤间的依赖关系
4. 预估需要的工具

输出格式示例：
计划思考: 用户需要比较两地天气，我需要先分别查询，再比较。
计划步骤:
1. [工具:weather] 查询北京天气 (无依赖)
2. [工具:weather] 查询上海天气 (无依赖)
3. [无需工具] 比较两地天气差异 (依赖: 1, 2)

第二步 - 执行计划：
按步骤顺序执行，格式：
步骤 1 [weather]:
Action Input: {"city": "北京"}
Observation: {"temperature": 25, "condition": "晴天"}
结果: 北京晴天25°C

步骤 2 [weather]:
Action Input: {"city": "上海"}
Observation: {"temperature": 22, "condition": "多云"}
结果: 上海多云22°C

步骤 3:
结果: 北京25°C晴天，比上海22°C多云更暖和。

Final Answer: 北京今天晴天25°C，上海多云22°C，北京更暖和。

重要规则：
- 步骤要具体、可执行
- 标注工具依赖关系
- 如果执行中发现问题，使用 "需要重新规划" 调整
- 最后必须给出 Final Answer"""

    def __init__(
        self,
        llm: BaseLLM,
        memory: BaseMemory,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        max_steps: int = 20,
        allow_replan: bool = True
    ):
        """
        初始化 Plan-and-Solve 推理引擎

        Args:
            llm: LLM 实例
            memory: 记忆系统
            tool_registry: 工具注册中心
            system_prompt: 系统提示词
            max_steps: 最大执行步骤数
            allow_replan: 是否允许执行中重新规划
        """
        self.llm = llm
        self.memory = memory
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt or ""
        self.max_steps = max_steps
        self.allow_replan = allow_replan
        self.logger = get_logger(self.__class__.__name__)

    def run(
        self,
        user_input: str,
        show_planning: bool = False,
        event_emitter: Optional[Callable] = None
    ) -> str:
        """
        执行 Plan-and-Solve 推理

        Args:
            user_input: 用户输入
            show_planning: 是否返回计划和执行过程
            event_emitter: 可选的事件发射函数

        Returns:
            Agent 响应
        """
        self.logger.info(f"Plan-and-Solve started: {user_input[:100]}...")

        # 初始化轨迹
        trace = PlanSolveTrace()

        # 添加用户消息到记忆
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 触发开始事件
        if event_emitter:
            event_emitter("before_chat", user_input=user_input, mode="plan_solve")

        # 第一步：制定计划
        plan = self._create_plan(user_input, event_emitter)
        if not plan:
            error_msg = "无法制定执行计划"
            self.logger.error(error_msg)
            return error_msg

        trace.set_plan(plan)
        self.logger.info(f"Plan created with {len(plan.steps)} steps")

        if event_emitter:
            event_emitter("plan_created", plan=plan.to_prompt())

        # 第二步：执行计划
        step_count = 0
        while step_count < self.max_steps:
            ready_steps = trace.get_ready_steps()

            if not ready_steps:
                if trace.is_complete():
                    break
                # 没有可执行的步骤，但计划未完成（可能是有失败的依赖）
                self.logger.warning("No ready steps but plan not complete")
                break

            # 执行第一个就绪的步骤
            plan_step = ready_steps[0]
            step_count += 1

            self.logger.info(f"Executing step {plan_step.step_id + 1}: {plan_step.description}")

            if event_emitter:
                event_emitter("step_start", step_id=plan_step.step_id, description=plan_step.description)

            # 执行步骤
            result = self._execute_step(plan_step, user_input, trace, event_emitter)

            if event_emitter:
                event_emitter("step_end", step_id=plan_step.step_id, result=result.status.value)

            # 检查是否需要重新规划
            if result.status == StepStatus.FAILED and self.allow_replan:
                self.logger.info("Step failed, considering replan")
                should_replan = self._should_replan(trace, result.error_message)
                if should_replan:
                    new_plan = self._replan(user_input, trace, event_emitter)
                    if new_plan:
                        trace.add_review(
                            content=f"步骤 {plan_step.step_id + 1} 失败，重新规划",
                            needs_replan=True,
                            new_plan=new_plan
                        )
                        trace.set_plan(new_plan)
                        self.logger.info(f"Replan created with {len(new_plan.steps)} steps")

            # 检查是否已完成（有最终答案）
            if trace.final_answer:
                break

        # 生成最终答案（如果还没有）
        if not trace.final_answer:
            final_answer = self._generate_final_answer(user_input, trace)
            trace.final_answer = final_answer

        # 保存到记忆
        self.memory.add(Message(role=Role.ASSISTANT, content=trace.final_answer))

        self.logger.info(f"Final answer: {trace.final_answer[:100]}...")

        if event_emitter:
            event_emitter("after_chat", response=trace.final_answer, mode="plan_solve")

        if show_planning:
            return trace.to_prompt()
        return trace.final_answer

    def _create_plan(
        self,
        user_input: str,
        event_emitter: Optional[Callable] = None
    ) -> Optional[Plan]:
        """制定执行计划"""
        messages = []

        # 系统提示
        system_content = self.system_prompt + "\n\n" + self.PLAN_INSTRUCTIONS
        messages.append(Message(role=Role.SYSTEM, content=system_content))

        # 工具描述
        if self.tool_registry.count() > 0:
            tools_desc = self._format_tools()
            messages.append(Message(role=Role.SYSTEM, content=f"可用工具：\n{tools_desc}"))

        # 用户问题
        messages.append(Message(role=Role.USER, content=f"请为以下问题制定执行计划：\n\n{user_input}\n\n请输出计划步骤（格式：1. [工具:xxx] 描述 (依赖: y, z)）："))

        # 调用 LLM
        response = self.llm.chat(messages)
        content = response.content or ""

        self.logger.debug(f"Plan response: {content[:500]}...")

        # 解析计划
        return self._parse_plan(content)

    def _parse_plan(self, content: str) -> Optional[Plan]:
        """解析 LLM 返回的计划"""
        plan = Plan()

        # 提取计划思考
        reasoning_match = re.search(r'计划思考[:：]\s*(.+?)(?=\n计划步骤|\n步骤|\Z)', content, re.DOTALL)
        if reasoning_match:
            plan.reasoning = reasoning_match.group(1).strip()

        # 提取计划步骤
        # 匹配格式：
        # 1. [工具:xxx] 描述 (依赖: y, z)
        # 或者：1. 描述 [工具:xxx] (依赖: y, z)
        step_pattern = r'(?:步骤\s*)?(\d+)[:.\s]+(?:\[工具[:：]?(\w*)\]\s*)?(.+?)(?:\s*\(依赖[:：]?\s*([^)]+)\))?$'

        for match in re.finditer(step_pattern, content, re.MULTILINE):
            step_num = int(match.group(1))
            tool = match.group(2)
            description = match.group(3).strip()
            deps_str = match.group(4)

            # 解析依赖
            depends_on = set()
            if deps_str:
                for dep in deps_str.split(','):
                    dep = dep.strip()
                    if dep.isdigit():
                        depends_on.add(int(dep) - 1)  # 转换为0-based索引

            # 如果没有工具，尝试从描述中提取
            if not tool:
                tool_match = re.search(r'\[(?:工具[:：]?)?(\w+)\]', description)
                if tool_match:
                    tool = tool_match.group(1)
                    # 移除工具标记
                    description = re.sub(r'\s*\[(?:工具[:：]?)?\w+\]\s*', ' ', description).strip()

            # 清理描述
            description = re.sub(r'\s*\([^)]+\)\s*$', '', description).strip()

            plan_step = PlanStep(
                step_id=step_num - 1,  # 转换为0-based
                description=description,
                tool=tool if tool else None,
                depends_on=depends_on
            )
            plan.steps.append(plan_step)

        # 如果没有解析到步骤，尝试备用模式
        if not plan.steps:
            # 简单模式：每行一个步骤
            lines = content.split('\n')
            step_id = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 匹配数字开头
                if re.match(r'^\d+[:.\s]', line):
                    description = re.sub(r'^\d+[:.\s]+', '', line)
                    plan_step = PlanStep(
                        step_id=step_id,
                        description=description
                    )
                    plan.steps.append(plan_step)
                    step_id += 1

        return plan if plan.steps else None

    def _execute_step(
        self,
        plan_step: PlanStep,
        user_input: str,
        trace: PlanSolveTrace,
        event_emitter: Optional[Callable] = None
    ) -> ExecutionStep:
        """执行单个步骤"""
        # 如果需要工具，执行工具调用
        if plan_step.tool and plan_step.tool in self.tool_registry.list_tools():
            # 构建工具调用提示
            tool_prompt = self._build_step_prompt(plan_step, user_input, trace)

            messages = [Message(role=Role.USER, content=tool_prompt)]
            response = self.llm.chat(messages)
            content = response.content or ""

            # 解析 Action Input
            action_input = self._parse_action_input(content)

            if event_emitter:
                event_emitter("tool_call", name=plan_step.tool, arguments=action_input)

            # 执行工具
            try:
                result = self.tool_registry.execute(plan_step.tool, **action_input)

                if event_emitter:
                    event_emitter("tool_result", name=plan_step.tool, result=result.content, is_error=result.is_error)

                if result.is_error:
                    return trace.fail_step(plan_step.step_id, result.error_message or "工具执行失败")

                # 记录执行结果
                return trace.execute_step(plan_step.step_id, tool_result=result.content)

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Tool execution error: {error_msg}")
                return trace.fail_step(plan_step.step_id, error_msg)

        else:
            # 无需工具的步骤（如思考、汇总等）
            step_prompt = self._build_step_prompt(plan_step, user_input, trace)
            messages = [Message(role=Role.USER, content=step_prompt)]
            response = self.llm.chat(messages)
            content = response.content or ""

            # 提取结果
            result_match = re.search(r'结果[:：]\s*(.+?)(?=\n\n|\Z)', content, re.DOTALL)
            result = result_match.group(1).strip() if result_match else content.strip()

            return trace.execute_step(plan_step.step_id, tool_result=result)

    def _build_step_prompt(self, plan_step: PlanStep, user_input: str, trace: PlanSolveTrace) -> str:
        """构建步骤执行提示"""
        lines = [
            f"执行计划步骤 {plan_step.step_id + 1}: {plan_step.description}",
            ""
        ]

        if plan_step.tool:
            lines.append(f"工具: {plan_step.tool}")
            lines.append("请提供 Action Input（JSON格式）：")
            lines.append("")

        # 添加上下文（已完成的步骤结果）
        completed = trace.get_completed_step_ids()
        if completed:
            lines.append("已完成的步骤结果：")
            for exec_step in trace.executions:
                if exec_step.plan_step.step_id in completed and exec_step.output:
                    lines.append(f"  步骤 {exec_step.plan_step.step_id + 1}: {exec_step.output}")
            lines.append("")

        lines.append(f"原始问题: {user_input}")
        lines.append("")

        if plan_step.tool:
            lines.append("输出格式：")
            lines.append("Action Input: {...}")
            lines.append("结果: 工具执行后的总结")
        else:
            lines.append("请直接输出结果：")
            lines.append("结果: ...")

        return "\n".join(lines)

    def _parse_action_input(self, content: str) -> Dict:
        """解析 Action Input"""
        input_match = re.search(r'Action Input[:：]\s*(\{[^}]*\})', content)
        if input_match:
            try:
                return json.loads(input_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试解析整个响应为 JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 默认返回空对象
        return {}

    def _should_replan(self, trace: PlanSolveTrace, error_message: str) -> bool:
        """判断是否需要重新规划"""
        # 简单策略：如果失败就尝试重新规划
        # 可以扩展为更复杂的逻辑
        return True

    def _replan(
        self,
        user_input: str,
        trace: PlanSolveTrace,
        event_emitter: Optional[Callable] = None
    ) -> Optional[Plan]:
        """重新制定计划"""
        messages = []

        system_content = self.system_prompt + "\n\n" + self.PLAN_INSTRUCTIONS
        system_content += "\n\n注意：之前的计划执行遇到问题，请调整计划。"
        messages.append(Message(role=Role.SYSTEM, content=system_content))

        # 添加上下文
        context = trace.to_prompt()
        messages.append(Message(role=Role.USER, content=f"执行历史：\n{context}\n\n请重新制定执行计划："))

        response = self.llm.chat(messages)
        content = response.content or ""

        return self._parse_plan(content)

    def _generate_final_answer(self, user_input: str, trace: PlanSolveTrace) -> str:
        """生成最终答案"""
        messages = []

        system_content = "基于执行计划的结果，生成最终答案。"
        messages.append(Message(role=Role.SYSTEM, content=system_content))

        context = trace.to_prompt()
        messages.append(Message(role=Role.USER, content=f"原始问题: {user_input}\n\n执行过程:\n{context}\n\n请给出 Final Answer："))

        response = self.llm.chat(messages)
        content = response.content or ""

        # 提取 Final Answer
        final_match = re.search(r'Final Answer[:：]\s*(.+?)(?=\n\n|\Z)', content, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()

        return content.strip()

    def _format_tools(self) -> str:
        """格式化工具描述"""
        definitions = self.tool_registry.get_all_definitions()
        lines = []
        for d in definitions:
            lines.append(f"- {d.name}: {d.description}")
            if d.parameters:
                lines.append(f"  参数: {d.parameters}")
        return "\n".join(lines)
