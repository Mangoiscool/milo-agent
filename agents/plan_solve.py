"""
PlanSolve Agent - 先规划后执行

Plan-and-Solve = Plan First, Then Solve

核心特性：
- 任务分解与规划（Plan）
- 按计划逐步执行（Solve）
- 执行中的反思与重新规划（Review）
- 执行进度追踪

与 ReActAgent 的区别：
- ReActAgent：边想边做，适合探索性问题
- PlanSolveAgent：先规划后执行，适合结构化任务

使用示例：
    from agents import PlanSolveAgent
    from core.tools import CalculatorTool, WeatherTool

    agent = PlanSolveAgent(
        llm=llm,
        tools=[CalculatorTool(), WeatherTool()]
    )

    # 普通对话
    response = agent.chat("帮我规划一次北京到上海的三日游")

    # 显示计划和执行过程
    response = agent.chat("比较北京和上海天气", show_planning=True)
"""

import json
import re
from typing import Dict, List, Optional, Set

from agents.base import AgentEvent, BaseAgent
from core.llm.base import BaseLLM, LLMResponse, Message, Role
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.memory.short_term import ShortTermMemory
from core.reasoning.plan_solve import (
    ExecutionStep,
    Plan,
    PlanSolveTrace,
    PlanStep,
    Review,
    StepStatus,
)
from core.tools.base import BaseTool
from agents.agent_config import AgentConfig


# ═══════════════════════════════════════════════════════════════
# Plan-and-Solve Prompt 模板
# ═══════════════════════════════════════════════════════════════

PLAN_SOLVE_SYSTEM_PROMPT = """你是一个智能助手，使用 Plan-and-Solve 方法帮助用户解决问题。

核心流程：
1. **Plan（制定计划）**：分析任务，将其分解为清晰的执行步骤
2. **Solve（执行计划）**：按顺序执行每个步骤
3. **Review（审查反思）**：如遇到问题，反思并重新规划
4. **Final Answer**：汇总结果给出最终答案

输出格式：

第一步 - 制定计划：
计划思考: <分析任务，确定分解策略>
计划步骤:
1. [工具:tool_name] 步骤描述 (依赖: 无)
2. [工具:tool_name] 步骤描述 (依赖: 1)
3. [无需工具] 步骤描述 (依赖: 1, 2)
...

第二步 - 执行计划：
对每个需要工具的步骤：
步骤 X [tool_name]:
Action Input: {"param": "value"}
Observation: <工具返回结果>
结果: <执行总结>

对无需工具的步骤：
步骤 X:
结果: <思考或处理结果>

第三步 - 最终答案：
Final Answer: <清晰完整的答案>

重要规则：
- 计划步骤要具体、可执行
- 明确标注工具依赖关系
- 使用可用工具完成实际查询/计算
- 如无必需工具，可直接回答
- 如果执行失败，使用 "需要重新规划" 调整策略

可用工具：
{tools}

示例 1 - 多步计算：
问题: 计算 (100 + 50) * 2 的平方根

计划思考: 这是一个多步计算，需要先加、再乘、最后开方。
计划步骤:
1. [工具:calculator] 计算 100 + 50 (依赖: 无)
2. [工具:calculator] 将结果乘以 2 (依赖: 1)
3. [工具:calculator] 计算平方根 (依赖: 2)

执行:
步骤 1 [calculator]:
Action Input: {"expression": "100 + 50"}
Observation: 150
结果: 100+50=150

步骤 2 [calculator]:
Action Input: {"expression": "150 * 2"}
Observation: 300
结果: 150*2=300

步骤 3 [calculator]:
Action Input: {"expression": "sqrt(300)"}
Observation: 17.32
结果: √300 ≈ 17.32

Final Answer: (100 + 50) * 2 = 300，平方根约为 17.32

示例 2 - 无需工具：
问题: 你好

计划思考: 这是简单问候，无需工具。
计划步骤:
1. [无需工具] 回应问候

执行:
步骤 1:
结果: 准备友好回复

Final Answer: 你好！很高兴为你服务。有什么我可以帮你的吗？"""


class PlanSolveAgent(BaseAgent):
    """
    PlanSolve Agent - 先规划后执行

    特性：
    - 任务分解与整体规划
    - 按计划逐步执行
    - 支持依赖管理和并行步骤
    - 失败时支持重新规划
    - 执行进度可视化

    使用示例：
        agent = PlanSolveAgent(llm, tools=[weather_tool, search_tool])
        response = agent.chat("帮我规划一次北京到上海的三日游")

        # 显示计划和执行过程：
        response = agent.chat("比较两地天气", show_planning=True)
    """

    DEFAULT_SYSTEM_PROMPT = PLAN_SOLVE_SYSTEM_PROMPT

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        max_steps: int = 20,
        allow_replan: bool = True
    ):
        """
        初始化 PlanSolve Agent

        Args:
            llm: LLM 实例
            memory: 记忆系统实例（默认使用 ShortTermMemory）
            tools: 初始工具列表
            system_prompt: 系统提示词
            config: Agent 配置
            max_steps: 最大执行步骤数
            allow_replan: 是否允许执行中重新规划
        """
        super().__init__(
            llm=llm,
            memory=memory,
            tools=tools,
            system_prompt=system_prompt,
            config=config
        )

        # Plan-and-Solve 配置
        self.max_steps = max_steps
        self.allow_replan = allow_replan

        # 执行轨迹
        self.trace: Optional[PlanSolveTrace] = None

        # 缓存工具描述
        self._tools_description: Optional[str] = None

    # ═══════════════════════════════════════════════════════════════
    # 核心对话接口
    # ═══════════════════════════════════════════════════════════════

    def chat(self, user_input: str, show_planning: bool = False) -> str:
        """
        Plan-and-Solve 对话

        Args:
            user_input: 用户输入
            show_planning: 是否返回计划和执行过程

        Returns:
            Agent 响应
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="plan_solve")
        self.logger.info(f"User input (PlanSolve): {user_input[:100]}...")

        # 1. 初始化轨迹
        self.trace = PlanSolveTrace()
        step_count = 0

        # 2. 制定计划
        plan = self._create_plan(user_input)
        if not plan:
            error_msg = "抱歉，无法为这个问题制定计划。"
            self.logger.error(error_msg)
            self._emit(AgentEvent.AFTER_CHAT, response=error_msg, mode="plan_solve")
            return error_msg

        self.trace.set_plan(plan)
        self.logger.info(f"Plan created: {len(plan.steps)} steps")
        self._emit(AgentEvent.PLAN_CREATED, plan=plan.to_prompt())

        # 3. 执行计划
        while step_count < self.max_steps:
            ready_steps = self.trace.get_ready_steps()

            if not ready_steps:
                if self.trace.is_complete():
                    break
                self.logger.warning("No ready steps but plan incomplete")
                break

            plan_step = ready_steps[0]
            step_count += 1

            self.logger.info(f"Executing step {plan_step.step_id + 1}: {plan_step.description}")
            self._emit(AgentEvent.STEP_START, step_id=plan_step.step_id, description=plan_step.description)

            # 执行步骤
            result = self._execute_step(plan_step, user_input)
            self._emit(AgentEvent.STEP_END, step_id=plan_step.step_id, status=result.status.value)

            # 检查失败并尝试重新规划
            if result.status == StepStatus.FAILED and self.allow_replan:
                self.logger.info("Step failed, attempting replan")
                if self._try_replan(user_input, result.error_message or "执行失败"):
                    continue  # 继续执行新计划
                else:
                    self.logger.warning("Replan failed")

        # 4. 生成最终答案
        if not self.trace.final_answer:
            final_answer = self._generate_final_answer(user_input)
            self.trace.final_answer = final_answer

        # 保存到记忆
        self.memory.add(Message(role=Role.USER, content=user_input))
        self.memory.add(Message(role=Role.ASSISTANT, content=self.trace.final_answer))

        self.logger.info(f"Final Answer: {self.trace.final_answer[:100]}...")
        self._emit(AgentEvent.AFTER_CHAT, response=self.trace.final_answer, mode="plan_solve")

        if show_planning:
            return self.trace.to_prompt()
        return self.trace.final_answer

    # ═══════════════════════════════════════════════════════════════
    # 计划制定
    # ═══════════════════════════════════════════════════════════════

    def _create_plan(self, user_input: str) -> Optional[Plan]:
        """制定执行计划"""
        prompt = self._build_plan_prompt(user_input)
        messages = [Message(role=Role.USER, content=prompt)]

        response = self.llm.chat(messages)
        content = response.content or ""

        self.logger.debug(f"Plan response: {content[:500]}...")

        return self._parse_plan(content)

    def _build_plan_prompt(self, user_input: str) -> str:
        """构建计划制定提示"""
        tools_desc = self._format_tools()
        system_prompt = (self.system_prompt or self.DEFAULT_SYSTEM_PROMPT).format(tools=tools_desc)

        return f"""{system_prompt}

请为以下问题制定执行计划：

问题: {user_input}

请输出：
1. 计划思考（分析任务）
2. 计划步骤（编号列表，标注工具和依赖）"""

    def _parse_plan(self, content: str) -> Optional[Plan]:
        """解析 LLM 返回的计划"""
        plan = Plan()

        # 提取计划思考
        reasoning_patterns = [
            r'计划思考[:：]\s*(.+?)(?=\n计划步骤|\n步骤|\Z)',
            r'(?:^|\n)思考[:：]\s*(.+?)(?=\n步骤|\n计划|\Z)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                plan.reasoning = match.group(1).strip()
                break

        # 提取计划步骤 - 支持多种格式
        # 格式1: 1. [工具:xxx] 描述 (依赖: y, z)
        # 格式2: 1. 描述 [工具:xxx] (依赖: y)
        # 格式3: 步骤 1: 描述
        step_patterns = [
            r'(?:^|\n)(?:步骤\s*)?(\d+)[:.\s]+(?:\[\s*工具[:：]?(\w*)\s*\]\s*)?(.+?)(?:\s*\(\s*依赖[:：]?\s*([^)]+)\))?(?=\n|$)',
            r'(?:^|\n)(?:步骤\s*)?(\d+)[:.\s]+(.+?)(?:\s*\[\s*工具[:：]?(\w*)\s*\])?(?:\s*\(\s*依赖[:：]?\s*([^)]+)\))?(?=\n|$)',
        ]

        for pattern in step_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
            if matches:
                for i, match in enumerate(matches):
                    step_num = int(match.group(1))
                    # 根据捕获组位置判断工具
                    groups = match.groups()
                    tool = None
                    description = None
                    deps_str = None

                    if len(groups) >= 4:
                        # 第一个模式：group1=num, group2=tool, group3=desc, group4=deps
                        tool = groups[1] if groups[1] else None
                        description = groups[2].strip() if groups[2] else ""
                        deps_str = groups[3]
                    elif len(groups) >= 3:
                        # 第二个模式：group1=num, group2=desc, group3=tool, group4=deps
                        description = groups[1].strip() if groups[1] else ""
                        tool = groups[2] if groups[2] else None
                        deps_str = groups[3] if len(groups) > 3 else None

                    # 解析依赖
                    depends_on = set()
                    if deps_str:
                        for dep in deps_str.split(','):
                            dep = dep.strip()
                            if dep.isdigit():
                                depends_on.add(int(dep) - 1)

                    # 清理描述中的工具标记
                    description = re.sub(r'\s*\[\s*工具[:：]?\w*\s*\]\s*', ' ', description).strip()
                    description = re.sub(r'\s*\([^)]+\)\s*$', '', description).strip()

                    # 判断是否无需工具
                    if not tool and ('无需工具' in description or '无需' in description):
                        description = re.sub(r'\[?\s*无需工具\s*\]?', '', description).strip()

                    plan_step = PlanStep(
                        step_id=step_num - 1,
                        description=description,
                        tool=tool,
                        depends_on=depends_on
                    )
                    plan.steps.append(plan_step)
                break

        # 如果没有解析到步骤，尝试备用模式
        if not plan.steps:
            lines = content.split('\n')
            step_id = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 匹配数字开头的行
                if re.match(r'^\d+[:.\s]', line):
                    description = re.sub(r'^\d+[:.\s]+', '', line).strip()
                    if description:
                        plan_step = PlanStep(step_id=step_id, description=description)
                        plan.steps.append(plan_step)
                        step_id += 1

        # 排序步骤
        plan.steps.sort(key=lambda s: s.step_id)

        return plan if plan.steps else None

    # ═══════════════════════════════════════════════════════════════
    # 步骤执行
    # ═══════════════════════════════════════════════════════════════

    def _execute_step(self, plan_step: PlanStep, user_input: str) -> ExecutionStep:
        """执行单个步骤"""
        if plan_step.tool and self.tool_registry.has(plan_step.tool):
            # 需要工具的步骤
            tool_prompt = self._build_step_execution_prompt(plan_step, user_input)
            messages = [Message(role=Role.USER, content=tool_prompt)]

            response = self.llm.chat(messages)
            content = response.content or ""

            # 解析 Action Input
            action_input = self._parse_action_input(content)
            self._emit(AgentEvent.TOOL_CALL, name=plan_step.tool, arguments=action_input)

            # 执行工具
            try:
                result = self.tool_registry.execute(plan_step.tool, **action_input)
                self._emit(AgentEvent.TOOL_RESULT, name=plan_step.tool, result=result.content, is_error=result.is_error)

                if result.is_error:
                    return self.trace.fail_step(plan_step.step_id, result.error_message or "工具执行失败")

                return self.trace.execute_step(plan_step.step_id, tool_result=result.content)

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Tool error: {error_msg}")
                return self.trace.fail_step(plan_step.step_id, error_msg)

        else:
            # 无需工具的步骤
            step_prompt = self._build_step_execution_prompt(plan_step, user_input)
            messages = [Message(role=Role.USER, content=step_prompt)]

            response = self.llm.chat(messages)
            content = response.content or ""

            # 提取结果
            result_match = re.search(r'(?:结果|Result)[:：]\s*(.+?)(?=\n\n|$)', content, re.DOTALL | re.IGNORECASE)
            result = result_match.group(1).strip() if result_match else content.strip()

            return self.trace.execute_step(plan_step.step_id, tool_result=result)

    def _build_step_execution_prompt(self, plan_step: PlanStep, user_input: str) -> str:
        """构建步骤执行提示"""
        lines = [
            f"执行步骤 {plan_step.step_id + 1}: {plan_step.description}",
            ""
        ]

        if plan_step.tool:
            lines.append(f"使用工具: {plan_step.tool}")
            lines.append("请提供 Action Input（JSON格式）")
            lines.append("")

        # 添加上下文
        completed = self.trace.get_completed_step_ids()
        if completed:
            lines.append("已完成的步骤结果：")
            for exec_step in self.trace.executions:
                if exec_step.plan_step.step_id in completed and exec_step.output:
                    lines.append(f"  {exec_step.plan_step.step_id + 1}. {exec_step.output}")
            lines.append("")

        lines.append(f"原始问题: {user_input}")

        if plan_step.tool:
            lines.append("")
            lines.append("格式：")
            lines.append("Action Input: {...}")
            lines.append("Observation: <工具返回结果>")
            lines.append("结果: <总结>")
        else:
            lines.append("")
            lines.append("请直接输出结果：")
            lines.append("结果: ...")

        return "\n".join(lines)

    def _parse_action_input(self, content: str) -> Dict:
        """解析 Action Input"""
        # 尝试提取 JSON
        json_patterns = [
            r'Action Input[:：]\s*(\{[^}]*\})',
            r'```json\s*(\{[^}]*\})',
        ]
        for pattern in json_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # 尝试解析整个内容为 JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        return {}

    # ═══════════════════════════════════════════════════════════════
    # 重新规划
    # ═══════════════════════════════════════════════════════════════

    def _try_replan(self, user_input: str, error_message: str) -> bool:
        """尝试重新规划"""
        self.logger.info("Replanning...")

        context = self.trace.to_prompt()
        prompt = f"""之前的计划执行遇到问题：

{context}

错误信息: {error_message}

请重新制定执行计划，考虑：
1. 是否任务分解不合理？
2. 是否需要不同的工具或方法？
3. 是否步骤依赖关系有问题？

请输出新的计划："""

        messages = [Message(role=Role.USER, content=prompt)]
        response = self.llm.chat(messages)
        content = response.content or ""

        new_plan = self._parse_plan(content)
        if new_plan and len(new_plan.steps) > 0:
            self.trace.add_review(
                content=f"执行失败，重新规划: {error_message}",
                needs_replan=True,
                new_plan=new_plan
            )
            self.logger.info(f"Replan successful: {len(new_plan.steps)} steps")
            return True

        self.logger.warning("Replan failed")
        return False

    # ═══════════════════════════════════════════════════════════════
    # 最终答案
    # ═══════════════════════════════════════════════════════════════

    def _generate_final_answer(self, user_input: str) -> str:
        """生成最终答案"""
        context = self.trace.to_prompt()
        prompt = f"""基于执行计划的结果，生成最终答案。

原始问题: {user_input}

执行过程:
{context}

请给出 Final Answer："""

        messages = [Message(role=Role.USER, content=prompt)]
        response = self.llm.chat(messages)
        content = response.content or ""

        # 提取 Final Answer
        final_match = re.search(r'(?:Final Answer|最终答案)[:：]\s*(.+?)(?=\n\n|$)', content, re.DOTALL | re.IGNORECASE)
        if final_match:
            return final_match.group(1).strip()

        return content.strip()

    # ═══════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════

    def _format_tools(self) -> str:
        """格式化工具描述"""
        if self._tools_description:
            return self._tools_description

        definitions = self.tool_registry.get_all_definitions()
        if not definitions:
            return "（无可用工具）"

        lines = []
        for d in definitions:
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

    def register_tool(self, tool: BaseTool) -> None:
        """注册工具（重写以清除缓存）"""
        super().register_tool(tool)
        self._tools_description = None

    def unregister_tool(self, name: str) -> bool:
        """注销工具（重写以清除缓存）"""
        result = super().unregister_tool(name)
        self._tools_description = None
        return result

    # ═══════════════════════════════════════════════════════════════
    # 辅助接口
    # ═══════════════════════════════════════════════════════════════

    def get_trace(self) -> Optional[PlanSolveTrace]:
        """获取当前执行轨迹"""
        return self.trace

    def get_progress(self) -> tuple:
        """获取执行进度 (已完成, 总数, 百分比)"""
        if self.trace:
            return self.trace.get_progress()
        return (0, 0, 0.0)

    def __repr__(self) -> str:
        return f"<PlanSolveAgent llm={self.llm} tools={self.tool_registry.count()}>"
