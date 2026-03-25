"""
Plan-and-Solve 推理范式

核心思想：
1. Plan（计划）：首先将复杂任务分解为可执行的子任务序列
2. Solve（求解）：然后按顺序执行每个子任务
3. Review（审查）：可选地反思执行结果并进行调整

与 ReAct 的区别：
- ReAct：边想边做，逐步探索
- Plan-and-Solve：先整体规划，再逐步执行

使用示例：
    trace = PlanSolveTrace()

    # 制定计划
    plan = Plan(
        steps=[
            PlanStep(description="搜索北京天气", tool="weather"),
            PlanStep(description="搜索上海天气", tool="weather"),
            PlanStep(description="比较两地天气", depends_on=[0, 1])
        ]
    )
    trace.set_plan(plan)

    # 执行步骤
    result = trace.execute_step(0, tool_result="晴天，25°C")
    result = trace.execute_step(1, tool_result="多云，22°C")
    result = trace.execute_step(2, final_answer="北京晴天25°C，上海多云22°C")

    # 转换为 prompt
    prompt = trace.to_prompt()
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"      # 待执行
    RUNNING = "running"      # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    SKIPPED = "skipped"      # 已跳过


@dataclass
class PlanStep:
    """
    计划步骤

    表示任务分解后的单个步骤
    """
    description: str                    # 步骤描述
    step_id: int = field(default=0)     # 步骤编号
    tool: Optional[str] = None          # 可选：指定工具
    tool_input: Optional[Dict] = None   # 可选：工具参数模板
    depends_on: Optional[Set[int]] = None  # 依赖的步骤ID
    expected_output: Optional[str] = None  # 期望的输出描述

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = set()

    def __repr__(self) -> str:
        deps = f" [depends:{self.depends_on}]" if self.depends_on else ""
        tool = f" [tool:{self.tool}]" if self.tool else ""
        return f"Step {self.step_id}: {self.description}{tool}{deps}"

    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        deps_str = f" (依赖步骤: {', '.join(map(str, self.depends_on))})" if self.depends_on else ""
        tool_str = f" [工具: {self.tool}]" if self.tool else ""
        return f"{self.step_id + 1}. {self.description}{tool_str}{deps_str}"


@dataclass
class ExecutionStep:
    """
    执行步骤

    记录计划步骤的实际执行结果
    """
    plan_step: PlanStep                 # 关联的计划步骤
    status: StepStatus = StepStatus.PENDING  # 执行状态
    input_data: Dict[str, Any] = field(default_factory=dict)  # 实际输入
    output: Optional[str] = None        # 执行输出
    error_message: Optional[str] = None # 错误信息
    start_time: Optional[float] = None  # 开始时间
    end_time: Optional[float] = None    # 结束时间
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.start_time is None and self.status == StepStatus.RUNNING:
            self.start_time = time.time()

    def mark_running(self) -> None:
        """标记为执行中"""
        self.status = StepStatus.RUNNING
        self.start_time = time.time()

    def mark_completed(self, output: str) -> None:
        """标记为已完成"""
        self.status = StepStatus.COMPLETED
        self.output = output
        self.end_time = time.time()

    def mark_failed(self, error: str) -> None:
        """标记为失败"""
        self.status = StepStatus.FAILED
        self.error_message = error
        self.end_time = time.time()

    def mark_skipped(self, reason: str = "") -> None:
        """标记为跳过"""
        self.status = StepStatus.SKIPPED
        self.output = reason
        self.end_time = time.time()

    @property
    def duration(self) -> Optional[float]:
        """执行时长"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def __repr__(self) -> str:
        status_icon = {
            StepStatus.PENDING: "⏳",
            StepStatus.RUNNING: "▶️",
            StepStatus.COMPLETED: "✅",
            StepStatus.FAILED: "❌",
            StepStatus.SKIPPED: "⏭️"
        }.get(self.status, "?")
        return f"{status_icon} Step {self.plan_step.step_id}: {self.status.value}"

    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        lines = [f"步骤 {self.plan_step.step_id + 1}: {self.plan_step.description}"]
        lines.append(f"  状态: {self.status.value}")

        if self.status == StepStatus.COMPLETED:
            lines.append(f"  结果: {self.output}")
        elif self.status == StepStatus.FAILED:
            lines.append(f"  错误: {self.error_message}")
        elif self.status == StepStatus.SKIPPED:
            lines.append(f"  跳过原因: {self.output}")

        if self.duration is not None:
            lines.append(f"  耗时: {self.duration:.2f}s")

        return "\n".join(lines)


@dataclass
class Plan:
    """
    执行计划

    包含任务分解后的完整步骤列表
    """
    steps: List[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    reasoning: Optional[str] = None     # 制定计划时的思考过程

    def __post_init__(self):
        # 自动设置步骤编号
        for i, step in enumerate(self.steps):
            step.step_id = i

    def add_step(self, description: str, tool: Optional[str] = None,
                 tool_input: Optional[Dict] = None,
                 depends_on: Optional[Set[int]] = None) -> PlanStep:
        """添加步骤"""
        step_id = len(self.steps)
        step = PlanStep(
            step_id=step_id,
            description=description,
            tool=tool,
            tool_input=tool_input,
            depends_on=depends_on or set()
        )
        self.steps.append(step)
        return step

    def get_ready_steps(self, completed_ids: Set[int]) -> List[PlanStep]:
        """
        获取当前可以执行的步骤

        依赖已完成的步骤才返回
        """
        ready = []
        for step in self.steps:
            if step.step_id in completed_ids:
                continue
            if step.depends_on.issubset(completed_ids):
                ready.append(step)
        return ready

    def is_complete(self, completed_ids: Set[int]) -> bool:
        """检查是否所有步骤都已完成"""
        return len(completed_ids) >= len(self.steps)

    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        lines = ["执行计划:"]
        if self.reasoning:
            lines.append(f"\n制定计划的思考:\n{self.reasoning}\n")
        for step in self.steps:
            lines.append(step.to_prompt())
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"<Plan steps={len(self.steps)}>"


@dataclass
class Review:
    """
    审查/反思

    执行过程中的反思和调整
    """
    content: str                        # 反思内容
    step_id: Optional[int] = None       # 关联的步骤ID
    needs_replan: bool = False          # 是否需要重新规划
    new_plan: Optional[Plan] = None     # 新的计划（如果需要）
    timestamp: float = field(default_factory=time.time)

    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        lines = ["审查反思:"]
        if self.step_id is not None:
            lines.append(f"（针对步骤 {self.step_id + 1}）")
        lines.append(self.content)
        if self.needs_replan:
            lines.append("【需要重新规划】")
        return "\n".join(lines)


@dataclass
class PlanSolveTrace:
    """
    Plan-and-Solve 执行轨迹

    记录完整的计划制定、执行和反思过程
    """
    plan: Optional[Plan] = None
    executions: List[ExecutionStep] = field(default_factory=list)
    reviews: List[Review] = field(default_factory=list)
    final_answer: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def set_plan(self, plan: Plan, reasoning: Optional[str] = None) -> None:
        """设置执行计划"""
        self.plan = plan
        if reasoning:
            plan.reasoning = reasoning

    def get_execution(self, step_id: int) -> Optional[ExecutionStep]:
        """获取指定步骤的执行记录"""
        for exec_step in self.executions:
            if exec_step.plan_step.step_id == step_id:
                return exec_step
        return None

    def execute_step(self, step_id: int, tool_result: Optional[str] = None,
                     final_answer: Optional[str] = None) -> ExecutionStep:
        """
        执行步骤

        Args:
            step_id: 步骤ID
            tool_result: 工具执行结果
            final_answer: 如果是最终答案

        Returns:
            ExecutionStep
        """
        if not self.plan:
            raise ValueError("Plan not set")

        plan_step = None
        for step in self.plan.steps:
            if step.step_id == step_id:
                plan_step = step
                break

        if not plan_step:
            raise ValueError(f"Step {step_id} not found in plan")

        exec_step = ExecutionStep(plan_step=plan_step)
        exec_step.mark_running()

        if final_answer:
            exec_step.mark_completed(final_answer)
            self.final_answer = final_answer
            self.completed_at = time.time()
        elif tool_result is not None:
            exec_step.mark_completed(tool_result)
        else:
            # 无需工具执行的步骤
            exec_step.mark_completed("完成")

        self.executions.append(exec_step)
        return exec_step

    def fail_step(self, step_id: int, error: str) -> ExecutionStep:
        """标记步骤失败"""
        if not self.plan:
            raise ValueError("Plan not set")

        plan_step = None
        for step in self.plan.steps:
            if step.step_id == step_id:
                plan_step = step
                break

        if not plan_step:
            raise ValueError(f"Step {step_id} not found in plan")

        exec_step = ExecutionStep(plan_step=plan_step)
        exec_step.mark_failed(error)
        self.executions.append(exec_step)
        return exec_step

    def add_review(self, content: str, step_id: Optional[int] = None,
                   needs_replan: bool = False, new_plan: Optional[Plan] = None) -> Review:
        """添加审查/反思"""
        review = Review(
            content=content,
            step_id=step_id,
            needs_replan=needs_replan,
            new_plan=new_plan
        )
        self.reviews.append(review)

        # 如果需要重新规划，更新当前计划
        if needs_replan and new_plan:
            self.plan = new_plan

        return review

    def get_completed_step_ids(self) -> Set[int]:
        """获取已完成的步骤ID"""
        return {
            exec_step.plan_step.step_id
            for exec_step in self.executions
            if exec_step.status == StepStatus.COMPLETED
        }

    def get_ready_steps(self) -> List[PlanStep]:
        """获取当前可以执行的步骤"""
        if not self.plan:
            return []
        completed = self.get_completed_step_ids()
        return self.plan.get_ready_steps(completed)

    def is_complete(self) -> bool:
        """检查是否执行完成"""
        if not self.plan:
            return False
        completed = self.get_completed_step_ids()
        return self.plan.is_complete(completed)

    def get_progress(self) -> tuple:
        """
        获取执行进度

        Returns:
            (已完成数, 总数, 百分比)
        """
        if not self.plan:
            return (0, 0, 0.0)
        completed = len(self.get_completed_step_ids())
        total = len(self.plan.steps)
        percentage = (completed / total * 100) if total > 0 else 0.0
        return (completed, total, percentage)

    def to_prompt(self) -> str:
        """
        转换为 LLM 可理解的格式
        """
        lines = []

        # 计划
        if self.plan:
            lines.append(self.plan.to_prompt())
            lines.append("")

        # 执行记录
        if self.executions:
            lines.append("执行记录:")
            for exec_step in self.executions:
                lines.append(exec_step.to_prompt())
            lines.append("")

        # 审查记录
        for review in self.reviews:
            lines.append(review.to_prompt())
            lines.append("")

        # 最终答案
        if self.final_answer:
            lines.append(f"最终答案: {self.final_answer}")

        return "\n".join(lines)

    def clear(self) -> None:
        """清空轨迹"""
        self.plan = None
        self.executions.clear()
        self.reviews.clear()
        self.final_answer = None
        self.completed_at = None

    def __len__(self) -> int:
        return len(self.executions)

    def __repr__(self) -> str:
        completed, total, pct = self.get_progress()
        return f"<PlanSolveTrace steps={completed}/{total}({pct:.0f}%)>"
