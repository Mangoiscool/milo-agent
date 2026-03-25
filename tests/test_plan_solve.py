"""
测试 Plan-and-Solve 推理模块
"""

import pytest
from core.reasoning.plan_solve import (
    ExecutionStep,
    Plan,
    PlanSolveTrace,
    PlanStep,
    Review,
    StepStatus,
)


class TestPlanStep:
    """测试计划步骤"""

    def test_basic_creation(self):
        step = PlanStep(description="测试步骤", step_id=0)
        assert step.description == "测试步骤"
        assert step.step_id == 0
        assert step.tool is None
        assert step.depends_on == set()

    def test_with_tool(self):
        step = PlanStep(
            description="查询天气",
            step_id=1,
            tool="weather",
            tool_input={"city": "北京"}
        )
        assert step.tool == "weather"
        assert step.tool_input == {"city": "北京"}

    def test_with_dependencies(self):
        step = PlanStep(
            description="比较天气",
            step_id=2,
            depends_on={0, 1}
        )
        assert step.depends_on == {0, 1}


class TestPlan:
    """测试执行计划"""

    def test_empty_plan(self):
        plan = Plan()
        assert len(plan) == 0
        assert plan.is_complete(set()) is True

    def test_add_steps(self):
        plan = Plan()
        step1 = plan.add_step("步骤1", tool="tool1")
        step2 = plan.add_step("步骤2", depends_on={0})

        assert len(plan) == 2
        assert step1.step_id == 0
        assert step2.step_id == 1
        assert step2.depends_on == {0}

    def test_get_ready_steps(self):
        plan = Plan()
        plan.add_step("步骤1")
        plan.add_step("步骤2", depends_on={0})
        plan.add_step("步骤3", depends_on={0})

        # 初始只有步骤1可以执行
        ready = plan.get_ready_steps(set())
        assert len(ready) == 1
        assert ready[0].step_id == 0

        # 步骤1完成后，步骤2和3可以执行
        ready = plan.get_ready_steps({0})
        assert len(ready) == 2
        assert {s.step_id for s in ready} == {1, 2}

    def test_is_complete(self):
        plan = Plan()
        plan.add_step("步骤1")
        plan.add_step("步骤2")

        assert plan.is_complete({0, 1}) is True
        assert plan.is_complete({0}) is False


class TestExecutionStep:
    """测试执行步骤"""

    def test_status_transitions(self):
        plan_step = PlanStep(description="测试", step_id=0)
        exec_step = ExecutionStep(plan_step=plan_step)

        assert exec_step.status == StepStatus.PENDING

        exec_step.mark_running()
        assert exec_step.status == StepStatus.RUNNING
        assert exec_step.start_time is not None

        exec_step.mark_completed("结果")
        assert exec_step.status == StepStatus.COMPLETED
        assert exec_step.output == "结果"
        assert exec_step.end_time is not None

    def test_failed_step(self):
        plan_step = PlanStep(description="测试", step_id=0)
        exec_step = ExecutionStep(plan_step=plan_step)

        exec_step.mark_failed("出错了")
        assert exec_step.status == StepStatus.FAILED
        assert exec_step.error_message == "出错了"


class TestPlanSolveTrace:
    """测试 Plan-and-Solve 执行轨迹"""

    def test_empty_trace(self):
        trace = PlanSolveTrace()
        assert trace.plan is None
        assert len(trace.executions) == 0
        assert trace.is_complete() is False

    def test_set_plan(self):
        trace = PlanSolveTrace()
        plan = Plan()
        plan.add_step("步骤1")
        plan.add_step("步骤2")

        trace.set_plan(plan)
        assert trace.plan == plan

    def test_execute_step(self):
        trace = PlanSolveTrace()
        plan = Plan()
        plan.add_step("步骤1")
        plan.add_step("步骤2")
        trace.set_plan(plan)

        # 执行步骤0
        result = trace.execute_step(0, tool_result="结果1")
        assert result.status == StepStatus.COMPLETED
        assert result.output == "结果1"

        # 执行步骤1
        result = trace.execute_step(1, final_answer="最终答案")
        assert trace.final_answer == "最终答案"

    def test_get_ready_steps(self):
        trace = PlanSolveTrace()
        plan = Plan()
        plan.add_step("步骤1")
        plan.add_step("步骤2", depends_on={0})
        trace.set_plan(plan)

        # 初始只有步骤1可以执行
        ready = trace.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].step_id == 0

        # 执行步骤1后
        trace.execute_step(0, tool_result="结果1")
        ready = trace.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].step_id == 1

    def test_get_progress(self):
        trace = PlanSolveTrace()
        plan = Plan()
        plan.add_step("步骤1")
        plan.add_step("步骤2")
        trace.set_plan(plan)

        completed, total, pct = trace.get_progress()
        assert completed == 0
        assert total == 2
        assert pct == 0.0

        trace.execute_step(0, tool_result="结果")
        completed, total, pct = trace.get_progress()
        assert completed == 1
        assert pct == 50.0

    def test_add_review(self):
        trace = PlanSolveTrace()
        plan = Plan()
        plan.add_step("步骤1")
        trace.set_plan(plan)

        review = trace.add_review("需要重新考虑", step_id=0, needs_replan=False)
        assert len(trace.reviews) == 1
        assert review.content == "需要重新考虑"
        assert review.step_id == 0

    def test_to_prompt(self):
        trace = PlanSolveTrace()
        plan = Plan()
        plan.add_step("查询北京天气")
        trace.set_plan(plan)
        trace.execute_step(0, tool_result="晴天25度")
        trace.final_answer = "北京今天晴天"

        prompt = trace.to_prompt()
        assert "执行计划" in prompt
        assert "执行记录" in prompt
        assert "最终答案" in prompt
        assert "晴天25度" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
