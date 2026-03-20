"""
ReAct 数据结构单元测试

可以独立运行，不依赖完整的 agent 导入链
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.reasoning.react import ThoughtStep, ActionStep, ObservationStep, ReActTrace


def test_thought_step():
    """测试 ThoughtStep"""
    print("\n=== Testing ThoughtStep ===")
    
    # 创建 thought
    thought = ThoughtStep(content="我需要计算天气和温度")
    
    # 验证属性
    assert thought.content == "我需要计算天气和温度"
    assert thought.timestamp > 0
    
    # 验证 __repr__
    repr_str = repr(thought)
    assert "Thought:" in repr_str
    
    # 验证 to_prompt
    prompt = thought.to_prompt()
    assert prompt == "Thought: 我需要计算天气和温度"
    
    print("✓ ThoughtStep tests passed")


def test_action_step():
    """测试 ActionStep"""
    print("\n=== Testing ActionStep ===")
    
    # 创建 action
    action = ActionStep(
        tool_name="weather",
        arguments={"city": "北京"}
    )
    
    # 验证属性
    assert action.tool_name == "weather"
    assert action.arguments == {"city": "北京"}
    assert action.timestamp > 0
    
    # 验证 __repr__
    repr_str = repr(action)
    assert "Action:" in repr_str
    assert "weather" in repr_str
    
    # 验证 to_prompt
    prompt = action.to_prompt()
    assert "Action: weather" in prompt
    assert "Action Input:" in prompt
    assert '"city": "北京"' in prompt
    
    print("✓ ActionStep tests passed")


def test_observation_step():
    """测试 ObservationStep"""
    print("\n=== Testing ObservationStep ===")
    
    # 创建 observation (成功)
    obs = ObservationStep(
        result="晴天，25°C",
        is_error=False
    )
    
    # 验证属性
    assert obs.result == "晴天，25°C"
    assert obs.is_error == False
    
    # 验证 __repr__
    repr_str = repr(obs)
    assert "Observation" in repr_str
    assert "OK" in repr_str
    
    # 验证 to_prompt
    prompt = obs.to_prompt()
    assert prompt == "Observation: 晴天，25°C"
    
    # 测试错误情况
    obs_error = ObservationStep(
        result="工具不存在",
        is_error=True
    )
    
    repr_str = repr(obs_error)
    assert "Error" in repr_str
    
    prompt = obs_error.to_prompt()
    assert "Error:" in prompt
    
    print("✓ ObservationStep tests passed")


def test_react_trace():
    """测试 ReActTrace"""
    print("\n=== Testing ReActTrace ===")
    
    # 创建 trace
    trace = ReActTrace()
    
    # 验证初始状态
    assert len(trace) == 0
    assert trace.count_iterations() == 0
    
    # 添加步骤
    thought = trace.add_thought("我需要查询天气")
    assert isinstance(thought, ThoughtStep)
    assert len(trace) == 1
    
    action = trace.add_action("weather", {"city": "北京"})
    assert isinstance(action, ActionStep)
    assert len(trace) == 2
    
    observation = trace.add_observation("晴天，25°C", is_error=False)
    assert isinstance(observation, ObservationStep)
    assert len(trace) == 3
    
    # 验证迭代次数
    assert trace.count_iterations() == 1
    
    # 验证获取方法
    thoughts = trace.get_thoughts()
    assert len(thoughts) == 1
    
    actions = trace.get_actions()
    assert len(actions) == 1
    
    observations = trace.get_observations()
    assert len(observations) == 1
    
    # 验证最后一个步骤
    last_action = trace.get_last_action()
    assert last_action == action
    
    last_obs = trace.get_last_observation()
    assert last_obs == observation
    
    print("✓ ReActTrace tests passed")


def test_react_trace_to_prompt():
    """测试 ReActTrace.to_prompt()"""
    print("\n=== Testing ReActTrace.to_prompt() ===")
    
    trace = ReActTrace()
    
    # 添加一个完整的 Thought-Action-Observation 循环
    trace.add_thought("用户询问北京天气，我需要查询")
    trace.add_action("weather", {"city": "北京"})
    trace.add_observation('{"temperature": 25, "condition": "晴天"}')
    
    # 添加另一个循环
    trace.add_thought("已经获取天气信息，可以回答了")
    
    # 获取 prompt
    prompt = trace.to_prompt()
    
    # 验证格式
    lines = prompt.split("\n")
    assert "Thought: 用户询问北京天气，我需要查询" in lines
    assert "Action: weather" in lines
    assert 'Action Input: {"city": "北京"}' in lines
    assert 'Observation: {"temperature": 25, "condition": "晴天"}' in lines
    assert "Thought: 已经获取天气信息，可以回答了" in lines
    
    print("Generated prompt:")
    print(prompt)
    print()
    
    print("✓ ReActTrace.to_prompt() tests passed")


def test_multiple_iterations():
    """测试多轮迭代"""
    print("\n=== Testing Multiple Iterations ===")
    
    trace = ReActTrace()
    
    # 第一轮：查询天气
    trace.add_thought("需要查询北京天气")
    trace.add_action("weather", {"city": "北京"})
    trace.add_observation("晴天，25°C")
    
    # 第二轮：计算降温后的温度
    trace.add_thought("今天25度，需要计算降温5度后的温度")
    trace.add_action("calculator", {"expression": "25 - 5"})
    trace.add_observation("20")
    
    # 第三轮：准备回答
    trace.add_thought("已经完成计算，可以回答用户了")
    
    # 验证迭代次数
    assert trace.count_iterations() == 2
    
    # 验证步骤数量
    assert len(trace.get_thoughts()) == 3
    assert len(trace.get_actions()) == 2
    assert len(trace.get_observations()) == 2
    
    # 验证完整的 prompt
    prompt = trace.to_prompt()
    assert prompt.count("Thought:") == 3
    assert prompt.count("Action:") == 2
    assert prompt.count("Observation:") == 2
    
    print("Generated multi-iteration prompt:")
    print(prompt)
    print()
    
    print("✓ Multiple iterations tests passed")


def test_trace_clear():
    """测试清空轨迹"""
    print("\n=== Testing Trace Clear ===")
    
    trace = ReActTrace()
    trace.add_thought("test")
    trace.add_action("test_tool", {})
    
    assert len(trace) == 2
    
    trace.clear()
    
    assert len(trace) == 0
    assert trace.count_iterations() == 0
    
    print("✓ Trace clear tests passed")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("ReAct 数据结构单元测试")
    print("="*60)
    
    try:
        test_thought_step()
        test_action_step()
        test_observation_step()
        test_react_trace()
        test_react_trace_to_prompt()
        test_multiple_iterations()
        test_trace_clear()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过!")
        print("="*60)
        return True
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)