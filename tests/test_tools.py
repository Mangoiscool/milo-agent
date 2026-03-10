"""
测试工具系统核心功能
"""

import os
import tempfile
from pathlib import Path

from core.tools import ToolRegistry
from core.tools.builtin import CalculatorTool, WeatherTool


def test_tool_registry():
    """测试工具注册中心"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    
    return registry


def test_tool_properties():
    """测试工具属性"""
    tool = registry.get("calculator")
    
    assert tool.name == "calculator"
    assert "计算" in tool.description


    assert "expression" in tool.parameters["properties"]
    assert "required" in tool.parameters["required"]


if __name__ == "__main__":
    test_tool_registry()
