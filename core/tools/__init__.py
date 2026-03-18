"""
工具系统

提供工具注册、管理和执行功能
支持本地工具和 MCP 协议工具
"""

from .base import BaseTool, ToolResult
from .registry import ToolRegistry
from .mcp import MCPClient, HTTPMCPClient, MCPTool, create_mcp_tools_from_server
from .builtin import (
    CalculatorTool,
    DateTimeTool,
    RandomTool,
    WeatherTool,
    WebSearchTool,
    FileReadTool,
    FileWriteTool,
    ListDirTool,
    CodeExecutionTool,
    ShellCommandTool,
)

__all__ = [
    # 基础类
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    # MCP 支持
    "MCPClient",
    "HTTPMCPClient",
    "MCPTool",
    "create_mcp_tools_from_server",
    # 内置工具
    "CalculatorTool",
    "DateTimeTool",
    "RandomTool",
    "WeatherTool",
    "WebSearchTool",
    "FileReadTool",
    "FileWriteTool",
    "ListDirTool",
    "CodeExecutionTool",
    "ShellCommandTool",
]
