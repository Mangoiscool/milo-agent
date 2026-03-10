"""
MCP (Model Context Protocol) 工具支持

学习重点：
- MCP 协议：统一的工具调用标准
- 工具适配器模式：将 MCP 工具包装为 BaseTool
- 自动发现：从 MCP 服务器获取工具列表

MCP 协议简介：
- 由 Anthropic 提出的开放协议
- 支持工具调用、资源访问、提示词模板
- 传输层：stdio 或 HTTP/SSE
"""

import json
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

import httpx

from .base import BaseTool, ToolResult
from core.logger import get_logger


# ═══════════════════════════════════════════════════════════════
# MCP 客户端抽象
# ═══════════════════════════════════════════════════════════════

class MCPClient(ABC):
    """MCP 客户端抽象基类"""
    
    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """获取服务器提供的工具列表"""
        pass
    
    @abstractmethod
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具"""
        pass


class HTTPMCPClient(MCPClient):
    """
    HTTP/SSE 方式的 MCP 客户端
    
    适用于独立部署的 MCP 服务器
    """
    
    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.logger = get_logger(self.__class__.__name__)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """获取工具列表"""
        url = f"{self.server_url}/tools/list"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                return data.get("tools", [])
        except Exception as e:
            self.logger.error(f"Failed to list MCP tools: {e}")
            return []
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具"""
        url = f"{self.server_url}/tools/call"
        payload = {
            "name": name,
            "arguments": arguments
        }
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # MCP 响应格式: {"content": [...], "isError": bool}
            if data.get("isError"):
                raise RuntimeError(f"MCP tool error: {data.get('content', [])}")
            
            # 提取内容
            content = data.get("content", [])
            if isinstance(content, list) and content:
                # 返回第一个文本内容
                return content[0].get("text", str(content))
            return content


# ═══════════════════════════════════════════════════════════════
# MCP 工具包装类
# ═══════════════════════════════════════════════════════════════

class MCPTool(BaseTool):
    """
    MCP 工具适配器
    
    将 MCP 服务器的工具包装为 milo-agent 的 BaseTool
    使 Agent 可以透明地调用 MCP 工具
    
    使用方法：
        client = HTTPMCPClient("http://localhost:3000/mcp")
        tool_info = client.list_tools()[0]
        mcp_tool = MCPTool(client, tool_info)
        registry.register(mcp_tool)
    """
    
    def __init__(self, client: MCPClient, tool_info: Dict[str, Any]):
        """
        初始化 MCP 工具包装
        
        Args:
            client: MCP 客户端
            tool_info: 工具信息（从 list_tools 获取）
                {
                    "name": "tool_name",
                    "description": "...",
                    "inputSchema": {...}  # JSON Schema
                }
        """
        super().__init__()
        self._client = client
        self._info = tool_info
    
    @property
    def name(self) -> str:
        return self._info.get("name", "unknown")
    
    @property
    def description(self) -> str:
        return self._info.get("description", "")
    
    @property
    def parameters(self) -> Dict[str, Any]:
        # MCP 使用 inputSchema，与我们的 parameters 格式一致
        return self._info.get("inputSchema", {
            "type": "object",
            "properties": {}
        })
    
    def execute(self, **kwargs) -> ToolResult:
        """执行 MCP 工具调用"""
        try:
            self.logger.info(f"Calling MCP tool: {self.name} with args: {kwargs}")
            result = self._client.call_tool(self.name, kwargs)
            
            # 确保结果是字符串
            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, ensure_ascii=False, indent=2)
            
            return ToolResult(content=content)
            
        except Exception as e:
            self.logger.error(f"MCP tool {self.name} failed: {e}")
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"MCP tool call failed: {str(e)}"
            )
    
    def __repr__(self) -> str:
        return f"<MCPTool {self.name}>"


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def create_mcp_tools_from_server(server_url: str) -> List[MCPTool]:
    """
    从 MCP 服务器创建所有工具
    
    Args:
        server_url: MCP 服务器地址
    
    Returns:
        MCPTool 实例列表
    
    使用方法：
        tools = create_mcp_tools_from_server("http://localhost:3000/mcp")
        for tool in tools:
            registry.register(tool)
    """
    client = HTTPMCPClient(server_url)
    tool_infos = client.list_tools()
    
    return [MCPTool(client, info) for info in tool_infos]


__all__ = [
    "MCPClient",
    "HTTPMCPClient",
    "MCPTool",
    "create_mcp_tools_from_server",
]
