"""
工具注册中心

学习重点：
- 注册模式：集中管理工具
- 工具查找和执行
- 错误处理
"""

from typing import Dict, List, Optional

from .base import BaseTool, ToolResult
from core.llm.base import ToolDefinition
from core.logger import get_logger


class ToolRegistry:
    """
    工具注册中心
    
    职责：
    1. 注册/注销工具
    2. 按名称查找工具
    3. 获取所有工具定义（给 LLM 用）
    4. 执行工具调用
    
    使用方法：
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WeatherTool())
        
        # 获取所有工具定义（传给 LLM）
        tools = registry.get_all_definitions()
        
        # 执行工具
        result = registry.execute("calculator", expression="2 + 3")
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    # ═══════════════════════════════════════════════════════════════
    # 工具管理
    # ═══════════════════════════════════════════════════════════════
    
    def register(self, tool: BaseTool) -> None:
        """
        注册工具
        
        Args:
            tool: 工具实例
        
        Raises:
            ValueError: 工具名称已存在
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> bool:
        """
        注销工具
        
        Args:
            name: 工具名称
        
        Returns:
            是否成功（工具不存在时返回 False）
        """
        if name in self._tools:
            del self._tools[name]
            self.logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[BaseTool]:
        """按名称获取工具"""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools
    
    # ═══════════════════════════════════════════════════════════════
    # 工具信息
    # ═══════════════════════════════════════════════════════════════
    
    def get_all_definitions(self) -> List[ToolDefinition]:
        """
        获取所有工具定义（用于传递给 LLM）
        
        Returns:
            工具定义列表
        """
        return [tool.get_definition() for tool in self._tools.values()]
    
    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())
    
    def count(self) -> int:
        """获取工具数量"""
        return len(self._tools)
    
    # ═══════════════════════════════════════════════════════════════
    # 工具执行
    # ═══════════════════════════════════════════════════════════════
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """
        执行工具
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
        
        Returns:
            执行结果
        
        注意：
        - 工具不存在时返回错误结果
        - 工具执行异常时返回错误结果
        """
        tool = self._tools.get(name)
        if not tool:
            self.logger.warning(f"Tool not found: {name}")
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"Tool '{name}' not found. Available tools: {', '.join(self.list_tools())}"
            )
        
        try:
            self.logger.info(f"Executing tool: {name} with args: {kwargs}")
            result = tool.execute(**kwargs)
            
            if result.is_error:
                self.logger.warning(f"Tool {name} failed: {result.error_message}")
            else:
                self.logger.debug(f"Tool {name} result: {result.content[:100]}...")
            
            return result
        except Exception as e:
            self.logger.error(f"Tool {name} execution error: {e}")
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"Tool execution failed: {str(e)}"
            )
    
    # ═══════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════
    
    def __repr__(self) -> str:
        return f"<ToolRegistry tools={len(self._tools)}>"
    
    def __contains__(self, name: str) -> bool:
        """支持 `in` 操作符"""
        return name in self._tools
    
    def __len__(self) -> int:
        """支持 len()"""
        return len(self._tools)
