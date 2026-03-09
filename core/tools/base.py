"""
工具系统抽象基类

学习重点：
- 工具的自描述设计：name + description + parameters
- JSON Schema 参数定义
- 统一的执行接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel

from core.llm.base import ToolDefinition
from core.logger import get_logger


# ═══════════════════════════════════════════════════════════════
# 工具执行结果
# ═══════════════════════════════════════════════════════════════

class ToolResult(BaseModel):
    """
    工具执行结果
    
    统一的返回格式，便于 Agent 处理
    """
    content: str                    # 执行结果（字符串形式）
    is_error: bool = False          # 是否执行失败
    error_message: str = ""         # 错误信息（如果失败）


# ═══════════════════════════════════════════════════════════════
# 工具抽象基类
# ═══════════════════════════════════════════════════════════════

class BaseTool(ABC):
    """
    工具抽象基类
    
    设计原则：
    1. 自描述：工具能告诉 LLM 自己是干什么的
    2. 类型安全：参数通过 JSON Schema 验证
    3. 统一接口：所有工具用相同方式调用
    
    使用方法：
        class CalculatorTool(BaseTool):
            @property
            def name(self) -> str:
                return "calculator"
            
            @property
            def description(self) -> str:
                return "计算数学表达式"
            
            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式"
                        }
                    },
                    "required": ["expression"]
                }
            
            def execute(self, expression: str, **kwargs) -> ToolResult:
                result = eval(expression)  # 实际使用要更安全
                return ToolResult(content=str(result))
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        工具名称（唯一标识）
        
        命名规范：
        - 使用 snake_case
        - 简洁、有意义
        - 例如：calculator, get_weather, search_web
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        工具描述
        
        重要：这是 LLM 决定是否调用的依据！
        
        好的描述应该包含：
        1. 工具用途
        2. 参数说明
        3. 返回格式
        4. 使用限制
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        参数定义（JSON Schema 格式）
        
        示例：
        {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如 '北京'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位"
                }
            },
            "required": ["city"]
        }
        """
        pass
    
    def get_definition(self) -> ToolDefinition:
        """
        获取工具定义（用于传递给 LLM）
        
        这个方法将工具信息转换为 LLM 需要的格式
        """
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters
        )
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数（从 LLM 的 tool_call.arguments 解析而来）
        
        Returns:
            执行结果
        
        注意：
        - 参数已经过 JSON Schema 验证（可选）
        - 应该捕获异常，返回 is_error=True 的 ToolResult
        - 返回的 content 应该是字符串格式
        """
        pass
    
    def __repr__(self) -> str:
        return f"<Tool {self.name}>"
    
    def __str__(self) -> str:
        return f"Tool({self.name}): {self.description[:50]}..."
