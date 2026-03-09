"""
日期时间工具

功能：获取当前日期时间、格式化时间
"""

from datetime import datetime
from typing import Any, Dict

from ..base import BaseTool, ToolResult


class DateTimeTool(BaseTool):
    """
    日期时间工具
    
    功能：
    - 获取当前日期时间
    - 支持多种格式输出
    - 支持时区（简化实现）
    """
    
    @property
    def name(self) -> str:
        return "datetime"
    
    @property
    def description(self) -> str:
        return """获取当前日期和时间。

功能：
- 获取当前日期时间
- 支持多种格式输出

格式说明：
- default: 2024-03-09 22:15:30
- date: 2024-03-09
- time: 22:15:30
- iso: 2024-03-09T22:15:30
- timestamp: 1709997330（Unix 时间戳）

示例：
- 不带参数 → 返回默认格式
- format=date → 只返回日期
- format=timestamp → 返回 Unix 时间戳"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["default", "date", "time", "iso", "timestamp"],
                    "description": "输出格式：default（默认）, date, time, iso, timestamp"
                }
            },
            "required": []
        }
    
    def execute(self, format: str = "default", **kwargs) -> ToolResult:
        """
        获取当前时间
        
        Args:
            format: 输出格式
        
        Returns:
            格式化后的时间字符串
        """
        try:
            now = datetime.now()
            
            if format == "date":
                result = now.strftime("%Y-%m-%d")
            elif format == "time":
                result = now.strftime("%H:%M:%S")
            elif format == "iso":
                result = now.isoformat()
            elif format == "timestamp":
                result = str(int(now.timestamp()))
            else:  # default
                result = now.strftime("%Y-%m-%d %H:%M:%S")
            
            return ToolResult(content=result)
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"获取时间失败: {str(e)}"
            )


__all__ = ["DateTimeTool"]
