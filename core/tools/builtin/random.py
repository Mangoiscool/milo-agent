"""
随机数生成工具

功能：生成随机整数、浮点数、随机选择
"""

import random
from typing import Any, Dict, List

from ..base import BaseTool, ToolResult


class RandomTool(BaseTool):
    """
    随机数生成工具
    
    功能：
    - 生成随机整数
    - 生成随机浮点数
    - 从列表中随机选择
    - 生成随机字符串
    """
    
    @property
    def name(self) -> str:
        return "random"
    
    @property
    def description(self) -> str:
        return """生成随机数或随机选择。

功能：
1. 生成随机整数：min=1, max=100 → 随机整数
2. 生成随机浮点数：min=0, max=1, float=true → 随机浮点数
3. 从列表选择：choices=["a","b","c"], pick=true → 随机选择一个
4. 生成随机字符串：length=10 → 10位随机字符串

示例：
- min=1, max=100 → 42
- choices=["苹果", "香蕉", "橙子"], pick=true → "香蕉"
- length=8 → "a1B2c3D4" """

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "min": {
                    "type": "number",
                    "description": "最小值（默认 0）"
                },
                "max": {
                    "type": "number",
                    "description": "最大值（默认 100）"
                },
                "float": {
                    "type": "boolean",
                    "description": "是否返回浮点数（默认 false）"
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "选项列表，从中随机选择"
                },
                "pick": {
                    "type": "boolean",
                    "description": "是否从 choices 中选择（默认 false）"
                },
                "length": {
                    "type": "integer",
                    "description": "随机字符串长度（默认 8）"
                }
            },
            "required": []
        }
    
    def execute(
        self,
        min: float = 0,
        max: float = 100,
        float: bool = False,
        choices: List[str] = None,
        pick: bool = False,
        length: int = None,
        **kwargs
    ) -> ToolResult:
        """
        生成随机数或随机选择
        
        Args:
            min: 最小值
            max: 最大值
            float: 是否返回浮点数
            choices: 选项列表
            pick: 是否从 choices 中选择
            length: 随机字符串长度
        
        Returns:
            随机结果
        """
        try:
            # 从列表中选择
            if pick and choices:
                result = random.choice(choices)
                return ToolResult(content=str(result))
            
            # 生成随机字符串
            if length:
                import string
                chars = string.ascii_letters + string.digits
                result = ''.join(random.choices(chars, k=length))
                return ToolResult(content=result)
            
            # 生成随机数
            if float:
                result = random.uniform(min, max)
                return ToolResult(content=f"{result:.6f}")
            else:
                result = random.randint(int(min), int(max))
                return ToolResult(content=str(result))
                
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"生成随机数失败: {str(e)}"
            )


__all__ = ["RandomTool"]
