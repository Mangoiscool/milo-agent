"""
计算器工具 - 示例实现

学习重点：
- BaseTool 的具体实现
- 安全的表达式计算（不用 eval）
- JSON Schema 参数定义
"""

import ast
import operator
from typing import Any, Dict

from ..base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """
    计算器工具
    
    功能：安全地计算数学表达式
    
    支持的运算：
    - 加减乘除：+, -, *, /
    - 幂运算：**
    - 括号：()
    - 负数：-5
    
    不支持：
    - 函数调用
    - 变量
    - 字符串操作
    """
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return """计算数学表达式的结果。

支持：
- 基本运算：加减乘除 (+, -, *, /)
- 幂运算：**
- 括号：()
- 负数：-5

示例：
- 2 + 3 * 4 → 14
- (10 - 2) / 4 → 2.0
- 2 ** 10 → 1024

注意：只支持纯数学表达式，不要输入其他内容。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 3 * 4' 或 '(10 - 2) / 4'"
                }
            },
            "required": ["expression"]
        }
    
    def execute(self, expression: str, **kwargs) -> ToolResult:
        """
        安全地计算表达式
        
        Args:
            expression: 数学表达式
        
        Returns:
            计算结果
        """
        try:
            # 清理表达式
            expr = expression.strip()
            
            # 安全计算
            result = self._safe_eval(expr)
            
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"计算错误: {str(e)}"
            )
    
    def _safe_eval(self, expr: str) -> float:
        """
        安全的表达式计算
        
        使用 AST 解析，只允许数学运算，不允许函数调用和变量
        
        Args:
            expr: 数学表达式字符串
        
        Returns:
            计算结果
        
        Raises:
            ValueError: 不支持的表达式
        """
        # 允许的运算符映射
        OPERATORS = {
            ast.Add: operator.add,      # +
            ast.Sub: operator.sub,      # -
            ast.Mult: operator.mul,     # *
            ast.Div: operator.truediv,  # /
            ast.Pow: operator.pow,      # **
            ast.USub: operator.neg,     # 一元负号 -
        }
        
        def eval_node(node):
            """递归计算 AST 节点"""
            # 数字常量
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"不支持的常量类型: {type(node.value)}")
            
            # 二元运算：a + b, a * b 等
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op_type = type(node.op)
                
                if op_type not in OPERATORS:
                    raise ValueError(f"不支持的运算符: {op_type.__name__}")
                
                return OPERATORS[op_type](left, right)
            
            # 一元运算：-5
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                op_type = type(node.op)
                
                if op_type not in OPERATORS:
                    raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
                
                return OPERATORS[op_type](operand)
            
            # 表达式节点
            elif isinstance(node, ast.Expression):
                return eval_node(node.body)
            
            else:
                raise ValueError(f"不支持的表达式类型: {type(node).__name__}")
        
        # 解析表达式
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"表达式语法错误: {e}")
        
        # 计算结果
        return eval_node(tree)


# 方便导入
__all__ = ["CalculatorTool"]
