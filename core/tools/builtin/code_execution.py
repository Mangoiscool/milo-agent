"""
代码执行工具

功能：安全执行 Python 代码
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseTool, ToolResult


from core.logger import get_logger


class CodeExecutionTool(BaseTool):
    """Python 代码执行工具"""
    
    DANGEROUS_MODULES = [
        'os.system', 'os.popen', 'subprocess.Popen', 
        'eval', 'exec', 'compile', '__import__'
    ]
    
    def __init__(self, timeout: int = 10, max_output: int = 10000):
        super().__init__()
        self.timeout = timeout
        self.max_output = max_output
    
    @property
    def name(self) -> str:
        return "code_execute"
    
    @property
    def description(self) -> str:
        return f"执行 Python 代码并返回结果。超时 {self.timeout} 秒。不支持危险操作。"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python 代码"
                },
                "timeout": {
                    "type": "integer",
                    "description": "执行超时(秒)",
                    "default": self.timeout
                }
            },
            "required": ["code"]
        }
    
    def execute(self, code: str, timeout: Optional[int] = None, **kwargs) -> ToolResult:
        """执行代码"""
        try:
            # 安全检查
            safety_issue = self._check_safety(code)
            if safety_issue:
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message=f"安全检查失败: {safety_issue}"
                )
            
            # 使用默认超时
            exec_timeout = timeout or self.timeout
            
            
            # 执行代码
            stdout, stderr, returncode = self._execute_code(code, exec_timeout)
            
            # 处理输出
            if len(stdout) > self.max_output:
                stdout = stdout[:self.max_output] + "... (输出过长，已截断)"
            
            # 构建结果
            content = "输出:\\n" + stdout
            if stderr:
                content += f"\\n错误:\\n{stderr}"
            if returncode != 0:
                content += f"\\n退出码: {returncode}"
                is_error = returncode != 0
            return ToolResult(
                content=content,
                is_error=is_error,
                error_message=stderr if is_error else ""
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"执行超时({self.timeout}秒)"
            )
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"执行失败: {str(e)}"
            )
    
    def _check_safety(self, code: str) -> Optional[str]:
        """检查代码安全性"""
        code_lower = code.lower()
        for module in self.DANGEROUS_MODULES:
            if module in code_lower:
                return f"检测到潜在危险操作: {module}"
        return None
    
    def _execute_code(self, code: str, timeout: int) -> tuple[str, str, int]:
        """执行代码"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout, result.stderr, result.returncode
        finally:
            Path(temp_file).unlink(missing_ok=True)


class ShellCommandTool(BaseTool):
    """Shell 命令执行工具"""
    
    ALLOWED_COMMANDS = [
        'ls', 'pwd', 'echo', 'cat', 'date', 'whoami', 'which',
        'grep', 'find', 'wc', 'head', 'tail'
 'mkdir', 'touch', 'rm', 'cp', 'mv'
    ]
    
    def __init__(self, work_dir: str = None, timeout: int = 5):
        super().__init__()
        self.work_dir = Path(work_dir or os.getcwd()).resolve()
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "shell_command"
    
    @property
    def description(self) -> str:
        allowed = ', '.join(self.ALLOWED_COMMANDS[:10])
        return f"执行 Shell 命令。允许命令: {', '.join(self.ALLOWED_COMMANDS[:10])}.超时 {self.timeout} 秒。 """
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell 命令"
                }
            },
            "required": ["command"]
        }
    
    def execute(self, command: str, **kwargs) -> ToolResult:
        """执行 Shell 命令"""
        try:
            cmd_name = command.split()[0] if command.split() else ''
            
            if cmd_name not in self.ALLOWED_COMMANDS:
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message=f"命令 '{cmd_name}' 不在允许列表中"
                )
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.work_dir
            )
            
            output = result.stdout or result.stderr
            
            if result.returncode != 0:
                return ToolResult(
                    content=output,
                    is_error=True,
                    error_message=result.stderr or "命令执行失败"
                )
            
            return ToolResult(content=output or "命令执行成功")
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"命令执行超时({self.timeout}秒)"
            )
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"执行失败: {str(e)}"
            )


__all__ = ["CodeExecutionTool", "ShellCommandTool"]
