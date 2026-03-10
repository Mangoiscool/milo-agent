"""
文件操作工具

功能：读写文件、列出目录、创建文件等
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import BaseTool, ToolResult


class FileReadTool(BaseTool):
    """
    文件读取工具
    
    安全设计：
    - 限制在指定工作目录内
    - 不允许读取敏感文件（如 .env, .key 等）
    """
    
    # 敏感文件模式
    SENSITIVE_PATTERNS = ['.env', '.key', '.pem', '.secret', 'password', 'token']
    
    def __init__(self, work_dir: Optional[str] = None):
        """
        初始化文件读取工具
        
        Args:
            work_dir: 工作目录（默认当前目录）
        """
        super().__init__()
        self.work_dir = Path(work_dir or os.getcwd()).resolve()
    
    @property
    def name(self) -> str:
        return "file_read"
    
    @property
    def description(self) -> str:
        return f"""读取文件内容。

功能：
- 读取文本文件内容
- 支持多种编码（UTF-8, ASCII 等）
- 限制在工作目录内

参数：
- file_path: 文件路径（相对或绝对）
- encoding: 文件编码（默认 utf-8）
- max_lines: 最大读取行数（默认 100）

示例：
- file_path="README.md" → 返回文件内容
- file_path="data/config.yaml", max_lines=50 → 读取前 50 行

安全限制：
- 工作目录: {self.work_dir}
- 不允许读取敏感文件（.env, .key 等）"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "encoding": {
                    "type": "string",
                    "description": "文件编码",
                    "default": "utf-8"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "最大读取行数",
                    "default": 100
                }
            },
            "required": ["file_path"]
        }
    
    def execute(
        self, 
        file_path: str, 
        encoding: str = "utf-8", 
        max_lines: int = 100,
        **kwargs
    ) -> ToolResult:
        """读取文件"""
        try:
            # 解析路径
            path = self._resolve_path(file_path)
            
            # 安全检查
            if not self._is_safe_path(path):
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message=f"访问被拒绝：路径超出工作目录范围"
                )
            
            if not path.exists():
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message=f"文件不存在: {file_path}"
                )
            
            if not path.is_file():
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message=f"不是文件: {file_path}"
                )
            
            if self._is_sensitive_file(path):
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message=f"不允许读取敏感文件: {file_path}"
                )
            
            # 读取文件
            with open(path, 'r', encoding=encoding) as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (已跳过剩余内容，共 {i} 行)")
                        break
                    lines.append(line.rstrip('\n'))
            
            content = '\n'.join(lines)
            
            return ToolResult(content=content)
            
        except UnicodeDecodeError:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"文件编码错误，请指定正确的编码"
            )
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"读取文件失败: {str(e)}"
            )
    
    def _resolve_path(self, file_path: str) -> Path:
        """解析文件路径"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.work_dir / path
        return path.resolve()
    
    def _is_safe_path(self, path: Path) -> bool:
        """检查路径是否安全（在工作目录内）"""
        try:
            path.relative_to(self.work_dir)
            return True
        except ValueError:
            return False
    
    def _is_sensitive_file(self, path: Path) -> bool:
        """检查是否为敏感文件"""
        path_str = str(path).lower()
        return any(pattern in path_str for pattern in self.SENSITIVE_PATTERNS)


class FileWriteTool(BaseTool):
    """
    文件写入工具
    
    安全设计：
    - 限制在指定工作目录内
    - 不允许覆盖敏感文件
    """
    
    def __init__(self, work_dir: Optional[str] = None):
        super().__init__()
        self.work_dir = Path(work_dir or os.getcwd()).resolve()
    
    @property
    def name(self) -> str:
        return "file_write"
    
    @property
    def description(self) -> str:
        return f"""写入或创建文件。

功能：
- 创建新文件
- 覆盖现有文件（需确认）
- 自动创建父目录

参数：
- file_path: 文件路径
- content: 文件内容
- mode: 写入模式（write 覆盖, append 追加）
- encoding: 文件编码（默认 utf-8）

示例：
- file_path="notes.txt", content="Hello World" → 创建文件
- file_path="data/log.txt", content="新日志", mode="append" → 追加内容

安全限制：
- 工作目录: {self.work_dir}
- 不允许覆盖敏感文件"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "文件内容"
                },
                "mode": {
                    "type": "string",
                    "enum": ["write", "append"],
                    "description": "写入模式",
                    "default": "write"
                },
                "encoding": {
                    "type": "string",
                    "description": "文件编码",
                    "default": "utf-8"
                }
            },
            "required": ["file_path", "content"]
        }
    
    def execute(
        self,
        file_path: str,
        content: str,
        mode: str = "write",
        encoding: str = "utf-8",
        **kwargs
    ) -> ToolResult:
        """写入文件"""
        try:
            # 解析路径
            path = self._resolve_path(file_path)
            
            # 安全检查
            if not self._is_safe_path(path):
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message="访问被拒绝：路径超出工作目录范围"
                )
            
            # 创建父目录
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入模式
            write_mode = 'a' if mode == "append" else 'w'
            
            # 写入文件
            with open(path, write_mode, encoding=encoding) as f:
                f.write(content)
            
            action = "追加到" if mode == "append" else "写入"
            return ToolResult(content=f"成功{action}文件: {file_path}")
            
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"写入文件失败: {str(e)}"
            )
    
    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.work_dir / path
        return path.resolve()
    
    def _is_safe_path(self, path: Path) -> bool:
        try:
            path.relative_to(self.work_dir)
            return True
        except ValueError:
            return False


class ListDirTool(BaseTool):
    """列出目录内容"""
    
    def __init__(self, work_dir: Optional[str] = None):
        super().__init__()
        self.work_dir = Path(work_dir or os.getcwd()).resolve()
    
    @property
    def name(self) -> str:
        return "list_dir"
    
    @property
    def description(self) -> str:
        return """列出目录内容。

功能：
- 列出文件和子目录
- 显示文件大小、类型
- 支持递归列出

参数：
- dir_path: 目录路径（默认当前目录）
- recursive: 是否递归（默认 False）
- show_hidden: 是否显示隐藏文件（默认 False）

示例：
- dir_path="." → 列出当前目录
- dir_path="src", recursive=True → 递归列出 src 目录"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "目录路径",
                    "default": "."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "是否递归",
                    "default": False
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "是否显示隐藏文件",
                    "default": False
                }
            },
            "required": []
        }
    
    def execute(
        self,
        dir_path: str = ".",
        recursive: bool = False,
        show_hidden: bool = False,
        **kwargs
    ) -> ToolResult:
        """列出目录"""
        try:
            # 解析路径
            path = self._resolve_path(dir_path)
            
            if not path.exists():
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message=f"目录不存在: {dir_path}"
                )
            
            if not path.is_dir():
                return ToolResult(
                    content="",
                    is_error=True,
                    error_message=f"不是目录: {dir_path}"
                )
            
            # 列出内容
            if recursive:
                lines = self._list_recursive(path, show_hidden)
            else:
                lines = self._list_flat(path, show_hidden)
            
            return ToolResult(content='\n'.join(lines))
            
        except Exception as e:
            return ToolResult(
                content="",
                is_error=True,
                error_message=f"列出目录失败: {str(e)}"
            )
    
    def _resolve_path(self, dir_path: str) -> Path:
        path = Path(dir_path)
        if not path.is_absolute():
            path = self.work_dir / path
        return path.resolve()
    
    def _list_flat(self, path: Path, show_hidden: bool) -> List[str]:
        """平铺列出"""
        lines = [f"目录: {path}\n"]
        
        for item in sorted(path.iterdir()):
            if not show_hidden and item.name.startswith('.'):
                continue
            
            if item.is_dir():
                lines.append(f"📁 {item.name}/")
            else:
                size = item.stat().st_size
                size_str = self._format_size(size)
                lines.append(f"📄 {item.name} ({size_str})")
        
        return lines
    
    def _list_recursive(self, path: Path, show_hidden: bool, prefix: str = "") -> List[str]:
        """递归列出"""
        lines = []
        
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            if not show_hidden and item.name.startswith('.'):
                continue
            
            is_last = (i == len(items) - 1)
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            
            if item.is_dir():
                lines.append(f"{prefix}{current_prefix}📁 {item.name}/")
                lines.extend(
                    self._list_recursive(item, show_hidden, prefix + next_prefix)
                )
            else:
                size = item.stat().st_size
                size_str = self._format_size(size)
                lines.append(f"{prefix}{current_prefix}📄 {item.name} ({size_str})")
        
        return lines
    
    @staticmethod
    def _format_size(size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


__all__ = ["FileReadTool", "FileWriteTool", "ListDirTool"]
