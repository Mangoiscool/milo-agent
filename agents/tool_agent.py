"""
完整的 Agent 对话循环

功能：
- 接收用户输入
- 调用 LLM
- 解析工具调用
- 执行工具
- 将结果返回给 LLM
- 生成最终响应

DEPRECATED: 此实现已被弃用，请使用 simple.py 中的 SimpleAgent。
SimpleAgent 提供了更完善的实现，包括：
- 依赖注入架构（更灵活的 LLM/Memory/ToolRegistry 替换）
- 异步支持 (async/await)
- 事件系统（可扩展的钩子机制）
- 更好的记忆管理（MemoryManager）
- 更清晰的设计（Minimal agent pattern）

此文件保留作为教学参考和早期实现的示例。
"""

from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from core.llm.base import Message, Role, ToolCall, ToolDefinition
from core.llm.factory import LLMFactory
from core.tools import ToolRegistry
from core.logger import get_logger


class ToolAgent:
    """
    支持工具调用的完整 Agent
    
    对话循环：
    1. 用户输入 → messages
    2. LLM 响应（可能包含 tool_calls）
    3. 如果有 tool_calls：
       a. 执行工具
       b. 将结果添加到 messages
       c. 再次调用 LLM
    4. 生成最终响应
    5. 重复直到 LLM 不再调用工具
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4",
        tools: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        """
        初始化 Agent
        
        Args:
            llm_provider: LLM 提供商（openai, azure, ollama）
            model: 模型名称
            tools: 工具注册中心
            system_prompt: 系统提示词
            max_iterations: 最大工具调用迭代次数
            verbose: 是否显示详细信息
        """
        self.llm = LLMFactory.create(provider=llm_provider, model=model)
        self.tools = tools or ToolRegistry()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.console = Console()
        self.logger = get_logger(self.__class__.__name__)
        
        # 默认系统提示词
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # 对话历史
        self.messages: List[Message] = []
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        tool_list = '\n'.join([
            f"- {name}: {self.tools.get(name).description.split(chr(10))[0]}"
            for name in self.tools.list_tools()
        ])
        
        return f"""你是一个有用的 AI 助手，可以使用工具完成任务。

可用工具：
{tool_list}

使用规则：
1. 仔细理解用户需求
2. 判断是否需要使用工具
3. 如果需要工具，先调用工具获取信息
4. 基于工具结果生成回答
5. 如果不需要工具，直接回答

注意事项：
- 不要猜测，使用工具获取准确信息
- 如果工具调用失败，告知用户
- 保持回答简洁、准确"""
    
    def chat(self, user_input: str) -> str:
        """
        与 Agent 对话
        
        Args:
            user_input: 用户输入
        
        Returns:
            Agent 响应
        """
        # 添加用户消息
        self.messages.append(Message(role=Role.USER, content=user_input))
        
        # 显示用户输入
        if self.verbose:
            self.console.print(Panel(
                user_input,
                title="[bold blue]用户[/bold blue]",
                border_style="blue"
            ))
        
        # 对话循环
        iteration = 0
        final_response = ""
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # 调用 LLM
            if self.verbose:
                self.console.print(f"[dim]迭代 {iteration}/{self.max_iterations}...[/dim]")
            
            response = self._call_llm()
            
            # 检查是否需要工具调用
            if not response.tool_calls:
                # 没有工具调用，这是最终响应
                final_response = response.content
                break
            
            # 执行工具调用
            if self.verbose:
                self.console.print(f"[yellow]调用 {len(response.tool_calls)} 个工具...[/yellow]")
            
            tool_results = self._execute_tools(response.tool_calls)
            
            # 添加助手消息（包含 tool_calls）
            self.messages.append(Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls
            ))
            
            # 添加工具结果消息
            for result in tool_results:
                self.messages.append(Message(
                    role=Role.TOOL,
                    content=result.content,
                    name=result.name,
                    tool_call_id=result.call_id,
                    is_error=result.is_error
                ))
        
        # 显示最终响应
        if self.verbose and final_response:
            self.console.print(Panel(
                Markdown(final_response),
                title="[bold green]Agent[/bold green]",
                border_style="green"
            ))
        
        return final_response
    
    def _call_llm(self) -> Any:
        """调用 LLM"""
        # 添加系统提示词（如果还没有）
        if not any(msg.role == Role.SYSTEM for msg in self.messages):
            self.messages.insert(0, Message(
                role=Role.SYSTEM,
                content=self.system_prompt
            ))
        
        # 获取工具定义
        tools = self.tools.get_all_definitions() if self.tools.count() > 0 else None
        
        # 调用 LLM
        response = self.llm.chat_with_tools(
            messages=self.messages,
            tools=tools
        )
        
        self.logger.info(f"LLM response: finish_reason={response.finish_reason}")
        
        return response
    
    def _execute_tools(self, tool_calls: List[ToolCall]) -> List[Any]:
        """
        执行工具调用
        
        Args:
            tool_calls: 工具调用列表
        
        Returns:
            工具结果列表
        """
        results = []
        
        for tool_call in tool_calls:
            # 显示工具调用信息
            if self.verbose:
                self.console.print(f"  [cyan]→ {tool_call.name}({tool_call.arguments})[/cyan]")
            
            # 执行工具
            result = self.tools.execute(tool_call.name, **tool_call.arguments)
            
            # 显示结果（简化）
            if self.verbose:
                preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                if result.is_error:
                    self.console.print(f"  [red]✗ {preview}[/red]")
                else:
                    self.console.print(f"  [green]✓ {preview}[/green]")
            
            # 创建工具结果对象
            from core.llm.base import ToolResult as LLMToolResult
            results.append(LLMToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                content=result.content,
                is_error=result.is_error
            ))
        
        return results
    
    def clear_history(self):
        """清空对话历史"""
        self.messages = []
        if self.verbose:
            self.console.print("[dim]对话历史已清空[/dim]")
    
    def add_tool(self, tool: Any):
        """添加工具"""
        self.tools.register(tool)
        # 更新系统提示词
        self.system_prompt = self._get_default_system_prompt()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史（格式化）"""
        return [
            {
                "role": msg.role.value,
                "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            }
            for msg in self.messages
            if msg.role != Role.SYSTEM
        ]


def create_demo_agent() -> ToolAgent:
    """创建演示 Agent（带默认工具）"""
    from core.tools.builtin import (
        CalculatorTool,
        DateTimeTool,
        RandomTool,
        WeatherTool,
        WebSearchTool,
        FileReadTool,
        CodeExecutionTool
    )
    
    # 创建工具注册中心
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    registry.register(RandomTool())
    registry.register(WeatherTool())
    registry.register(WebSearchTool(engine="duckduckgo"))
    registry.register(FileReadTool())
    registry.register(CodeExecutionTool())
    
    # 创建 Agent
    agent = ToolAgent(
        llm_provider="openai",
        model="gpt-4",
        tools=registry,
        verbose=True
    )
    
    return agent


__all__ = ["ToolAgent", "create_demo_agent"]
