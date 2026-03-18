"""
Simple Agent - Minimal agent implementation
Provides basic conversational capabilities with memory and tool calling

This is now a thin wrapper around BaseAgent for backward compatibility.
All core functionality is implemented in BaseAgent.
"""

from typing import AsyncIterator, Callable, List, Optional

from agents.agent_config import AgentConfig
from agents.base import AgentEvent, BaseAgent
from core.llm.base import BaseLLM, Message
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.memory.short_term import ShortTermMemory
from core.tools.base import BaseTool


# Re-export AgentEvent for backward compatibility
__all__ = ["SimpleAgent", "AgentEvent"]


class SimpleAgent(BaseAgent):
    """
    Minimal agent implementation

    Features:
    - Conversational interface with memory
    - Support for system prompts
    - Synchronous and asynchronous APIs
    - Streaming support with automatic fallback
    - Event system for extensibility
    - Tool calling (Function Calling)

    This class now inherits from BaseAgent and maintains full backward compatibility.

    Usage:
        llm = create_llm("qwen", api_key="sk-xxx")
        agent = SimpleAgent(llm, system_prompt="You are a helpful assistant")

        # Sync
        response = agent.chat("Hello!")

        # Async
        response = await agent.achat("Hello!")

        # Streaming
        async for chunk in agent.astream("Hello!"):
            print(chunk, end="", flush=True)

        # With tools
        from core.tools import CalculatorTool
        agent = SimpleAgent(llm, tools=[CalculatorTool()])
        response = agent.chat_with_tools("帮我算一下 2+3")

        # Event system
        agent.on(AgentEvent.AFTER_CHAT, lambda response: print(response))
        agent.on(AgentEvent.TOOL_CALL, lambda name, args: print(f"Calling {name}"))
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        enable_stream_fallback: bool = True,
        tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize SimpleAgent

        Args:
            llm: LLM instance for generating responses
            memory: Memory instance (default: ShortTermMemory with 50 messages)
            system_prompt: System prompt for the agent (optional)
            enable_stream_fallback: Enable automatic fallback from stream to async chat (default: True)
            tools: List of tools to register (optional)
            max_tool_iterations: Maximum tool calling iterations (default: 5)
            config: AgentConfig object for unified configuration (optional, overrides other params)
        """
        # Build config from individual params if needed
        if config is None:
            config = AgentConfig(
                enable_stream_fallback=enable_stream_fallback,
                max_memory_messages=50,
                system_prompt=system_prompt,
                use_intelligent_pruning=False
            )

        # Initialize BaseAgent
        super().__init__(
            llm=llm,
            memory=memory,
            tools=tools,
            system_prompt=config.system_prompt,
            config=config,
            max_tool_iterations=max_tool_iterations
        )

        # Store additional attributes for backward compatibility
        self.enable_stream_fallback = enable_stream_fallback

    # Note: All core methods (chat, achat, astream, chat_with_tools, achat_with_tools)
    # are inherited from BaseAgent and work exactly the same way.

    # Re-expose some methods for backward compatibility with specific signatures

    def get_conversation_history(self) -> List[Message]:
        """
        Get conversation history (alias for get_history())

        Returns:
            List of all messages in memory
        """
        return self.get_history()

    def __repr__(self) -> str:
        return f"<SimpleAgent llm={self.llm} memory={self.memory} tools={self.tool_registry.count()}>"