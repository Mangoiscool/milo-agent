"""
Simple Agent - Minimal agent implementation
Provides basic conversational capabilities with memory
"""

from enum import Enum
from typing import AsyncIterator, Callable, Dict, List, Optional, Union

from agents.config import AgentConfig
from core.llm.base import BaseLLM, LLMResponse, Message, Role
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.memory.short_term import ShortTermMemory


class AgentEvent(str, Enum):
    """Events that can be emitted by agent"""
    BEFORE_CHAT = "before_chat"          # Before sending user input to LLM
    AFTER_CHAT = "after_chat"            # After receiving LLM response
    STREAM_START = "stream_start"         # When streaming starts
    STREAM_CHUNK = "stream_chunk"         # For each streaming chunk
    STREAM_END = "stream_end"           # When streaming ends
    MEMORY_PRUNED = "memory_pruned"     # When memory is pruned


class SimpleAgent:
    """
    Minimal agent implementation

    Features:
    - Conversational interface with memory
    - Support for system prompts
    - Synchronous and asynchronous APIs
    - Streaming support with automatic fallback
    - Event system for extensibility

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

        # Event system
        agent.on(AgentEvent.AFTER_CHAT, lambda response: print(f"Response: {response[:50]}..."))

        # Config
        from agents.config import AgentConfig
        config = AgentConfig(
            enable_stream_fallback=True,
            max_memory_messages=100,
            system_prompt="You are a helpful assistant"
        )
        agent = SimpleAgent(llm, config=config)
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        enable_stream_fallback: bool = True,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize SimpleAgent

        Args:
            llm: LLM instance for generating responses
            memory: Memory instance (default: ShortTermMemory with 50 messages)
            system_prompt: System prompt for the agent (optional)
            enable_stream_fallback: Enable automatic fallback from stream to async chat (default: True)
            config: AgentConfig object for unified configuration (optional, overrides other params)
        """
        # Use config if provided, otherwise create from individual params
        effective_config: AgentConfig
        if config is None:
            effective_config = AgentConfig(
                enable_stream_fallback=enable_stream_fallback,
                max_memory_messages=50,
                system_prompt=system_prompt
            )
        else:
            effective_config = config

        self.llm = llm
        self.memory = memory or ShortTermMemory(effective_config.max_memory_messages)
        self.system_prompt = effective_config.system_prompt
        self.enable_stream_fallback = effective_config.enable_stream_fallback
        self.config = effective_config
        self.logger = get_logger(self.__class__.__name__)

        # Event handlers: event_type -> list[handler]
        self._handlers: Dict[AgentEvent, List[Callable]] = {}

    def on(self, event: AgentEvent, handler: Callable) -> None:
        """
        Register an event handler

        Args:
            event: Event type to listen for
            handler: Callable that will be called when event is emitted

        Example:
            agent.on(AgentEvent.AFTER_CHAT, lambda response: print(response))
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def _emit(self, event: AgentEvent, **kwargs) -> None:
        """
        Emit an event to all registered handlers

        Args:
            event: Event type to emit
            **kwargs: Arguments to pass to handlers
        """
        for handler in self._handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception as e:
                self.logger.warning(f"Event handler failed for {event.value}: {e}")

    def _build_messages(self) -> List[Message]:
        """
        Build message list for LLM

        Combines system prompt (if set) with memory messages

        Returns:
            Complete list of messages for LLM
        """
        messages = []

        # Add system prompt if configured
        if self.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))

        # Add memory messages
        messages.extend(self.memory.get_all())

        return messages

    def chat(self, user_input: str) -> str:
        """
        Synchronous chat

        Args:
            user_input: User's input message

        Returns:
            Agent's response
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="sync")

        self.logger.info(f"User input: {user_input[:100]}...")

        # 1. Create user message and add to memory
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 2. Build message list
        messages = self._build_messages()

        # 3. Call LLM
        response = self.llm.chat(messages)

        # 4. Create assistant message and add to memory
        assistant_message = Message(role=Role.ASSISTANT, content=response.content)
        self.memory.add(assistant_message)

        self.logger.info(f"Agent response: {response.content[:100]}...")

        # 5. Emit event and return response
        self._emit(AgentEvent.AFTER_CHAT, response=response.content, mode="sync")
        return response.content

    async def achat(self, user_input: str) -> str:
        """
        Asynchronous chat

        Args:
            user_input: User's input message

        Returns:
            Agent's response
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="async")

        self.logger.info(f"User input (async): {user_input[:100]}...")

        # 1. Create user message and add to memory
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 2. Build message list
        messages = self._build_messages()

        # 3. Call LLM
        response = await self.llm.achat(messages)

        # 4. Create assistant message and add to memory
        assistant_message = Message(role=Role.ASSISTANT, content=response.content)
        self.memory.add(assistant_message)

        self.logger.info(f"Agent response (async): {response.content[:100]}...")

        # 5. Emit event and return response
        self._emit(AgentEvent.AFTER_CHAT, response=response.content, mode="async")
        return response.content

    async def astream(self, user_input: str) -> AsyncIterator[str]:
        """
        Asynchronous streaming chat with automatic fallback

        If streaming fails, automatically falls back to async chat mode.

        Args:
            user_input: User's input message

        Yields:
            Chunks of the agent's response
        """
        self._emit(AgentEvent.BEFORE_CHAT, user_input=user_input, mode="stream")
        self.logger.info(f"User input (stream): {user_input[:100]}...")

        # 1. Create user message and add to memory
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)

        # 2. Build message list
        messages = self._build_messages()

        # 3. Stream from LLM and collect response
        full_response = []

        try:
            self._emit(AgentEvent.STREAM_START, user_input=user_input)
            async for chunk in self.llm.astream(messages):
                full_response.append(chunk)
                self._emit(AgentEvent.STREAM_CHUNK, chunk=chunk)
                yield chunk
            self._emit(AgentEvent.STREAM_END, complete="".join(full_response))
        except Exception as e:
            if self.enable_stream_fallback:
                self.logger.warning(f"Streaming failed: {e}, falling back to async chat")
                self._emit(AgentEvent.STREAM_END, complete=None, error=str(e))

                # Fallback: use async chat
                response = await self.llm.achat(messages)

                # Only yield if we have actual content
                if response.content:
                    yield response.content
                    full_response = [response.content]
                else:
                    self.logger.error(f"Async chat returned empty response")
                    full_response = []
            else:
                raise

        # 4. Create assistant message and add to memory
        complete_response = "".join(full_response)
        assistant_message = Message(role=Role.ASSISTANT, content=complete_response)
        self.memory.add(assistant_message)

        self.logger.info(f"Agent response (stream complete): {complete_response[:100]}...")
        self._emit(AgentEvent.AFTER_CHAT, response=complete_response, mode="stream")

    def clear_history(self) -> None:
        """Clear conversation history from memory"""
        self.memory.clear()
        self.logger.info("Conversation history cleared")

    def get_history(self) -> List[Message]:
        """
        Get conversation history

        Returns:
            List of all messages in memory
        """
        return self.memory.get_all()

    def __repr__(self) -> str:
        return f"<SimpleAgent llm={self.llm} memory={self.memory}>"
