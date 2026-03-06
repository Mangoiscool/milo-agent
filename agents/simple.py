"""
Simple Agent - Minimal agent implementation
Provides basic conversational capabilities with memory
"""

from typing import AsyncIterator, List, Optional

from core.llm.base import BaseLLM, Message, Role
from core.logger import get_logger
from core.memory.base import BaseMemory
from core.memory.short_term import ShortTermMemory


class SimpleAgent:
    """
    Minimal agent implementation
    
    Features:
    - Conversational interface with memory
    - Support for system prompts
    - Synchronous and asynchronous APIs
    - Streaming support
    
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
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize SimpleAgent
        
        Args:
            llm: LLM instance for generating responses
            memory: Memory instance (default: ShortTermMemory with 50 messages)
            system_prompt: System prompt for the agent (optional)
        """
        self.llm = llm
        self.memory = memory or ShortTermMemory()
        self.system_prompt = system_prompt
        self.logger = get_logger(self.__class__.__name__)
    
    def _build_messages(self) -> List[Message]:
        """
        Build the message list for LLM
        
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
        
        # 5. Return response
        return response.content
    
    async def achat(self, user_input: str) -> str:
        """
        Asynchronous chat
        
        Args:
            user_input: User's input message
        
        Returns:
            Agent's response
        """
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
        
        # 5. Return response
        return response.content
    
    async def astream(self, user_input: str) -> AsyncIterator[str]:
        """
        Asynchronous streaming chat
        
        Args:
            user_input: User's input message
        
        Yields:
            Chunks of the agent's response
        """
        self.logger.info(f"User input (stream): {user_input[:100]}...")
        
        # 1. Create user message and add to memory
        user_message = Message(role=Role.USER, content=user_input)
        self.memory.add(user_message)
        
        # 2. Build message list
        messages = self._build_messages()
        
        # 3. Stream from LLM and collect response
        full_response = []
        async for chunk in self.llm.astream(messages):
            full_response.append(chunk)
            yield chunk
        
        # 4. Create assistant message and add to memory
        complete_response = "".join(full_response)
        assistant_message = Message(role=Role.ASSISTANT, content=complete_response)
        self.memory.add(assistant_message)
        
        self.logger.info(f"Agent response (stream complete): {complete_response[:100]}...")
    
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
