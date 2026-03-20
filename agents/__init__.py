"""Agent implementations for milo-agent.

This module provides various agent implementations:

- BaseAgent: Abstract base class for all agents
- SimpleAgent: Basic conversational agent with tool support
- ReActAgent: Reasoning + Acting agent with explicit thought process
- MainAgent: Unified agent with RAG, Browser, and builtin tools
- RAGAgent: Retrieval-Augmented Generation agent
- BrowserAgent: Browser automation agent

Usage:
    from agents import MainAgent, SimpleAgent, ReActAgent

    # Simple chat
    agent = SimpleAgent(llm)
    response = agent.chat("Hello!")

    # ReAct agent with reasoning
    agent = ReActAgent(llm, tools=[weather_tool])
    response = agent.chat("What's the weather?", show_reasoning=True)

    # Full-featured agent
    agent = MainAgent(
        llm=llm,
        enable_rag=True,
        embedding_model=embedding,
        enable_browser=True
    )
    response = agent.chat_with_tools("Help me with...")
"""

from .agent_config import AgentConfig
from .base import AgentEvent, BaseAgent
from .browser import BrowserAgent, browse
from .main import MainAgent
from .rag import MultiKnowledgeBaseManager, RAGAgent
from .react import ReActAgent
from .simple import SimpleAgent

__all__ = [
    # Base
    "AgentConfig",
    "BaseAgent",
    "AgentEvent",
    # Agents
    "SimpleAgent",
    "ReActAgent",
    "MainAgent",
    "RAGAgent",
    "BrowserAgent",
    # Managers
    "MultiKnowledgeBaseManager",
    # Convenience functions
    "browse",
]