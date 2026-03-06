"""
Memory system base class
Defines the abstract interface for memory implementations
"""

from abc import ABC, abstractmethod
from typing import List

from core.llm.base import Message
from core.logger import get_logger


class BaseMemory(ABC):
    """
    Memory system abstract base class
    
    Design principles:
    - Unified interface for different memory strategies
    - Short-term, long-term, and hybrid implementations
    - Automatic message management (pruning, summarization)
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def add(self, message: Message) -> None:
        """
        Add a message to memory
        
        Args:
            message: Message to add
        """
        pass
    
    @abstractmethod
    def get_all(self) -> List[Message]:
        """
        Get all messages in memory
        
        Returns:
            List of all messages
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from memory"""
        pass
    
    @abstractmethod
    def get_recent(self, n: int) -> List[Message]:
        """
        Get the most recent n messages
        
        Args:
            n: Number of messages to retrieve
        
        Returns:
            List of recent messages
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Get total message count
        
        Returns:
            Number of messages in memory
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} count={self.count()}>"
