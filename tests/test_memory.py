"""Unit tests for memory system."""

import pytest

from core.llm.base import Message, Role
from core.memory.base import BaseMemory
from core.memory.short_term import ShortTermMemory


class TestShortTermMemory:
    """Tests for ShortTermMemory implementation."""
    
    def test_add_single_message(self):
        """Test adding a single message."""
        memory = ShortTermMemory()
        msg = Message(role=Role.USER, content="Hello")
        
        memory.add(msg)
        
        assert memory.count() == 1
        assert memory.get_all()[0] == msg
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        memory = ShortTermMemory()
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
            Message(role=Role.USER, content="How are you?"),
        ]
        
        for msg in messages:
            memory.add(msg)
        
        assert memory.count() == 3
        assert memory.get_all() == messages
    
    def test_get_all(self):
        """Test get_all returns copy of messages."""
        memory = ShortTermMemory()
        msg = Message(role=Role.USER, content="Test")
        memory.add(msg)
        
        all_messages = memory.get_all()
        all_messages.append(Message(role=Role.USER, content="Another"))
        
        # Original memory should not be modified
        assert memory.count() == 1
    
    def test_clear(self):
        """Test clearing memory."""
        memory = ShortTermMemory()
        memory.add(Message(role=Role.USER, content="Test"))
        memory.add(Message(role=Role.ASSISTANT, content="Response"))
        
        memory.clear()
        
        assert memory.count() == 0
        assert memory.get_all() == []
    
    def test_get_recent(self):
        """Test getting recent messages."""
        memory = ShortTermMemory()
        for i in range(5):
            memory.add(Message(role=Role.USER, content=f"Message {i}"))
        
        recent = memory.get_recent(3)
        
        assert len(recent) == 3
        assert recent[0].content == "Message 2"
        assert recent[1].content == "Message 3"
        assert recent[2].content == "Message 4"
    
    def test_get_recent_more_than_available(self):
        """Test get_recent when n > message count."""
        memory = ShortTermMemory()
        memory.add(Message(role=Role.USER, content="Only one"))
        
        recent = memory.get_recent(10)
        
        assert len(recent) == 1
    
    def test_get_recent_zero_or_negative(self):
        """Test get_recent with zero or negative n."""
        memory = ShortTermMemory()
        memory.add(Message(role=Role.USER, content="Test"))
        
        assert memory.get_recent(0) == []
        assert memory.get_recent(-5) == []
    
    def test_count(self):
        """Test count method."""
        memory = ShortTermMemory()
        assert memory.count() == 0
        
        memory.add(Message(role=Role.USER, content="One"))
        assert memory.count() == 1
        
        memory.add(Message(role=Role.ASSISTANT, content="Two"))
        assert memory.count() == 2
    
    def test_pruning_basic(self):
        """Test basic pruning when limit is exceeded."""
        memory = ShortTermMemory(max_messages=5)
        
        # Add 6 messages to trigger pruning
        for i in range(6):
            memory.add(Message(role=Role.USER, content=f"Message {i}"))
        
        # Should keep max_messages
        assert memory.count() == 5
        
        # Should keep the most recent ones
        all_messages = memory.get_all()
        assert all_messages[-1].content == "Message 5"
        assert all_messages[0].content == "Message 1"  # Oldest remaining
    
    def test_pruning_preserves_system_messages(self):
        """Test that pruning preserves SYSTEM messages."""
        memory = ShortTermMemory(max_messages=5)
        
        # Add system message first
        memory.add(Message(role=Role.SYSTEM, content="You are helpful"))
        
        # Add more user messages to exceed limit
        for i in range(6):
            memory.add(Message(role=Role.USER, content=f"User message {i}"))
        
        # SYSTEM message should still be there
        all_messages = memory.get_all()
        system_messages = [m for m in all_messages if m.role == Role.SYSTEM]
        assert len(system_messages) == 1
        assert system_messages[0].content == "You are helpful"
    
    def test_pruning_with_multiple_system_messages(self):
        """Test pruning with multiple SYSTEM messages."""
        memory = ShortTermMemory(max_messages=6)
        
        # Add 2 system messages
        memory.add(Message(role=Role.SYSTEM, content="System 1"))
        memory.add(Message(role=Role.SYSTEM, content="System 2"))
        
        # Add 6 user messages (total 8, exceeds limit)
        for i in range(6):
            memory.add(Message(role=Role.USER, content=f"User {i}"))
        
        # All SYSTEM messages should be preserved
        all_messages = memory.get_all()
        system_messages = [m for m in all_messages if m.role == Role.SYSTEM]
        assert len(system_messages) == 2
        
        # Total should be at max_messages
        assert memory.count() == 6
        
        # Most recent user messages should be kept
        user_messages = [m for m in all_messages if m.role == Role.USER]
        assert len(user_messages) == 4  # 6 - 2 system = 4 user slots
        assert user_messages[-1].content == "User 5"
    
    def test_default_max_messages(self):
        """Test default max_messages value."""
        memory = ShortTermMemory()
        assert memory.max_messages == 50
    
    def test_custom_max_messages(self):
        """Test custom max_messages value."""
        memory = ShortTermMemory(max_messages=100)
        assert memory.max_messages == 100
    
    def test_repr(self):
        """Test string representation."""
        memory = ShortTermMemory()
        assert "ShortTermMemory" in repr(memory)
        assert "count=0" in repr(memory)
        
        memory.add(Message(role=Role.USER, content="Test"))
        assert "count=1" in repr(memory)
