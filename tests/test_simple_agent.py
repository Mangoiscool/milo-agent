"""Unit tests for SimpleAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.llm.base import BaseLLM, Message, Role, LLMResponse
from core.memory.short_term import ShortTermMemory
from agents.simple import SimpleAgent


class TestSimpleAgent:
    """Tests for SimpleAgent implementation."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = MagicMock(spec=BaseLLM)
        llm.model = "test-model"
        llm.__repr__ = lambda self: "<MockLLM model=test-model>"
        return llm
    
    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return LLMResponse(
            content="This is a test response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create a SimpleAgent instance with mock LLM."""
        return SimpleAgent(mock_llm)
    
    @pytest.fixture
    def agent_with_system(self, mock_llm):
        """Create a SimpleAgent with system prompt."""
        return SimpleAgent(
            mock_llm,
            system_prompt="You are a helpful test assistant"
        )
    
    def test_init_basic(self, mock_llm):
        """Test basic initialization."""
        agent = SimpleAgent(mock_llm)
        
        assert agent.llm == mock_llm
        assert agent.system_prompt is None
        assert isinstance(agent.memory, ShortTermMemory)
    
    def test_init_with_custom_memory(self, mock_llm):
        """Test initialization with custom memory."""
        memory = ShortTermMemory(max_messages=100)
        agent = SimpleAgent(mock_llm, memory=memory)
        
        assert agent.memory == memory
    
    def test_init_with_system_prompt(self, mock_llm):
        """Test initialization with system prompt."""
        agent = SimpleAgent(
            mock_llm,
            system_prompt="You are helpful"
        )
        
        assert agent.system_prompt == "You are helpful"
    
    def test_chat_single_turn(self, agent, mock_llm, mock_llm_response):
        """Test single turn conversation."""
        mock_llm.chat.return_value = mock_llm_response
        
        response = agent.chat("Hello")
        
        assert response == "This is a test response"
        
        # Verify LLM was called with correct messages
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        
        # Should have USER message
        assert len(messages) == 1
        assert messages[0].role == Role.USER
        assert messages[0].content == "Hello"
        
        # Memory should have 2 messages (user + assistant)
        assert agent.memory.count() == 2
    
    def test_chat_with_system_prompt(self, agent_with_system, mock_llm, mock_llm_response):
        """Test chat with system prompt."""
        mock_llm.chat.return_value = mock_llm_response
        
        response = agent_with_system.chat("Hello")
        
        assert response == "This is a test response"
        
        # Verify system prompt was included
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        
        # Should have SYSTEM + USER messages
        assert len(messages) == 2
        assert messages[0].role == Role.SYSTEM
        assert messages[0].content == "You are a helpful test assistant"
        assert messages[1].role == Role.USER
    
    def test_chat_multi_turn(self, agent, mock_llm):
        """Test multi-turn conversation."""
        # First turn
        mock_llm.chat.return_value = LLMResponse(content="Response 1")
        response1 = agent.chat("Turn 1")
        
        # Second turn
        mock_llm.chat.return_value = LLMResponse(content="Response 2")
        response2 = agent.chat("Turn 2")
        
        assert response1 == "Response 1"
        assert response2 == "Response 2"
        
        # Memory should have 4 messages (2 user + 2 assistant)
        assert agent.memory.count() == 4
        
        # Verify second call included conversation history
        # Note: assistant message for "Turn 2" is added AFTER LLM call
        # So at the time of LLM call, memory has 3 messages:
        # - Turn 1 (USER)
        # - Response 1 (ASSISTANT)
        # - Turn 2 (USER)
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        
        # Should have 3 messages (assistant response is added after call)
        assert len(messages) == 3
        assert messages[0].content == "Turn 1"
        assert messages[1].content == "Response 1"
        assert messages[2].content == "Turn 2"
    
    @pytest.mark.asyncio
    async def test_achat(self, agent, mock_llm, mock_llm_response):
        """Test asynchronous chat."""
        mock_llm.achat = AsyncMock(return_value=mock_llm_response)
        
        response = await agent.achat("Hello async")
        
        assert response == "This is a test response"
        assert mock_llm.achat.called
        
        # Memory should be updated
        assert agent.memory.count() == 2
    
    @pytest.mark.asyncio
    async def test_astream(self, agent, mock_llm):
        """Test asynchronous streaming."""
        # Mock streaming response
        async def mock_stream(messages):
            for chunk in ["Hello", " ", "world", "!"]:
                yield chunk
        
        mock_llm.astream = MagicMock(return_value=mock_stream(None))
        
        chunks = []
        async for chunk in agent.astream("Stream test"):
            chunks.append(chunk)
        
        assert chunks == ["Hello", " ", "world", "!"]
        
        # Memory should have complete response
        assert agent.memory.count() == 2
        history = agent.memory.get_all()
        assert history[1].content == "Hello world!"
    
    def test_clear_history(self, agent, mock_llm, mock_llm_response):
        """Test clearing conversation history."""
        mock_llm.chat.return_value = mock_llm_response
        
        # Add some messages
        agent.chat("Message 1")
        agent.chat("Message 2")
        
        assert agent.memory.count() == 4
        
        # Clear history
        agent.clear_history()
        
        assert agent.memory.count() == 0
    
    def test_get_history(self, agent, mock_llm, mock_llm_response):
        """Test getting conversation history."""
        mock_llm.chat.return_value = mock_llm_response
        
        agent.chat("User message")
        
        history = agent.get_history()
        
        assert len(history) == 2
        assert history[0].role == Role.USER
        assert history[0].content == "User message"
        assert history[1].role == Role.ASSISTANT
    
    def test_get_history_returns_copy(self, agent):
        """Test that get_history returns a copy."""
        history = agent.get_history()
        history.append(Message(role=Role.USER, content="New"))
        
        # Original memory should not be modified
        assert agent.memory.count() == 0
    
    def test_repr(self, agent):
        """Test string representation."""
        repr_str = repr(agent)
        assert "SimpleAgent" in repr_str
        assert "MockLLM" in repr_str
        assert "ShortTermMemory" in repr_str
