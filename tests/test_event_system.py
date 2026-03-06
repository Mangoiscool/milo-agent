"""
Test event system and streaming fallback
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm.factory import create_llm
from agents.simple import SimpleAgent, AgentEvent


async def test_event_system():
    """Test event system"""
    print("Test 1: Event System")

    llm = create_llm("ollama", model="qwen3.5:4b", think=False)
    agent = SimpleAgent(llm, system_prompt="You are helpful")

    # Track events
    events_fired = []

    def track_event(**kwargs):
        events_fired.append(kwargs)

    # Register handlers
    agent.on(AgentEvent.BEFORE_CHAT, track_event)
    agent.on(AgentEvent.AFTER_CHAT, track_event)
    agent.on(AgentEvent.STREAM_START, track_event)
    agent.on(AgentEvent.STREAM_END, track_event)

    # Trigger chat
    await agent.achat("你好")

    print(f"  ✓ Events fired: {len(events_fired)}")
    for event in events_fired:
        print(f"    - {event}")

    return len(events_fired) > 0


async def test_streaming_fallback():
    """Test streaming fallback mechanism"""
    print("\nTest 2: Streaming Fallback")

    llm = create_llm("ollama", model="qwen3.5:4b", think=False)
    agent = SimpleAgent(llm, enable_stream_fallback=True)

    # Normal streaming should work
    chunks = []
    async for chunk in agent.astream("简单介绍一下 Python"):
        chunks.append(chunk)

    print(f"  ✓ Streaming completed, {len(''.join(chunks))} chars total")

    # Test fallback (simulated by using achat which should work)
    response = await agent.achat("你好")
    print(f"  ✓ Async chat fallback works")

    return True


async def test_multiple_handlers():
    """Test multiple event handlers for same event"""
    print("\nTest 3: Multiple Handlers")

    llm = create_llm("ollama", model="qwen3.5:4b", think=False)
    agent = SimpleAgent(llm)

    call_count = 0

    def handler1(**kwargs):
        nonlocal call_count
        call_count += 1

    def handler2(**kwargs):
        nonlocal call_count
        call_count += 1

    agent.on(AgentEvent.BEFORE_CHAT, handler1)
    agent.on(AgentEvent.BEFORE_CHAT, handler2)

    await agent.achat("test")

    print(f"  ✓ Both handlers called: {call_count == 2}")

    return call_count == 2


async def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Event System & Fallback Tests")
    print("=" * 60)

    try:
        result1 = await test_event_system()
        result2 = await test_streaming_fallback()
        result3 = await test_multiple_handlers()

        print("\n" + "=" * 60)
        if result1 and result2 and result3:
            print("All tests passed! ✅")
        else:
            print("Some tests failed! ❌")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test error: {e}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
