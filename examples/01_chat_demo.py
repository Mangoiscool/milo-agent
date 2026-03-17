#!/usr/bin/env python3
"""
Interactive CLI chat demo for milo-agent

Features:
- Select LLM provider (qwen, glm, deepseek, ollama)
- Multi-turn conversation
- Command support: quit, clear, history
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.llm.factory import create_llm
from agents.simple import SimpleAgent


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("  Milo Agent - Interactive Chat Demo")
    print("=" * 60)
    print()


def select_provider():
    """Let user select LLM provider."""
    print("Available LLM providers:")
    print("  1. Qwen (通义千问) - requires API key")
    print("  2. GLM (智谱) - requires API key")
    print("  3. DeepSeek - requires API key")
    print("  4. Ollama (本地) - no API key needed")
    print()
    
    while True:
        choice = input("Select provider [1-4]: ").strip()
        
        providers = {
            "1": "qwen",
            "2": "glm",
            "3": "deepseek",
            "4": "ollama"
        }
        
        if choice in providers:
            return providers[choice]
        
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


def get_api_key(provider: str) -> str:
    """Get API key from user (if needed).

    Args:
        provider: LLM provider name

    Returns:
        API key string (or empty string if not needed)
    """
    if provider == "ollama":
        return None

    # 尝试从环境变量读取
    env_var_name = f"{provider.upper()}_API_KEY"
    import os
    api_key = os.getenv(env_var_name)

    if api_key:
        return api_key

    print(f"\nEnter your {provider.upper()} API key:")
    print("(You can also set it via environment variable later)")
    api_key = input("API key: ").strip()

    return api_key


def get_model(provider: str) -> str:
    """Get model name from user."""
    default_models = {
        "qwen": "qwen-turbo-latest",
        "glm": "glm-4-flash",
        "deepseek": "deepseek-chat",
        "ollama": "qwen3.5:4b"
    }
    
    print(f"\nDefault model for {provider}: {default_models[provider]}")
    use_default = input("Use default? [Y/n]: ").strip().lower()
    
    if use_default in ["", "y", "yes"]:
        return default_models[provider]
    
    model = input("Enter model name: ").strip()
    return model


def create_agent(provider: str, api_key: str, model: str) -> SimpleAgent:
    """Create SimpleAgent instance."""
    print(f"\nInitializing {provider} with model {model}...")
    
    llm = create_llm(provider, api_key=api_key, model=model)
    agent = SimpleAgent(
        llm,
        system_prompt="You are a helpful AI assistant. Be concise and friendly."
    )
    
    return agent


def print_help():
    """Print available commands."""
    print("\nCommands:")
    print("  quit, exit, q  - Exit the chat")
    print("  clear          - Clear conversation history")
    print("  history        - Show conversation history")
    print("  help, ?        - Show this help message")
    print()


def print_history(agent: SimpleAgent):
    """Print conversation history."""
    history = agent.get_history()
    
    if not history:
        print("\n(No conversation history)")
        return
    
    print("\n" + "=" * 60)
    print("Conversation History:")
    print("=" * 60)
    
    for msg in history:
        role_emoji = {
            "system": "⚙️",
            "user": "👤",
            "assistant": "🤖"
        }.get(msg.role.value, "💬")
        
        print(f"\n{role_emoji} {msg.role.value.upper()}:")
        print(f"  {msg.content}")
    
    print("\n" + "=" * 60)


def run_sync_chat(agent: SimpleAgent):
    """Run synchronous chat loop."""
    print("\nChat started! Type 'quit' to exit, 'help' for commands.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! 👋")
                break
            
            elif user_input.lower() == "clear":
                agent.clear_history()
                print("\n✓ Conversation history cleared")
                continue
            
            elif user_input.lower() == "history":
                print_history(agent)
                continue
            
            elif user_input.lower() in ["help", "?"]:
                print_help()
                continue
            
            # Chat
            print("\nAssistant: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


async def run_async_chat(agent: SimpleAgent):
    """Run asynchronous chat loop with streaming."""
    print("\nChat started! Type 'quit' to exit, 'help' for commands.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! 👋")
                break
            
            elif user_input.lower() == "clear":
                agent.clear_history()
                print("\n✓ Conversation history cleared")
                continue
            
            elif user_input.lower() == "history":
                print_history(agent)
                continue
            
            elif user_input.lower() in ["help", "?"]:
                print_help()
                continue
            
            # Chat with streaming
            print("\nAssistant: ", end="", flush=True)
            async for chunk in agent.astream(user_input):
                print(chunk, end="", flush=True)
            print()  # Newline after response
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def main():
    """Main entry point."""
    print_banner()
    
    # Setup
    provider = select_provider()
    api_key = get_api_key(provider)
    model = get_model(provider)
    
    # Create agent
    try:
        agent = create_agent(provider, api_key, model)
        print("\n✓ Agent initialized successfully!")
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {e}")
        sys.exit(1)
    
    # Ask for chat mode
    print("\nChat mode:")
    print("  1. Synchronous (simple)")
    print("  2. Asynchronous with streaming (recommended)")
    mode = input("Select mode [1-2, default=2]: ").strip()
    
    # Run chat
    if mode == "1":
        run_sync_chat(agent)
    else:
        asyncio.run(run_async_chat(agent))


if __name__ == "__main__":
    main()
