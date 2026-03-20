#!/usr/bin/env python3
"""Milo Agent TUI - Terminal User Interface

Modern chat interface inspired by Claude Code.

Usage:
    python -m tui.main
    python -m tui.main --rag --react

Shortcuts:
    Ctrl+C  - Quit
    Ctrl+L  - Clear chat
    Ctrl+T  - Toggle thinking mode
    Ctrl+R  - Toggle RAG
    Ctrl+B  - Toggle browser

Commands:
    /help    - Show help
    /clear   - Clear chat
    /model   - Change model
    /quit    - Exit
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tui.app import run_tui, MiloTUIApp
from agents.main import MainAgent
from core.llm.factory import create_llm
from core.rag.embeddings import create_embedding


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Milo Agent TUI - Terminal chat interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tui.main                    # Start with defaults
  python -m tui.main --rag --react      # Enable RAG and thinking
  python -m tui.main --browser          # Enable browser
  python -m tui.main --provider qwen    # Use Qwen provider
        """
    )

    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="ollama",
        choices=["qwen", "glm", "deepseek", "ollama"],
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="API key"
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG"
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Enable browser"
    )
    parser.add_argument(
        "--react", "--think",
        action="store_true",
        dest="react",
        help="Enable thinking mode (ReAct)"
    )

    return parser.parse_args()


def build_agent(args) -> MainAgent:
    """Build the agent"""
    print(f"Initializing Agent...")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model or 'default'}")

    # Create LLM
    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.api_key:
        kwargs["api_key"] = args.api_key

    llm = create_llm(args.provider, **kwargs)

    # Create embedding if needed
    embedding = None
    if args.rag:
        try:
            embedding = create_embedding("ollama")
            print("  Embedding: ollama")
        except Exception as e:
            print(f"  Warning: Embedding failed: {e}")

    # Create agent
    agent = MainAgent(
        llm=llm,
        enable_builtin_tools=True,
        enable_rag=args.rag and embedding is not None,
        embedding_model=embedding,
        enable_browser=args.browser,
        enable_react=args.react,
    )

    print("Agent ready!")
    return agent


def main():
    """Main entry point"""
    args = parse_args()

    # Check textual is installed
    try:
        import textual
    except ImportError:
        print("Error: textual not installed")
        print("Run: pip install textual")
        return 1

    # Build agent
    try:
        agent = build_agent(args)
    except Exception as e:
        print(f"Agent init failed: {e}")
        return 1

    # Run TUI
    print("\nStarting TUI...")
    print("Press Ctrl+C to exit\n")

    try:
        app = MiloTUIApp(agent=agent)
        app.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
