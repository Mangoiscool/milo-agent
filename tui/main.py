#!/usr/bin/env python3
"""Milo Agent TUI - Terminal User Interface

Modern chat interface inspired by Claude Code.

Usage:
    python -m tui.main
    python -m tui.main --rag --react
    python -m tui.main --theme light

Shortcuts:
    Enter      - Send message
    Ctrl+Q     - Quit
    Ctrl+L     - Clear chat
    Ctrl+N     - New chat
    Ctrl+S     - Toggle sidebar
    Ctrl+T     - Toggle thinking mode
    Ctrl+R     - Toggle RAG
    Ctrl+B     - Toggle browser
    Ctrl+,     - Settings

Commands:
    /help      - Show help
    /clear     - Clear chat
    /model     - Change model
    /quit      - Exit
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tui.app import run_tui, MiloApp
from tui.config import config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Milo Agent TUI - Terminal chat interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tui.main                    # Start with defaults
  python -m tui.main --rag --react      # Enable RAG and thinking
  python -m tui.main --browser          # Enable browser
  python -m tui.main --provider qwen    # Use Qwen provider
  python -m tui.main --theme light      # Use light theme
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
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        choices=["claude", "light"],
        help="UI theme"
    )
    parser.add_argument(
        "--no-welcome",
        action="store_true",
        help="Skip welcome screen"
    )

    return parser.parse_args()


def build_agent(args):
    """构建 Agent"""
    from agents.milo_agent import MiloAgent
    from core.llm.factory import create_llm
    from core.rag.embeddings import create_embedding

    print(f"Initializing Agent...")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model or 'default'}")

    # 创建 LLM
    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.api_key:
        kwargs["api_key"] = args.api_key

    llm = create_llm(args.provider, **kwargs)

    # 创建 embedding（如果需要）
    embedding = None
    if args.rag:
        try:
            embedding = create_embedding("ollama")
            print("  Embedding: ollama")
        except Exception as e:
            print(f"  Warning: Embedding failed: {e}")

    # 创建 Agent
    agent = MiloAgent(
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
    """主入口"""
    args = parse_args()

    # 检查 textual 是否安装
    try:
        import textual
    except ImportError:
        print("Error: textual not installed")
        print("Run: pip install textual")
        return 1

    # 保存配置
    if args.theme:
        config.set("theme", args.theme)
    if args.rag:
        config.set("default_rag", True)
    if args.browser:
        config.set("default_browser", True)
    if args.react:
        config.set("default_react", True)

    # 构建 Agent（如果参数指定）
    agent = None
    if args.provider != "ollama" or args.rag or args.browser or args.react:
        try:
            agent = build_agent(args)
        except Exception as e:
            print(f"Agent init failed: {e}")
            return 1

    # 运行 TUI
    print("\nStarting TUI...")
    print("Press Ctrl+Q to exit\n")

    try:
        app = MiloApp(agent=agent)
        if args.no_welcome:
            app.push_screen(None)  # 跳过欢迎界面
        app.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
