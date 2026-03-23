#!/usr/bin/env python3
"""Milo Agent CLI

使用方式：
    python -m cli.main chat "你好"
    python -m cli.main chat --provider ollama --model qwen3.5:4b "你的名字是什么？"
    python -m cli.main chat --react "帮我计算 15 + 25"
    python -m cli.main tui                    # 启动 TUI
    python -m cli.main webui                  # 启动 Web UI
"""

import argparse
import logging
import sys
from typing import Optional

from core.llm.factory import create_llm
from core.logger import setup_logger, get_logger


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """添加公共参数（LLM 配置）"""
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="ollama",
        choices=["qwen", "glm", "deepseek", "ollama"],
        help="LLM 提供者（默认: ollama）"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="模型名称（可选）"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="API 密钥（API 提供者必需）"
    )
    parser.add_argument(
        "--base-url", "-u",
        type=str,
        default=None,
        help="自定义 endpoint"
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="思考模式（仅 Ollama Qwen3 等支持）"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="温度参数 (0.0-1.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大输出 token 数"
    )


def add_capability_args(parser: argparse.ArgumentParser) -> None:
    """添加 Agent 能力开关参数"""
    parser.add_argument(
        "--tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用内置工具（默认开启）"
    )
    parser.add_argument(
        "--react",
        action="store_true",
        default=False,
        help="启用 ReAct 推理模式"
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        default=False,
        help="启用 RAG 能力"
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        default=False,
        help="启用浏览器能力"
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        default=False,
        help="启用长期记忆"
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Milo Agent - AI 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令：
  chat    单次对话模式
  tui     启动 TUI 终端界面
  webui   启动 Web UI 服务

示例：
  python -m cli.main chat "你好"
  python -m cli.main chat --provider qwen --api-key sk-xxx "写个快排"
  python -m cli.main chat --react --rag "搜索并计算"
  python -m cli.main tui --rag --react
  python -m cli.main webui --port 8080
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")
    subparsers.required = False  # 允许无子命令，在 main 中处理

    # ===== chat 子命令 =====
    chat_parser = subparsers.add_parser(
        "chat",
        help="单次对话模式",
        description="发送单条消息并获取回复"
    )
    chat_parser.add_argument(
        "prompt",
        type=str,
        help="要发送的消息"
    )
    add_common_args(chat_parser)
    add_capability_args(chat_parser)
    chat_parser.add_argument(
        "--show-reasoning",
        action="store_true",
        default=False,
        help="显示 ReAct 思考过程"
    )
    chat_parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用调试日志"
    )

    # ===== tui 子命令 =====
    tui_parser = subparsers.add_parser(
        "tui",
        help="启动 TUI 终端界面",
        description="启动交互式终端聊天界面"
    )
    add_common_args(tui_parser)
    add_capability_args(tui_parser)

    # ===== webui 子命令 =====
    webui_parser = subparsers.add_parser(
        "webui",
        help="启动 Web UI 服务",
        description="启动 Web 界面服务"
    )
    add_common_args(webui_parser)
    add_capability_args(webui_parser)
    webui_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="监听地址（默认: 0.0.0.0）"
    )
    webui_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口（默认: 8000）"
    )
    webui_parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载（开发模式）"
    )

    return parser.parse_args()


def build_llm_kwargs(args) -> dict:
    """构建 LLM 创建参数"""
    kwargs = {}

    if args.model is not None:
        kwargs["model"] = args.model
    if args.api_key is not None:
        kwargs["api_key"] = args.api_key
    if args.base_url is not None:
        kwargs["base_url"] = args.base_url
    if args.think is not None:
        kwargs["think"] = args.think
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens

    return kwargs


def build_agent_kwargs(args, llm) -> dict:
    """构建 Agent 创建参数"""
    agent_kwargs = {
        "llm": llm,
        "enable_builtin_tools": args.tools,
        "enable_react": args.react,
        "enable_rag": args.rag,
        "enable_browser": args.browser,
        "enable_long_term_memory": args.memory,
    }

    # 如果启用 RAG 或长期记忆，需要 embedding model
    if args.rag or args.memory:
        try:
            from core.rag.embeddings import create_embedding
            embedding = create_embedding("ollama")
            agent_kwargs["embedding_model"] = embedding
        except Exception as e:
            if args.rag:
                print(f"警告: RAG 需要 Embedding 模型: {e}")
                agent_kwargs["enable_rag"] = False
            if args.memory:
                print(f"警告: 长期记忆需要 Embedding 模型: {e}")
                agent_kwargs["enable_long_term_memory"] = False

    return agent_kwargs


def cmd_chat(args) -> int:
    """执行 chat 子命令"""
    from agents.milo_agent import MiloAgent

    # 设置日志
    setup_logger("milo", level=logging.DEBUG if args.debug else logging.INFO)
    logger = get_logger("CLI")

    # 构建 LLM 参数
    kwargs = build_llm_kwargs(args)

    # 创建 LLM
    logger.info(f"创建 LLM: provider={args.provider}, {kwargs}")
    try:
        llm = create_llm(args.provider, **kwargs)
    except ValueError as e:
        logger.error(f"创建 LLM 失败: {e}")
        return 1

    # 创建 Agent
    agent_kwargs = build_agent_kwargs(args, llm)
    logger.info(f"创建 Agent: tools={args.tools}, react={args.react}, "
                f"rag={args.rag}, browser={args.browser}, memory={args.memory}")

    try:
        agent = MiloAgent(**agent_kwargs)
        logger.info(f"Agent 创建成功，可用工具: {agent.list_tools()}")
    except Exception as e:
        logger.error(f"创建 Agent 失败: {e}")
        return 1

    # 使用 Agent 对话
    logger.info(f"发送消息: {args.prompt}")
    try:
        response = agent.chat_with_tools(args.prompt, show_reasoning=args.show_reasoning)
        print(response)
    except Exception as e:
        logger.error(f"请求失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def cmd_tui(args) -> int:
    """执行 tui 子命令"""
    # 延迟导入 textual，检查是否已安装
    try:
        from textual.app import App
        del App  # 仅用于检查，避免未使用导入警告
    except ImportError:
        print("错误: textual 未安装")
        print("请运行: pip install textual")
        return 1

    from tui.app import MiloTUIApp
    from agents.milo_agent import MiloAgent

    print("正在初始化 TUI...")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model or 'default'}")
    print(f"  Tools: {args.tools}, ReAct: {args.react}, RAG: {args.rag}, Browser: {args.browser}")

    try:
        # 创建 LLM
        kwargs = build_llm_kwargs(args)
        llm = create_llm(args.provider, **kwargs)

        # 创建 Agent
        agent_kwargs = build_agent_kwargs(args, llm)
        agent = MiloAgent(**agent_kwargs)

        print("Agent 就绪！")
        print("\n启动 TUI，按 Ctrl+C 退出\n")

        app = MiloTUIApp(agent=agent)
        app.run()
    except KeyboardInterrupt:
        print("\n再见！")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def cmd_webui(args) -> int:
    """执行 webui 子命令"""
    try:
        import uvicorn
    except ImportError:
        print("错误: 缺少必要的依赖包")
        print("请运行: pip install 'milo-agent[webui]'")
        return 1

    print("\n" + "=" * 60)
    print("  Milo Agent Web UI")
    print("=" * 60)
    print(f"  访问地址: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print("=" * 60)
    print("  按 Ctrl+C 停止服务器")
    print("=" * 60 + "\n")

    try:
        uvicorn.run(
            "webui.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    except KeyboardInterrupt:
        print("\n\n服务器已停止")

    return 0


def main():
    """主函数"""
    args = parse_args()

    # 根据子命令执行相应功能
    if args.command == "chat":
        return cmd_chat(args)
    elif args.command == "tui":
        return cmd_tui(args)
    elif args.command == "webui":
        return cmd_webui(args)
    else:
        # 没有子命令时显示帮助
        print("请指定子命令: chat, tui, webui")
        print()
        print("示例:")
        print('  python -m cli.main chat "你好"')
        print("  python -m cli.main tui")
        print("  python -m cli.main webui")
        print()
        print("查看完整帮助: python -m cli.main --help")
        return 1


if __name__ == "__main__":
    sys.exit(main())
