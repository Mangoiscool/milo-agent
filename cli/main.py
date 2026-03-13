#!/usr/bin/env python3
"""
Milo Agent CLI

使用方式：
    python -m cli.main --provider ollama --think false "你好"
    python -m cli.main --provider ollama --model qwen3.5:4b "你的名字是什么？"
"""

import argparse
import logging
from typing import Optional

from core.llm.factory import create_llm
from core.llm.base import Message
from core.logger import setup_logger, get_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Milo Agent - AI 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m cli.main "你好"
  python -m cli.main --provider ollama --think false "简单介绍一下 Python"
  python -m cli.main --provider qwen --api-key sk-xxx --model qwen-max "写个快排"
  python -m cli.main webui  # 启动 Web UI
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # chat 子命令
    chat_parser = subparsers.add_parser("chat", help="对话模式（默认）")
    chat_parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        help="要发送的消息"
    )
    chat_parser.add_argument(
        "--provider", "-p",
        type=str,
        default="ollama",
        choices=["qwen", "glm", "deepseek", "ollama"],
        help="LLM 提供者（默认: ollama）"
    )
    chat_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="模型名称（可选）"
    )
    chat_parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="API 密钥（API 提供者必需）"
    )
    chat_parser.add_argument(
        "--base-url", "-u",
        type=str,
        default=None,
        help="自定义 endpoint"
    )
    chat_parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="思考模式（仅 Ollama Qwen3 等支持）"
    )
    chat_parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="温度参数 (0.0-1.0)"
    )
    chat_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大输出 token 数"
    )

    # webui 子命令
    webui_parser = subparsers.add_parser("webui", help="启动 Web UI 界面")
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

    # 兼容旧版：如果没有子命令，默认使用 chat
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        help="要发送的消息"
    )
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
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用调试日志"
    )

    return parser.parse_args()


def build_kwargs(args) -> dict:
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


def main():
    """主函数"""
    args = parse_args()

    # 处理 webui 命令
    if args.command == "webui":
        try:
            import uvicorn
            from webui.server import app

            print("\n" + "=" * 60)
            print("  Milo Agent Web UI")
            print("=" * 60)
            print(f"  访问地址: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
            print("=" * 60)
            print("  按 Ctrl+C 停止服务器")
            print("=" * 60 + "\n")

            uvicorn.run(
                "webui.server:app",
                host=args.host,
                port=args.port,
                reload=args.reload
            )
        except ImportError as e:
            print("错误: 缺少必要的依赖包")
            print("请运行: pip install 'milo-agent[webui]'")
            print(f"详细信息: {e}")
            return 1
        except KeyboardInterrupt:
            print("\n\n服务器已停止")
        return 0

    # 默认 chat 命令（兼容旧版）
    # 设置日志
    setup_logger("milo", level=logging.DEBUG if args.debug else logging.INFO)
    logger = get_logger("CLI")

    # 检查是否提供了 prompt
    if not args.prompt:
        print("错误: 请提供要发送的消息")
        print("使用: python -m cli.main <message>")
        print("或使用: python -m cli.main chat <message>")
        print("查看帮助: python -m cli.main --help")
        return 1

    # 构建 LLM 参数
    kwargs = build_kwargs(args)

    # 创建 LLM
    logger.info(f"创建 LLM: provider={args.provider}, {kwargs}")
    try:
        llm = create_llm(args.provider, **kwargs)
    except ValueError as e:
        logger.error(f"创建 LLM 失败: {e}")
        return 1

    # 发送消息
    logger.info(f"发送消息: {args.prompt}")
    try:
        response = llm.chat([Message(role="user", content=args.prompt)])
        print(response.content)
    except Exception as e:
        logger.error(f"请求失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
