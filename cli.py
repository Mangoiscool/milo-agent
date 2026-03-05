#!/usr/bin/env python3
"""
Milo Agent CLI

使用方式：
    python -m cli --provider ollama --think false "你好"
    python -m cli --provider ollama --model qwen3.5:4b "你的名字是什么？"
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
  python -m cli "你好"
  python -m cli --provider ollama --think false "简单介绍一下 Python"
  python -m cli --provider qwen --api-key sk-xxx --model qwen-max "写个快排"
        """
    )
    parser.add_argument(
        "prompt",
        type=str,
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

    # 设置日志
    setup_logger("milo", level=logging.DEBUG if args.debug else logging.INFO)
    logger = get_logger("CLI")

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
