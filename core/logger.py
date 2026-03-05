"""
统一日志配置

学习重点：
- 使用 logging 模块而非 print，便于生产环境管理
- 支持日志级别、格式化、文件输出
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "milo",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径（可选）
        format_string: 自定义格式字符串（可选）

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 默认格式：时间 + 级别 + 名称 + 消息
    if format_string is None:
        format_string = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已配置的日志记录器

    使用示例：
        from core.logger import get_logger

        logger = get_logger(__name__)
        logger.info("这是一条信息")
        logger.debug("这是一条调试信息")
    """
    # 确保根 logger 已配置
    if not logging.getLogger("milo").handlers:
        setup_logger("milo")

    return logging.getLogger(f"milo.{name}")
