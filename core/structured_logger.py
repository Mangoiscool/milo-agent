"""
Structured logging with JSON format support
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import settings

try:
    from pythonjsonlogger import jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False


class StructuredFormatter(logging.Formatter):
    """
    结构化日志格式器（JSON 格式）

    输出格式：
    {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "level": "INFO",
        "logger": "milo.SimpleAgent",
        "message": "Tool executed successfully",
        "context": {
            "session_id": "abc123",
            "tool_name": "calculator",
            "execution_time": 0.123
        }
    }
    """

    def __init__(self, *args, **kwargs):
        if HAS_JSON_LOGGER:
            super().__init__(*args, **kwargs)
        else:
            # Fallback to text format
            fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        if HAS_JSON_LOGGER:
            return super().format(record)

        # Text format fallback
        message = super().format(record)
        if hasattr(record, 'context') and record.context:
            import json
            try:
                context_str = json.dumps(record.context, ensure_ascii=False)
                message = f"{message} | context: {context_str}"
            except Exception:
                pass
        return message


if HAS_JSON_LOGGER:
    class StructuredFormatter(jsonlogger.JsonFormatter):
        """
        JSON 格式化器（python-json-logger 可用时）
        """

        def add_fields(
            self,
            log_record: Dict[str, Any],
            record: logging.LogRecord,
            message_dict: Dict[str, Any],
        ) -> None:
            super().add_fields(log_record, record, message_dict)

            # 标准化时间戳
            if "timestamp" not in log_record:
                log_record["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            # 添加标准字段
            log_record["level"] = record.levelname
            log_record["logger"] = record.name

            # 添加进程和线程信息
            log_record["pid"] = record.process
            log_record["thread_id"] = record.thread

            # 添加调用位置
            log_record["file"] = record.pathname
            log_record["line"] = record.lineno
            log_record["function"] = record.funcName

            # 如果有异常，添加堆栈信息
            if record.exc_info:
                log_record["exception"] = {
                    "type": record.exc_info[0].__name__,
                    "message": str(record.exc_info[1]),
                    "traceback": self.formatException(record.exc_info),
                }


class StructuredLogger:
    """
    结构化日志记录器包装器

    支持添加上下文信息
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._context: Dict[str, Any] = {}

    def bind(self, **kwargs) -> "StructuredLogger":
        """
        绑定上下文

        Returns:
            新的 StructuredLogger 实例，包含额外的上下文
        """
        new_logger = StructuredLogger(self._logger)
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def _log_with_context(
        self,
        level: int,
        msg: str,
        *args,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        带上下文记录日志
        """
        if extra is None:
            extra = {}

        # 合并上下文
        context = {**self._context, **extra}

        # 添加到 extra 字段
        log_record = self._logger.makeRecord(
            self._logger.name,
            level,
            self._logger.findCaller(),
            args,
            msg,
            None,
            None,
            extra={"context": context} if context else None,
        )
        self._logger.handle(log_record)

    def debug(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.CRITICAL, msg, **kwargs)


def setup_structured_logger(
    name: str = "milo",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    设置结构化日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径（可选）

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 使用 JSON 格式器（或文本格式）
    formatter = StructuredFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        timestamp=True,
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_structured_logger(name: str) -> StructuredLogger:
    """
    获取结构化日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        StructuredLogger 实例
    """
    # 确保根 logger 已配置
    if settings().use_structured_logging:
        if not logging.getLogger("milo").handlers:
            setup_structured_logger(
                "milo",
                level=getattr(logging, settings().log_level),
                log_file=settings().log_file,
            )
    else:
        if not logging.getLogger("milo").handlers:
            from core.logger import setup_logger as setup_text_logger
            setup_text_logger(
                "milo",
                level=getattr(logging, settings().log_level),
                log_file=settings().log_file,
            )

    return StructuredLogger(logging.getLogger(f"milo.{name}"))
