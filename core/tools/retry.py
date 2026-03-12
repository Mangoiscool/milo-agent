"""
Retry mechanism for tool execution
"""

import asyncio
import time
from functools import wraps
from typing import Callable, Optional

from core.logger import get_logger


# 可重试的异常类型
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# 可重试的错误消息关键字
RETRYABLE_PATTERNS = [
    "timeout",
    "connection",
    "network",
    "rate limit",
    "429",
    "503",
    "502",
    "504",
]


def is_retryable_error(error: Exception) -> bool:
    """
    判断错误是否可重试

    Args:
        error: 异常对象

    Returns:
        是否可重试
    """
    # 检查异常类型
    if isinstance(error, RETRYABLE_EXCEPTIONS):
        return True

    # 检查错误消息
    error_msg = str(error).lower()
    return any(pattern in error_msg for pattern in RETRYABLE_PATTERNS)


class RetryConfig:
    """重试配置"""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        初始化重试配置

        Args:
            max_retries: 最大重试次数
            initial_delay: 初始延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            exponential_base: 指数退避基数
            jitter: 是否添加随机抖动
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter  # 添加随机抖动避免惊群效应

    def get_delay(self, attempt: int) -> float:
        """
        计算重试延迟（指数退避 + 抖动）

        Args:
            attempt: 当前重试次数（从0开始）

        Returns:
            延迟时间（秒）
        """
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)

        return delay


def retry_tool(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    工具执行重试装饰器

    Args:
        config: 重试配置，为 None 时使用默认配置
        on_retry: 每次重试前的回调，接收 (异常, 重试次数)

    Returns:
        装饰器函数

    Example:
        @retry_tool(max_retries=3)
        def my_tool(**kwargs):
            # Tool implementation
            pass
    """
    if config is None:
        config = RetryConfig()

    logger = get_logger("RetryTool")

    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # 最后一次尝试失败，不再重试
                    if attempt >= config.max_retries:
                        break

                    # 检查是否可重试
                    if not is_retryable_error(e):
                        logger.debug(f"Error is not retryable: {e}")
                        break

                    # 执行重试回调
                    if on_retry:
                        on_retry(e, attempt)

                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Tool execution failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            # 所有重试都失败
            logger.error(f"Tool execution failed after {config.max_retries} retries: {last_error}")
            raise last_error

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if attempt >= config.max_retries:
                        break

                    if not is_retryable_error(e):
                        break

                    if on_retry:
                        on_retry(e, attempt)

                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Tool execution failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

            logger.error(f"Tool execution failed after {config.max_retries} retries: {last_error}")
            raise last_error

        # 返回适当的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
