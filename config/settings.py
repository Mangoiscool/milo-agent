"""
Application settings with .env file support
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # 如果没有安装 pydantic-settings，提供简化版本
    import os
    from pydantic import BaseModel

    class SettingsConfigDict:
        """简化版 SettingsConfigDict"""
        pass

    class BaseSettings(BaseModel):
        """简化版 BaseSettings"""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # 从环境变量加载
            for key in self.model_fields:
                env_value = os.environ.get(key.upper())
                if env_value is not None:
                    field_type = self.model_fields[key].annotation
                    # 简单的类型转换
                    if field_type == bool:
                        env_value = env_value.lower() in ('true', '1', 'yes')
                    setattr(self, key, env_value)


class Settings(BaseSettings):
    """
    应用程序设置

    配置优先级：
    1. 环境变量
    2. .env 文件
    3. 默认值
    """

    # ====================
    # 应用配置
    # ====================
    app_name: str = "milo-agent"
    debug: bool = False
    log_level: str = "INFO"

    # ====================
    # LLM 配置
    # ====================
    # 默认 provider
    default_provider: str = "ollama"

    # Qwen 配置
    qwen_api_key: Optional[str] = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "MiniMax-M2.1"

    # GLM 配置
    glm_api_key: Optional[str] = None
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4-flash"

    # DeepSeek 配置
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # Ollama 配置
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_model: str = "qwen3.5:4b"
    ollama_think: bool = False

    # ====================
    # Agent 配置
    # ====================
    max_memory_messages: int = 50
    enable_stream_fallback: bool = True
    auto_save: bool = True
    use_intelligent_pruning: bool = False

    # ====================
    # 工具配置
    # ====================
    tool_max_retries: int = 3
    tool_initial_delay: float = 1.0
    tool_max_delay: float = 30.0

    # ====================
    # Web UI 配置
    # ====================
    webui_host: str = "0.0.0.0"
    webui_port: int = 8000
    webui_reload: bool = True  # 开发模式自动重载

    # ====================
    # 日志配置
    # ====================
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    use_structured_logging: bool = False  # JSON 格式日志

    # ====================
    # 存储配置
    # ====================
    storage_dir: Path = Path.home() / ".milo-agent"
    memory_file: Path = storage_dir / "memory.json"

    # ====================
    # 搜索配置
    # ====================
    tavily_api_key: Optional[str] = None

    # SettingsConfigDict 配置
    try:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )
    except NameError:
        pass  # 简化版本不需要这个


@lru_cache
def get_settings() -> Settings:
    """
    获取单例 Settings 实例

    Returns:
        Settings 实例
    """
    return Settings()


# 便捷访问函数
def settings() -> Settings:
    """获取设置实例的便捷函数"""
    return get_settings()
