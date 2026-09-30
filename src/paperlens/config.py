"""运行配置：从环境变量和当前目录下的 .env 读取。"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # .env 里留空的 KEY= 视为未配置
        extra="ignore",
    )

    # LLM（OpenAI 兼容协议，默认 DeepSeek）
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-flash"
    llm_send_thinking_param: bool = True
    llm_timeout: float = 60.0

    # 硅基流动：Embedding 与 Reranker
    siliconflow_api_key: SecretStr | None = None
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = Field(default=32, ge=1, le=32)  # 接口上限为每次 32 条
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    siliconflow_timeout: float = 30.0

    # 运行数据目录
    data_dir: Path = Path("data")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def mask_secret(secret: SecretStr | None) -> str:
    """用于展示的密钥掩码，只保留末 4 位。"""
    if secret is None:
        return "（未配置）"
    value = secret.get_secret_value()
    if len(value) <= 8:
        return "****"
    return f"{value[:3]}****{value[-4:]}"
