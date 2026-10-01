"""运行配置：从环境变量和当前目录下的 .env 读取。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    embedding_dim: int = 1024
    embedding_batch_size: int = Field(default=32, ge=1, le=32)  # 接口上限为每次 32 条
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    siliconflow_timeout: float = 30.0

    # arXiv：官方要求请求间隔不少于 3 秒
    arxiv_delay_s: float = 3.0

    # 分块（token 数按字符数 / 4 近似）
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 600

    # 检索
    collection: str = "chunks"
    retrieve_candidates: int = 30  # 稠密、BM25 各取多少候选
    retrieve_fused: int = 20  # RRF 融合后保留多少条送去重排
    retrieve_top_k: int = 6  # 重排后送入生成的条数
    evidence_threshold: float = 0.3  # 重排最高分低于它视为证据不足

    # 证据不足时是否去 arXiv 找论文（ask：询问用户；off：直接拒答）
    online_fallback: Literal["ask", "off"] = "ask"

    # 运行数据目录
    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "paperlens.db"

    @property
    def milvus_path(self) -> Path:
        return self.data_dir / "milvus.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"


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
