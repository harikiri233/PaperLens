import os

import pytest
from pydantic import SecretStr

from paperlens.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """测试不读取真实密钥。

    pymilvus 在 import 时会调用 load_dotenv()，把当前目录的 .env 写进 os.environ（2026-09-30 实测），
    之后创建的 Settings 即使传了 _env_file=None 也会读到真实 Key，测试就会去调用真实接口。
    """
    for name in list(os.environ):
        if name.startswith(("LLM_", "SILICONFLOW_", "EMBEDDING_", "RERANK_", "DATA_DIR")):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """不读取 .env 的测试配置。"""
    return Settings(
        _env_file=None,
        llm_api_key=SecretStr("sk-test-llm-000000"),
        siliconflow_api_key=SecretStr("sk-test-sf-000000"),
        data_dir=tmp_path / "data",
    )
