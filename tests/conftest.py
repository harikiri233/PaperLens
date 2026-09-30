import pytest
from pydantic import SecretStr

from paperlens.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """不读取 .env 的测试配置。"""
    return Settings(
        _env_file=None,
        llm_api_key=SecretStr("sk-test-llm-000000"),
        siliconflow_api_key=SecretStr("sk-test-sf-000000"),
        data_dir=tmp_path / "data",
    )
