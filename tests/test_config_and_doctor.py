import httpx

from paperlens.config import Settings, mask_secret
from paperlens.doctor import check_arxiv, check_config, run_all
from paperlens.models import UsageTracker


def test_empty_env_values_are_ignored(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=\nLLM_MODEL=deepseek-flash\n", encoding="utf-8")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = Settings(_env_file=env)
    assert settings.llm_api_key is None
    assert settings.llm_model == "deepseek-flash"


def test_mask_secret(settings):
    masked = mask_secret(settings.llm_api_key)
    assert "test-llm" not in masked
    assert masked.endswith("0000")
    assert mask_secret(None) == "（未配置）"


def test_doctor_skips_without_keys(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    results = run_all(settings, UsageTracker(), skip_arxiv=True)
    by_name = {r.name: r.status for r in results}
    assert by_name == {
        "配置": "warn",
        "LLM": "skip",
        "Embedding": "skip",
        "Rerank": "skip",
        "arXiv": "skip",
    }


def test_check_config_ok(settings):
    assert check_config(settings).status == "ok"


def test_check_arxiv_with_mock(monkeypatch):
    monkeypatch.setattr("paperlens.doctor.time.sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "export.arxiv.org":
            return httpx.Response(200, text="<feed><entry>...</entry></feed>")
        return httpx.Response(200, text='<div class="ltx_para">...</div>')

    result = check_arxiv(transport=httpx.MockTransport(handler))
    assert result.status == "ok"


def test_check_arxiv_network_failure(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = check_arxiv(transport=httpx.MockTransport(handler))
    assert result.status == "fail"
    assert any("HTTPS_PROXY" in d for d in result.details)
