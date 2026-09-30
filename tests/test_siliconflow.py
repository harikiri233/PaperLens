import json

import httpx
import pytest

from paperlens.models.siliconflow import SiliconFlowClient, SiliconFlowError, cosine


def _embedding_handler(requests: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        # 故意打乱返回顺序，客户端应按 index 排回来
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(text)), 1.0]}
            for i, text in enumerate(body["input"])
        ][::-1]
        return httpx.Response(
            200, json={"data": data, "usage": {"total_tokens": 7 * len(body["input"])}}
        )

    return handler


def test_embed_batches_and_keeps_order(settings):
    settings.embedding_batch_size = 2
    requests: list[dict] = []
    transport = httpx.MockTransport(_embedding_handler(requests))
    with SiliconFlowClient(settings, transport=transport) as sf:
        vectors = sf.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [len(r["input"]) for r in requests] == [2, 2, 1]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    usage = sf.tracker.snapshot()["BAAI/bge-m3"]
    assert usage.calls == 3
    assert usage.input_tokens == 35


def test_embed_rejects_empty_text(settings):
    transport = httpx.MockTransport(lambda r: httpx.Response(200))
    with SiliconFlowClient(settings, transport=transport) as sf, pytest.raises(ValueError):
        sf.embed(["ok", "  "])


def test_rerank_sorted_by_score(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["return_documents"] is False
        assert body["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.2},
                    {"index": 0, "relevance_score": 0.9},
                ],
                "meta": {"tokens": {"input_tokens": 30}},
            },
        )

    with SiliconFlowClient(settings, transport=httpx.MockTransport(handler)) as sf:
        results = sf.rerank("q", ["d0", "d1", "d2"], top_n=2)
    assert [(r.index, r.score) for r in results] == [(0, 0.9), (1, 0.2)]


def test_auth_error_is_clear(settings):
    transport = httpx.MockTransport(lambda r: httpx.Response(401, text="Invalid token"))
    with (
        SiliconFlowClient(settings, transport=transport) as sf,
        pytest.raises(SiliconFlowError, match="鉴权失败"),
    ):
        sf.embed(["x"])


def test_retries_on_rate_limit(settings, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # 跳过退避等待
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, text="TPM limit")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}], "usage": {}})

    with SiliconFlowClient(settings, transport=httpx.MockTransport(handler)) as sf:
        assert sf.embed(["x"]) == [[1.0]]
    assert attempts["n"] == 3


def test_missing_key_raises(settings):
    settings.siliconflow_api_key = None
    with pytest.raises(SiliconFlowError, match="SILICONFLOW_API_KEY"):
        SiliconFlowClient(settings)


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0
