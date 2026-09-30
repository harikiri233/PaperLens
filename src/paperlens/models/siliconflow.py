"""硅基流动的 Embedding 与 Reranker 客户端。"""

import math
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from paperlens.config import Settings
from paperlens.models.usage import UsageTracker

_MAX_EMBED_BATCH = 32  # 接口限制：input 数组最多 32 条


class SiliconFlowError(RuntimeError):
    pass


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(5),
    reraise=True,
)


@dataclass
class RerankResult:
    index: int  # 在输入 documents 中的位置
    score: float


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class SiliconFlowClient:
    def __init__(
        self,
        settings: Settings,
        *,
        tracker: UsageTracker | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """transport 参数用于测试时注入 httpx.MockTransport。"""
        if settings.siliconflow_api_key is None:
            raise SiliconFlowError("未配置 SILICONFLOW_API_KEY，请在 .env 中填写。")
        self.settings = settings
        self.tracker = tracker or UsageTracker()
        self._http = httpx.Client(
            base_url=settings.siliconflow_base_url,
            headers={"Authorization": f"Bearer {settings.siliconflow_api_key.get_secret_value()}"},
            timeout=settings.siliconflow_timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SiliconFlowClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @_retry
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._http.post(path, json=payload)
        if response.status_code in (401, 403):
            raise SiliconFlowError(
                f"鉴权失败（HTTP {response.status_code}），请检查 SILICONFLOW_API_KEY。"
            )
        response.raise_for_status()
        return response.json()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成向量，返回顺序与输入一致。"""
        if not texts:
            return []
        if any(not t.strip() for t in texts):
            raise ValueError("embed 的输入里有空字符串，接口不接受空文本。")
        batch_size = min(self.settings.embedding_batch_size, _MAX_EMBED_BATCH)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            data = self._post(
                "/embeddings",
                {
                    "model": self.settings.embedding_model,
                    "input": batch,
                    "encoding_format": "float",
                },
            )
            items = sorted(data["data"], key=lambda item: item["index"])
            if len(items) != len(batch):
                raise SiliconFlowError(f"返回了 {len(items)} 个向量，期望 {len(batch)} 个。")
            vectors.extend(item["embedding"] for item in items)
            tokens = (data.get("usage") or {}).get("total_tokens", 0)
            self.tracker.record(self.settings.embedding_model, input_tokens=tokens)
        return vectors

    def rerank(
        self, query: str, documents: list[str], *, top_n: int | None = None
    ) -> list[RerankResult]:
        """用 cross-encoder 对文档重新打分，按分数从高到低返回。"""
        if not documents:
            return []
        payload: dict[str, Any] = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        data = self._post("/rerank", payload)
        results = [
            RerankResult(index=item["index"], score=float(item["relevance_score"]))
            for item in data["results"]
        ]
        tokens = ((data.get("meta") or {}).get("tokens") or {}).get("input_tokens", 0)
        self.tracker.record(self.settings.rerank_model, input_tokens=tokens)
        return sorted(results, key=lambda r: r.score, reverse=True)
