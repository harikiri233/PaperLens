"""PaperLens API 的 HTTP 客户端，供 Streamlit 界面使用（测试时可以传入 FastAPI 的 TestClient）。"""

import json
from collections.abc import Iterator
from typing import Any

import httpx


class APIError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}：{detail}")
        self.status = status
        self.detail = detail


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text
    if isinstance(detail, list):  # 422 校验错误
        return "；".join(str(d.get("msg", d)) for d in detail)
    return str(detail)


def iter_sse(lines: Iterator[str]) -> Iterator[dict]:
    """解析 SSE 文本流：每个事件的 data 行是一个 JSON 对象。忽略注释行（ping）。"""
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield json.loads("\n".join(data))
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield json.loads("\n".join(data))


class PaperLensClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        api_key: str | None = None,
        http: httpx.Client | None = None,
        timeout: float = 180.0,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self.http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self.http.headers.update(headers)

    def close(self) -> None:
        self.http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.http.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise APIError(response.status_code, _detail(response))
        return response.json()

    def _stream(self, path: str, body: dict) -> Iterator[dict]:
        with self.http.stream("POST", path, json=body) as response:
            if response.status_code >= 400:
                response.read()
                raise APIError(response.status_code, _detail(response))
            yield from iter_sse(response.iter_lines())

    # ---- 问答 ----

    def chat(self, question: str, thread_id: str) -> Iterator[dict]:
        return self._stream("/chat", {"question": question, "thread_id": thread_id})

    def resume(self, thread_id: str, picked: list[str]) -> Iterator[dict]:
        return self._stream(f"/chat/{thread_id}/resume", {"picked": picked})

    def history(self, thread_id: str) -> dict:
        return self._request("GET", f"/chat/{thread_id}")

    # ---- 论文库 ----

    def health(self) -> dict:
        return self._request("GET", "/health")

    def stats(self) -> dict:
        return self._request("GET", "/stats")

    def papers(self) -> list[dict]:
        return self._request("GET", "/papers")

    def paper(self, paper_id: str) -> dict:
        return self._request("GET", f"/papers/{paper_id}")

    def delete_paper(self, paper_id: str) -> dict:
        return self._request("DELETE", f"/papers/{paper_id}")

    def card(self, paper_id: str, *, refresh: bool = False) -> dict:
        return self._request("GET", f"/papers/{paper_id}/card", params={"refresh": refresh})

    def compare(self, paper_ids: list[str], focus: str = "") -> dict:
        return self._request("POST", "/compare", json={"paper_ids": paper_ids, "focus": focus})

    def search(self, query: str, *, paper_ids: list[str] | None = None) -> dict:
        return self._request("POST", "/search", json={"query": query, "paper_ids": paper_ids})

    def arxiv_search(self, q: str, limit: int = 10) -> list[dict]:
        return self._request("GET", "/arxiv/search", params={"q": q, "limit": limit})

    def ingest(self, ids: list[str], *, force: bool = False) -> dict:
        return self._request("POST", "/papers/ingest", json={"ids": ids, "force": force})

    def upload(self, filename: str, data: bytes) -> dict:
        files = {"file": (filename, data, "application/pdf")}
        return self._request("POST", "/papers/upload", files=files)

    def jobs(self) -> list[dict]:
        return self._request("GET", "/jobs")
