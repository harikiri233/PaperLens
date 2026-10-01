"""arXiv 客户端：元数据查询、关键词检索、HTML / PDF 下载。

- 所有请求串行，间隔不少于 settings.arxiv_delay_s（arXiv 要求 3 秒）。
- 下载的 HTML / PDF 缓存在 data/raw/，重复入库不再访问网络。
"""

import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from paperlens.config import Settings
from paperlens.ingest.schema import PaperMeta

API_URL = "https://export.arxiv.org/api/query"
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# 新式 ID（2310.11511v2）与旧式 ID（cs/0112017v1）
_ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)


class ArxivError(RuntimeError):
    pass


def parse_arxiv_id(text: str) -> tuple[str, str]:
    """从 ID 或链接中取出 (基础 ID, 版本号)，如 "https://arxiv.org/abs/2310.11511v2" → ("2310.11511", "v2")。"""
    match = _ID_RE.search(text.strip())
    if not match:
        raise ArxivError(f"无法识别的 arXiv ID：{text!r}")
    return match.group(1), (match.group(2) or "")


def build_query(keywords: list[str], *, op: str = "AND") -> str:
    """关键词 → arXiv 查询语法；短语加引号，去掉会破坏语法的字符。"""
    terms = []
    for k in keywords:
        k = re.sub(r'[()":]', " ", k).strip()
        if k:
            terms.append(f'all:"{k}"' if " " in k else f"all:{k}")
    return f" {op} ".join(terms)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_feed(xml_text: str) -> list[PaperMeta]:
    """解析 arXiv API 返回的 Atom feed。"""
    root = ET.fromstring(xml_text)
    papers: list[PaperMeta] = []
    for entry in root.findall("a:entry", _ATOM):
        raw_id = entry.findtext("a:id", default="", namespaces=_ATOM)
        if "/abs/" not in raw_id:  # 查询出错时 arXiv 会返回一个说明错误的 entry
            continue
        paper_id, version = parse_arxiv_id(raw_id.split("/abs/", 1)[1])
        papers.append(
            PaperMeta(
                paper_id=paper_id,
                version=version,
                title=_clean(entry.findtext("a:title", namespaces=_ATOM)),
                authors=[
                    _clean(a.findtext("a:name", namespaces=_ATOM))
                    for a in entry.findall("a:author", _ATOM)
                ],
                abstract=_clean(entry.findtext("a:summary", namespaces=_ATOM)),
                categories=[c.get("term", "") for c in entry.findall("a:category", _ATOM)],
                published=(entry.findtext("a:published", default="", namespaces=_ATOM))[:10],
            )
        )
    return papers


class ArxivClient:
    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        """transport 参数用于测试时注入 httpx.MockTransport。"""
        self.settings = settings
        self.cache_dir = settings.raw_dir
        self._http = httpx.Client(
            timeout=60,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": "PaperLens/0.1 (personal research assistant)"},
        )
        self._lock = threading.Lock()
        self._last_request = 0.0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ArxivClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=3, min=3, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        with self._lock:
            wait = self._last_request + self.settings.arxiv_delay_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                response = self._http.get(url, params=params)
            finally:
                self._last_request = time.monotonic()
        if response.status_code != 404:
            response.raise_for_status()
        return response

    # ---- 元数据 ----

    def fetch_meta(self, ids: list[str]) -> list[PaperMeta]:
        """按 ID 批量查询元数据，返回顺序与输入一致；查不到的 ID 抛错。"""
        base_ids = [parse_arxiv_id(i)[0] for i in ids]
        found: dict[str, PaperMeta] = {}
        for start in range(0, len(base_ids), 50):
            batch = base_ids[start : start + 50]
            response = self._get(API_URL, {"id_list": ",".join(batch), "max_results": len(batch)})
            for meta in parse_feed(response.text):
                found[meta.paper_id] = meta
        missing = [i for i in base_ids if i not in found]
        if missing:
            raise ArxivError(f"arXiv 上找不到这些论文：{', '.join(missing)}")
        return [found[i] for i in base_ids]

    def search(self, query: str, *, max_results: int = 10) -> list[PaperMeta]:
        """关键词检索。query 可以是 arXiv 查询语法（含 ti:/abs:/all: 等前缀），也可以是普通关键词。"""
        if not re.search(r"\b(ti|abs|au|cat|all|co|jr|rn|id):", query):
            query = build_query(re.split(r"\s+", query.strip()))
        response = self._get(
            API_URL,
            {"search_query": query, "max_results": max_results, "sortBy": "relevance"},
        )
        return parse_feed(response.text)

    def search_keywords(self, keywords: list[str], *, max_results: int = 5) -> list[PaperMeta]:
        """按关键词检索：先要求全部命中（AND），没有结果时放宽为任一命中（OR）。

        实测：把整句问题拆成词做 AND 往往一篇也搜不到，所以关键词要由 LLM 先提炼成 2–4 个主题词。
        """
        keywords = [k.strip() for k in keywords if k.strip()]
        if not keywords:
            return []
        found = self.search(build_query(keywords), max_results=max_results)
        if not found and len(keywords) > 1:
            found = self.search(build_query(keywords, op="OR"), max_results=max_results)
        return found

    # ---- 全文 ----

    def _cached(self, path: Path, url: str, *, binary: bool) -> bytes | None:
        if path.exists():
            return path.read_bytes()
        response = self._get(url)
        if response.status_code == 404:
            return None
        content = response.content
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return content

    def fetch_html(self, paper_id: str, version: str = "") -> str | None:
        """下载 arXiv HTML 版；论文没有 HTML 版时返回 None。"""
        name = paper_id.replace("/", "_") + version
        content = self._cached(
            self.cache_dir / f"{name}.html",
            f"https://arxiv.org/html/{paper_id}{version}",
            binary=False,
        )
        if content is None:
            return None
        text = content.decode("utf-8", errors="replace")
        # 没有 HTML 版时，arXiv 有时返回 200 的说明页而不是 404
        return text if "ltx_page_main" in text or "ltx_document" in text else None

    def fetch_pdf(self, paper_id: str, version: str = "") -> Path:
        name = paper_id.replace("/", "_") + version
        path = self.cache_dir / f"{name}.pdf"
        content = self._cached(path, f"https://arxiv.org/pdf/{paper_id}{version}", binary=True)
        if content is None:
            raise ArxivError(f"下载 PDF 失败：{paper_id}{version}")
        if not content.startswith(b"%PDF"):
            path.unlink(missing_ok=True)
            raise ArxivError(f"{paper_id}{version} 返回的内容不是 PDF")
        return path
