"""REST API：FastAPI + SSE 流式问答。

进程模型：Milvus Lite 同一个库文件只能被一个进程打开，所以 API 是唯一持有存储的进程，
Streamlit 界面通过 HTTP 调用它。

- 问答：POST /chat 返回 SSE 事件流（status / token / citations / card / compare / message /
  interrupt / done）。Agent 在后台线程里跑，客户端断开也会跑完，保证会话检查点完整；
  同一个会话同时只允许一个请求。
- 入库：放进单线程任务队列串行执行（arXiv 要求请求间隔 3 秒，Milvus 写入也不必并发），
  前端轮询 GET /jobs/{id} 查看进度。
- 鉴权：默认只监听 127.0.0.1、不鉴权；配置了 API_KEY 时所有接口都要带 X-API-Key 请求头。
"""

import json
import queue
import re
import secrets
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse, ServerSentEvent

from paperlens import __version__
from paperlens.config import Settings, get_settings
from paperlens.ingest.arxiv_client import ArxivClient
from paperlens.ingest.pipeline import IngestResult, ingest_arxiv, ingest_pdf, remove_paper
from paperlens.rag.card import CardError, anchor_url, card_markdown, get_card
from paperlens.rag.compare import compare_markdown, compare_papers
from paperlens.rag.query import analyze_query
from paperlens.rag.retriever import Retriever
from paperlens.services import Services
from paperlens.store import Hit

_DONE = object()
_MAX_JOBS = 50  # 内存里保留最近多少个入库任务


# ---------------------------------------------------------------- 请求体


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=64, pattern=r"^[\w-]+$")


class ResumeRequest(BaseModel):
    picked: list[str] = Field(default_factory=list, max_length=10)


class IngestRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=30)
    force: bool = False


class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)
    focus: str = Field(default="", max_length=500)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    english_query: str | None = Field(default=None, max_length=2000)
    paper_ids: list[str] | None = None
    explain: bool = True


# ---------------------------------------------------------------- 入库任务


@dataclass
class Job:
    job_id: str
    kind: Literal["arxiv", "pdf"]
    description: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    results: list[dict] = field(default_factory=list)
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class JobQueue:
    """单线程执行的入库任务队列，任务状态保存在内存里（服务重启后清空）。"""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, kind, description: str, fn: Callable[[Job], None]) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, description=description)
        with self._lock:
            self._jobs[job.job_id] = job
            while len(self._jobs) > _MAX_JOBS:
                self._jobs.popitem(last=False)

        def run() -> None:
            job.status = "running"
            try:
                fn(job)
                job.status = "done"
            except Exception as exc:
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"

        self._executor.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------- 工具函数


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _hit(h: Hit) -> dict:
    c = h.chunk
    return {
        "score": round(h.score, 4),
        "paper_id": c.paper_id,
        "title": c.title,
        "section": c.section,
        "anchor": c.anchor,
        "url": c.url,
        "content": c.content,
    }


def _paper_dict(record) -> dict:
    m = record.meta
    return {
        "paper_id": m.paper_id,
        "version": m.version,
        "title": m.title,
        "authors": m.authors,
        "abstract": m.abstract,
        "published": m.published,
        "year": m.year,
        "source": record.source,
        "n_chunks": record.n_chunks,
        "added_at": record.added_at,
        "url": m.abs_url,
    }


def _safe_filename(name: str) -> str:
    """只保留文件名本身（去掉客户端传来的路径），非常规字符换成下划线。"""
    base = re.split(r"[\\/]", name or "")[-1]
    stem = re.sub(r"[^\w.\-]+", "_", base).lstrip(".")[-80:] or "upload.pdf"
    return stem if stem.lower().endswith(".pdf") else f"{stem}.pdf"


# ---------------------------------------------------------------- 应用


def create_app(
    settings: Settings | None = None,
    *,
    services: Services | None = None,
    agent_factory: Callable[[Services], Any] | None = None,
) -> FastAPI:
    """services / agent_factory 供测试注入替身；正常启动时按配置创建。"""
    settings = settings or (services.settings if services else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own = services is None
        svc = services or Services(settings)
        if own:
            # 启动时创建好所有客户端：cached_property 不是线程安全的，避免多个请求同时初始化
            _ = svc.db, svc.vectors, svc.llm, svc.sf
        if agent_factory:
            agent = agent_factory(svc)
        else:
            from paperlens.agent.graph import PaperAgent

            agent = PaperAgent(svc)
        app.state.services = svc
        app.state.agent = agent
        app.state.jobs = JobQueue()
        app.state.thread_locks = {}
        app.state.thread_locks_guard = threading.Lock()
        try:
            yield
        finally:
            app.state.jobs.shutdown()
            if hasattr(agent, "close"):
                agent.close()
            if own:
                svc.close()

    app = FastAPI(title="PaperLens API", version=__version__, lifespan=lifespan)

    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def check_key(key: str | None = Depends(header)) -> None:
        expected = settings.api_key
        if expected is None:
            return
        if not key or not secrets.compare_digest(key, expected.get_secret_value()):
            raise HTTPException(401, "缺少或错误的 X-API-Key")

    auth = [Depends(check_key)]

    def svc_of(request: Request) -> Services:
        return request.app.state.services

    # ------------------------------------------------------------ 基础

    @app.get("/health")
    def health(request: Request) -> dict:
        svc = svc_of(request)
        return {
            "status": "ok",
            "version": __version__,
            "papers": len(svc.db.list_papers()),
            "model": settings.llm_model,
            "auth": settings.api_key is not None,
        }

    @app.get("/stats", dependencies=auth)
    def stats(request: Request) -> dict:
        tracker = svc_of(request).tracker
        return {
            "total_cost_usd": tracker.total_cost_usd,
            "models": {k: asdict(v) for k, v in tracker.snapshot().items()},
        }

    # ------------------------------------------------------------ 问答（SSE）

    def _thread_lock(request: Request, thread_id: str) -> threading.Lock:
        state = request.app.state
        with state.thread_locks_guard:
            return state.thread_locks.setdefault(thread_id, threading.Lock())

    def _sse(request: Request, thread_id: str, run: Callable[[], Iterator[dict]]):
        lock = _thread_lock(request, thread_id)
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "这个会话还有一个回答没结束，请稍后再试")
        events: queue.Queue = queue.Queue()

        def worker() -> None:
            interrupted = False
            try:
                for event in run():
                    interrupted = interrupted or event["type"] == "interrupt"
                    events.put(_jsonable(event))
            except Exception as exc:
                events.put({"type": "message", "text": f"出错了：{type(exc).__name__}: {exc}"})
            finally:
                events.put({"type": "done", "thread_id": thread_id, "interrupted": interrupted})
                events.put(_DONE)
                lock.release()

        threading.Thread(target=worker, daemon=True, name=f"chat-{thread_id}").start()

        def stream() -> Iterator[ServerSentEvent]:
            while (event := events.get()) is not _DONE:
                yield ServerSentEvent(
                    data=json.dumps(event, ensure_ascii=False), event=event["type"]
                )

        return EventSourceResponse(stream(), ping=15)

    @app.post("/chat", dependencies=auth)
    def chat(body: ChatRequest, request: Request):
        thread_id = body.thread_id or uuid.uuid4().hex[:8]
        agent = request.app.state.agent
        return _sse(request, thread_id, lambda: agent.stream(thread_id, body.question))

    @app.post("/chat/{thread_id}/resume", dependencies=auth)
    def resume(thread_id: str, body: ResumeRequest, request: Request):
        agent = request.app.state.agent
        if agent.pending_interrupt(thread_id) is None:
            raise HTTPException(409, "这个会话没有等待确认的操作")
        return _sse(request, thread_id, lambda: agent.resume(thread_id, body.picked))

    @app.get("/chat/{thread_id}", dependencies=auth)
    def chat_history(thread_id: str, request: Request) -> dict:
        agent = request.app.state.agent
        return {
            "thread_id": thread_id,
            "messages": agent.history(thread_id),
            "pending": agent.pending_interrupt(thread_id),
        }

    # ------------------------------------------------------------ 检索调试

    @app.post("/search", dependencies=auth)
    def search(body: SearchRequest, request: Request) -> dict:
        svc = svc_of(request)
        english = body.english_query
        if english is None:
            english = analyze_query(svc.llm, body.query).english_query
        r = Retriever(svc).retrieve(
            body.query, english_query=english, paper_ids=body.paper_ids, explain=body.explain
        )
        return {
            "query": r.query,
            "bm25_query": r.bm25_query,
            "threshold": settings.evidence_threshold,
            "top_score": r.top_score,
            "dense": [_hit(h) for h in r.dense],
            "bm25": [_hit(h) for h in r.bm25],
            "fused": [_hit(h) for h in r.fused],
            "reranked": [_hit(Hit(e.chunk, e.score)) for e in r.evidence],
        }

    # ------------------------------------------------------------ arXiv 与入库

    @app.get("/arxiv/search", dependencies=auth)
    def arxiv_search(request: Request, q: str, limit: int = 10) -> list[dict]:
        """在 arXiv 上按关键词找论文（不入库），标出哪些已经在库里。"""
        svc = svc_of(request)
        if not q.strip():
            raise HTTPException(422, "关键词不能为空")
        limit = min(max(limit, 1), 30)
        with ArxivClient(settings) as arxiv:
            # 先按整句短语匹配；实测用户随手输入的多个词很少作为连续短语出现，没结果时拆成单词检索
            found = arxiv.search_keywords([q.strip()], max_results=limit)
            if not found and len(q.split()) > 1:
                found = arxiv.search_keywords(q.split(), max_results=limit)
        return [
            {
                "paper_id": m.paper_id,
                "version": m.version,
                "title": m.title,
                "authors": m.authors[:5],
                "published": m.published,
                "abstract": m.abstract,
                "in_library": svc.db.get_paper(m.paper_id) is not None,
            }
            for m in found
        ]

    def _result_dict(r: IngestResult) -> dict:
        return asdict(r)

    @app.post("/papers/ingest", dependencies=auth, status_code=202)
    def ingest(body: IngestRequest, request: Request) -> dict:
        svc = svc_of(request)

        def run(job: Job) -> None:
            ingest_arxiv(
                svc,
                body.ids,
                force=body.force,
                on_result=lambda r: job.results.append(_result_dict(r)),
            )

        job = request.app.state.jobs.submit("arxiv", ", ".join(body.ids), run)
        return asdict(job)

    @app.post("/papers/upload", dependencies=auth, status_code=202)
    async def upload(request: Request, file: UploadFile = File(...)) -> dict:
        svc = svc_of(request)
        limit = settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"文件超过 {settings.max_upload_mb} MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "只支持 PDF 文件")
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        path = settings.upload_dir / f"{uuid.uuid4().hex[:8]}-{_safe_filename(file.filename)}"
        path.write_bytes(data)

        def run(job: Job) -> None:
            job.results.append(_result_dict(ingest_pdf(svc, path)))

        job = request.app.state.jobs.submit("pdf", file.filename or path.name, run)
        return asdict(job)

    @app.get("/jobs", dependencies=auth)
    def jobs(request: Request) -> list[dict]:
        return [asdict(j) for j in request.app.state.jobs.list()]

    @app.get("/jobs/{job_id}", dependencies=auth)
    def job(job_id: str, request: Request) -> dict:
        found = request.app.state.jobs.get(job_id)
        if found is None:
            raise HTTPException(404, "没有这个任务")
        return asdict(found)

    # ------------------------------------------------------------ 论文、卡片、对比

    @app.get("/papers", dependencies=auth)
    def papers(request: Request) -> list[dict]:
        return [_paper_dict(r) for r in svc_of(request).db.list_papers()]

    # 老式 arXiv ID 带斜杠（如 hep-th/9901001），所以用 path 类型；/card 路由要写在前面
    @app.get("/papers/{paper_id:path}/card", dependencies=auth)
    def paper_card(paper_id: str, request: Request, refresh: bool = False) -> dict:
        svc = svc_of(request)
        record = svc.db.get_paper(paper_id)
        if record is None:
            raise HTTPException(404, f"知识库里没有论文 {paper_id}")
        try:
            card = get_card(svc, paper_id, refresh=refresh)
        except CardError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "paper_id": paper_id,
            "title": record.meta.title,
            "card": card.model_dump(),
            "markdown": card_markdown(
                card, title=record.meta.title, url_of=lambda a: anchor_url(record, a)
            ),
        }

    @app.get("/papers/{paper_id:path}", dependencies=auth)
    def paper(paper_id: str, request: Request) -> dict:
        svc = svc_of(request)
        record = svc.db.get_paper(paper_id)
        if record is None:
            raise HTTPException(404, f"知识库里没有论文 {paper_id}")
        return {**_paper_dict(record), "has_card": svc.db.get_card(paper_id) is not None}

    @app.delete("/papers/{paper_id:path}", dependencies=auth)
    def delete(paper_id: str, request: Request) -> dict:
        svc = svc_of(request)
        if svc.db.get_paper(paper_id) is None:
            raise HTTPException(404, f"知识库里没有论文 {paper_id}")
        remove_paper(svc, paper_id)
        return {"deleted": paper_id}

    @app.post("/compare", dependencies=auth)
    def compare(body: CompareRequest, request: Request) -> dict:
        try:
            result = compare_papers(svc_of(request), body.paper_ids, focus=body.focus)
        except CardError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {**result.model_dump(), "markdown": compare_markdown(result)}

    return app
