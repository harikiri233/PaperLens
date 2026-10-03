"""REST API 测试：FastAPI TestClient + 替身 Agent / 模型 / 向量库，不访问网络。"""

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import paperlens.api.app as api_module
from paperlens.api.app import create_app
from paperlens.api.client import PaperLensClient, iter_sse
from paperlens.ingest.pipeline import IngestResult, parsed_path
from paperlens.ingest.schema import Block, PaperMeta, ParsedPaper, Section
from paperlens.models import UsageTracker
from paperlens.rag.card import CardPoint, PaperCard
from paperlens.store import Database


class FakeAgent:
    """第一轮问题触发 interrupt，resume 后回答。"""

    def __init__(self):
        self.pending: dict[str, dict] = {}
        self.closed = False

    def stream(self, thread_id, question):
        yield {"type": "status", "text": "分析问题"}
        if "补充" in question:
            payload = {"message": "选择要入库的论文", "candidates": [{"paper_id": "new"}]}
            self.pending[thread_id] = payload
            yield {"type": "interrupt", "payload": payload}
            return
        yield {"type": "token", "text": "答案"}
        yield {"type": "token", "text": " [1]"}
        yield {"type": "citations", "citations": [], "invalid": []}

    def resume(self, thread_id, picked):
        self.pending.pop(thread_id, None)
        yield {"type": "message", "text": f"已入库 {picked}"}

    def pending_interrupt(self, thread_id):
        return self.pending.get(thread_id)

    def history(self, thread_id):
        return [{"role": "user", "content": "q"}]

    def close(self):
        self.closed = True


class FakeVectors:
    def __init__(self):
        self.deleted = []

    def delete_paper(self, paper_id):
        self.deleted.append(paper_id)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, messages, schema, **kwargs):
        self.calls += 1
        return PaperCard(tldr="卡片", method=[CardPoint(text="方法", anchor="S1")])


@pytest.fixture
def env(settings):
    db = Database(settings.db_path)
    services = SimpleNamespace(
        settings=settings,
        db=db,
        vectors=FakeVectors(),
        llm=FakeLLM(),
        tracker=UsageTracker(),
    )
    paper = ParsedPaper(
        meta=PaperMeta(paper_id="2310.11511", version="v1", title="Self-RAG: Learning"),
        source="html",
        sections=[Section(anchor="S1", path=["1 Intro"], blocks=[Block(anchor="S1.p1", text="x")])],
    )
    db.upsert_paper(paper.meta, "html", 1)
    path = parsed_path(services, "2310.11511")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paper.model_dump_json(), encoding="utf-8")
    agent = FakeAgent()
    app = create_app(settings, services=services, agent_factory=lambda svc: agent)
    with TestClient(app) as http:
        yield SimpleNamespace(http=http, services=services, agent=agent, settings=settings)
    db.close()


def _events(response) -> list[dict]:
    return list(iter_sse(response.iter_lines()))


def test_health_and_papers(env):
    assert env.http.get("/health").json()["papers"] == 1
    papers = env.http.get("/papers").json()
    assert papers[0]["paper_id"] == "2310.11511"
    assert papers[0]["url"] == "https://arxiv.org/abs/2310.11511v1"
    assert env.http.get("/papers/2310.11511").json()["has_card"] is False
    assert env.http.get("/papers/nope").status_code == 404


def test_chat_streams_sse_events(env):
    with env.http.stream("POST", "/chat", json={"question": "问题", "thread_id": "t1"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _events(r)
    kinds = [e["type"] for e in events]
    assert kinds == ["status", "token", "token", "citations", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "答案 [1]"
    assert events[-1] == {"type": "done", "thread_id": "t1", "interrupted": False}


def test_chat_interrupt_and_resume(env):
    with env.http.stream("POST", "/chat", json={"question": "需要补充", "thread_id": "t2"}) as r:
        events = _events(r)
    assert events[-2]["type"] == "interrupt"
    assert events[-1]["interrupted"] is True
    assert env.http.get("/chat/t2").json()["pending"]["message"] == "选择要入库的论文"

    with env.http.stream("POST", "/chat/t2/resume", json={"picked": ["new"]}) as r:
        events = _events(r)
    assert events[0] == {"type": "message", "text": "已入库 ['new']"}
    # 没有待确认的操作时不能 resume
    assert env.http.post("/chat/t2/resume", json={"picked": []}).status_code == 409


def test_chat_rejects_bad_input(env):
    assert env.http.post("/chat", json={"question": ""}).status_code == 422
    bad = {"question": "q", "thread_id": "../etc"}
    assert env.http.post("/chat", json=bad).status_code == 422


def test_card_is_cached(env):
    first = env.http.get("/papers/2310.11511/card").json()
    assert first["card"]["tldr"] == "卡片"
    assert "[§S1](https://arxiv.org/html/2310.11511v1#S1)" in first["markdown"]
    env.http.get("/papers/2310.11511/card")
    assert env.services.llm.calls == 1
    env.http.get("/papers/2310.11511/card", params={"refresh": True})
    assert env.services.llm.calls == 2
    assert env.http.get("/papers/nope/card").status_code == 404


def test_compare_validates_count(env):
    assert env.http.post("/compare", json={"paper_ids": ["2310.11511"]}).status_code == 422
    r = env.http.post("/compare", json={"paper_ids": ["2310.11511", "missing"]})
    assert r.status_code == 400
    assert "missing" in r.json()["detail"]


def _wait(http, job_id: str) -> dict:
    for _ in range(100):
        job = http.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("任务没有结束")


def test_ingest_runs_in_background(env, monkeypatch):
    def fake_ingest(services, ids, force=False, on_result=None, **kwargs):
        for i in ids:
            on_result(IngestResult(i, "T", "added", "html", 3))

    monkeypatch.setattr(api_module, "ingest_arxiv", fake_ingest)
    r = env.http.post("/papers/ingest", json={"ids": ["2401.15884", "2305.06983"]})
    assert r.status_code == 202
    job = _wait(env.http, r.json()["job_id"])
    assert job["status"] == "done"
    assert [x["paper_id"] for x in job["results"]] == ["2401.15884", "2305.06983"]
    assert env.http.get("/jobs").json()[0]["job_id"] == job["job_id"]


def test_upload_checks_pdf_and_size(env, monkeypatch):
    seen = []

    def fake_pdf(services, path, title=""):
        seen.append(path)
        return IngestResult("local-abc", "Uploaded", "added", "pdf", 2)

    monkeypatch.setattr(api_module, "ingest_pdf", fake_pdf)
    bad = env.http.post("/papers/upload", files={"file": ("a.pdf", b"hello", "application/pdf")})
    assert bad.status_code == 415

    env.settings.max_upload_mb = 0  # 任何非空文件都超限
    big = env.http.post("/papers/upload", files={"file": ("a.pdf", b"%PDF-1", "application/pdf")})
    assert big.status_code == 413
    env.settings.max_upload_mb = 50

    files = {"file": ("../../evil name.pdf", b"%PDF-1.7 ...", "application/pdf")}
    r = env.http.post("/papers/upload", files=files)
    assert r.status_code == 202
    job = _wait(env.http, r.json()["job_id"])
    assert job["results"][0]["paper_id"] == "local-abc"
    # 文件名里的路径分隔符被清理，只会写进 uploads 目录
    assert seen[0].parent == env.settings.upload_dir
    assert seen[0].name.endswith("-evil_name.pdf")


def test_delete_paper(env):
    assert env.http.delete("/papers/2310.11511").json() == {"deleted": "2310.11511"}
    assert env.services.vectors.deleted == ["2310.11511"]
    assert env.http.get("/papers").json() == []
    assert env.http.delete("/papers/2310.11511").status_code == 404


def test_api_key_required_when_configured(env):
    env.settings.api_key = SecretStr("secret-key-123")
    try:
        assert env.http.get("/papers").status_code == 401
        assert env.http.get("/papers", headers={"X-API-Key": "wrong"}).status_code == 401
        ok = env.http.get("/papers", headers={"X-API-Key": "secret-key-123"})
        assert ok.status_code == 200
        assert env.http.get("/health").status_code == 200  # 健康检查不需要 Key
    finally:
        env.settings.api_key = None


def test_python_client_against_app(env):
    client = PaperLensClient(http=env.http)
    assert client.papers()[0]["title"] == "Self-RAG: Learning"
    events = list(client.chat("问题", "t3"))
    assert events[-1]["type"] == "done"
    with pytest.raises(Exception, match="404"):
        client.paper("nope")


def test_iter_sse_parses_multiline_and_ignores_comments():
    lines = [": ping", "", "event: token", 'data: {"type": "token",', 'data: "text": "a"}', ""]
    assert list(iter_sse(iter(lines))) == [{"type": "token", "text": "a"}]


def test_safe_filename():
    assert api_module._safe_filename("..\\..\\a b.PDF") == "a_b.PDF"
    assert api_module._safe_filename("") == "upload.pdf"
    assert api_module._safe_filename("notes") == "notes.pdf"


def test_jsonable_handles_dataclasses():
    from paperlens.rag.answer import Citation

    cite = Citation(1, "p", "t", "s", "a", "u", "snip")
    assert api_module._jsonable({"citations": [cite]})["citations"][0]["n"] == 1
