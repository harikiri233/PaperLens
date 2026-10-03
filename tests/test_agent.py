"""Agent 路由测试：模型、检索、arXiv 和入库全部用替身，只验证状态图的走向。"""

from types import SimpleNamespace

import pytest

import paperlens.agent.graph as graph_module
from paperlens.agent.graph import PaperAgent
from paperlens.ingest.pipeline import IngestResult
from paperlens.ingest.schema import Chunk, PaperMeta
from paperlens.models.siliconflow import RerankResult
from paperlens.rag.query import QueryAnalysis
from paperlens.rag.retriever import RetrievalResult, ScoredChunk
from paperlens.store import Database


class FakeLLM:
    def __init__(self, analyses: list[QueryAnalysis]):
        self.analyses = list(analyses)
        self.histories: list[str] = []

    def complete_json(self, messages, schema, **kwargs):
        self.histories.append(messages[-1]["content"])
        return self.analyses.pop(0)

    def stream(self, messages, **kwargs):
        yield from ["答案 [1]", "，越界 [9]。"]

    def chat(self, messages, **kwargs):
        return SimpleNamespace(content="你好，我是 PaperLens。")


class FakeSF:
    def rerank(self, query, documents, top_n=None):
        # 标题里带 "Relevant" 的候选论文得高分
        scores = [0.9 if "Relevant" in d else 0.001 for d in documents]
        ranked = [RerankResult(i, s) for i, s in enumerate(scores)]
        return sorted(ranked, key=lambda r: r.score, reverse=True)


class FakeRetriever:
    """库里有 good 论文时给高分证据；否则低分。"""

    def __init__(self, db: Database):
        self.db = db
        self.calls: list[list[str] | None] = []

    def retrieve(self, query, *, english_query="", paper_ids=None, top_k=None):
        self.calls.append(paper_ids)
        has_good = self.db.get_paper("good") is not None
        chunk = Chunk(
            chunk_id="good#0" if has_good else "other#0",
            paper_id="good" if has_good else "other",
            index=0,
            kind="text",
            title="Good Paper" if has_good else "Other Paper",
            section="1 Intro",
            anchor="S1.p1",
            url="https://arxiv.org/html/good#S1.p1",
            content="text",
        )
        score = 0.95 if has_good else 0.05
        return RetrievalResult(
            query=query, bm25_query=english_query, evidence=[ScoredChunk(chunk, score)]
        )


class FakeArxiv:
    def __init__(self, settings):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def search_keywords(self, keywords, max_results=5):
        return [
            PaperMeta(paper_id="noise", version="v1", title="Stellar Flares"),
            PaperMeta(
                paper_id="good", version="v1", title="Relevant Paper", published="2023-05-01"
            ),
        ]


@pytest.fixture
def make_agent(settings, monkeypatch):
    agents = []

    def factory(analyses: list[QueryAnalysis], *, online="ask"):
        settings.online_fallback = online
        db = Database(settings.db_path)
        db.upsert_paper(PaperMeta(paper_id="other", title="Other Paper"), "html", 1)
        services = SimpleNamespace(settings=settings, db=db, llm=FakeLLM(analyses), sf=FakeSF())
        ingested: list[list[str]] = []

        def fake_ingest(services_, targets, on_result=None, **kwargs):
            ingested.append(targets)
            results = []
            for t in targets:
                pid = t.removesuffix("v1")
                db.upsert_paper(PaperMeta(paper_id=pid, title="Good Paper"), "html", 3)
                r = IngestResult(pid, "Good Paper", "added", "html", 3)
                results.append(r)
                if on_result:
                    on_result(r)
            return results

        monkeypatch.setattr(graph_module, "ArxivClient", FakeArxiv)
        monkeypatch.setattr(graph_module, "ingest_arxiv", fake_ingest)
        agent = PaperAgent(services)
        agent.retriever = FakeRetriever(db)
        agent.ingested = ingested
        agents.append((agent, db))
        return agent

    yield factory
    for agent, db in agents:
        agent.close()
        db.close()


def _collect(events):
    out = {"status": [], "tokens": "", "messages": [], "citations": None, "interrupt": None}
    for e in events:
        if e["type"] == "status":
            out["status"].append(e["text"])
        elif e["type"] == "token":
            out["tokens"] += e["text"]
        elif e["type"] == "message":
            out["messages"].append(e["text"])
        elif e["type"] == "citations":
            out["citations"] = e
        elif e["type"] == "interrupt":
            out["interrupt"] = e["payload"]
    return out


def _qa(question: str, mentions: list[str] | None = None, intent="qa") -> QueryAnalysis:
    return QueryAnalysis(
        intent=intent,
        standalone_question=question,
        english_query=question,
        paper_mentions=mentions or [],
        search_keywords=["good"],
    )


def test_chat_intent(make_agent):
    agent = make_agent([_qa("你好", intent="chat")])
    out = _collect(agent.stream("t", "你好"))
    assert out["messages"] == ["你好，我是 PaperLens。"]
    assert agent.retriever.calls == []


def test_low_evidence_asks_then_ingests_and_answers(make_agent):
    agent = make_agent([_qa("Good 论文讲了什么？")])
    out = _collect(agent.stream("t", "Good 论文讲了什么？"))
    payload = out["interrupt"]
    assert payload is not None
    # reranker 过滤掉了无关的候选论文
    assert [c["paper_id"] for c in payload["candidates"]] == ["good"]
    assert agent.pending_interrupt("t") is not None

    out = _collect(agent.resume("t", ["good"]))
    assert agent.ingested == [["goodv1"]]
    assert out["tokens"] == "答案 [1]，越界 [9]。"
    cites = out["citations"]
    assert [c.paper_id for c in cites["citations"]] == ["good"]
    assert cites["invalid"] == [9]
    history = agent.history("t")
    assert history[-1]["content"] == "答案 [1]，越界 。"  # 越界编号已去掉
    assert agent.pending_interrupt("t") is None


def test_user_declines_supplement(make_agent):
    agent = make_agent([_qa("冷门问题")])
    assert _collect(agent.stream("t", "冷门问题"))["interrupt"] is not None
    out = _collect(agent.resume("t", []))
    assert agent.ingested == []
    assert "没有找到足够相关的内容" in out["messages"][0]
    assert "Other Paper" in out["messages"][0]


def test_online_fallback_off_refuses_directly(make_agent):
    agent = make_agent([_qa("冷门问题")], online="off")
    out = _collect(agent.stream("t", "冷门问题"))
    assert out["interrupt"] is None
    assert "联网补充已关闭" in out["messages"][0]


def test_mentioned_paper_filters_retrieval(make_agent):
    agent = make_agent([_qa("Other Paper 的方法？", mentions=["Other Paper"])])
    agent.retriever.db.upsert_paper(PaperMeta(paper_id="good", title="Good Paper"), "html", 3)
    out = _collect(agent.stream("t", "Other Paper 的方法？"))
    assert agent.retriever.calls == [["other"]]
    assert out["tokens"]


def test_missing_mentioned_paper_triggers_search(make_agent):
    agent = make_agent([_qa("Good 的方法？", mentions=["Good"])])
    # 库里没有 Good 这篇，即使全库检索分数不低也应该去 arXiv 找
    agent.retriever.db.upsert_paper(PaperMeta(paper_id="x", title="Unrelated"), "html", 1)
    out = _collect(agent.stream("t", "Good 的方法？"))
    assert out["interrupt"] is not None
    out = _collect(agent.resume("t", ["good"]))
    assert agent.retriever.calls[-1] == ["good"]  # 新入库的论文并入检索范围
    assert out["tokens"]


def test_discover_ingests_without_answering(make_agent):
    agent = make_agent([_qa("找几篇 Good 相关论文", intent="discover")])
    out = _collect(agent.stream("t", "找几篇 Good 相关论文"))
    assert out["interrupt"]["message"].startswith("在 arXiv 上找到")
    out = _collect(agent.resume("t", ["good"]))
    assert "已入库 1 篇论文" in out["messages"][0]
    assert agent.retriever.calls == []


def test_history_is_passed_to_next_turn(make_agent):
    agent = make_agent([_qa("你好", intent="chat"), _qa("第二个问题", intent="chat")])
    _collect(agent.stream("t", "你好"))
    _collect(agent.stream("t", "它呢？"))
    assert "你好，我是 PaperLens。" in agent.services.llm.histories[-1]
    assert len(agent.history("t")) == 4


def test_explain_mentioned_paper_routes_to_card(make_agent, monkeypatch):
    from paperlens.rag.card import CardPoint, PaperCard

    calls = []

    def fake_card(services, paper_id, refresh=False):
        calls.append(paper_id)
        return PaperCard(tldr="一句话总结", method=[CardPoint(text="方法", anchor="S3")])

    monkeypatch.setattr(graph_module, "get_card", fake_card)
    agent = make_agent([_qa("讲讲 Other Paper", mentions=["Other Paper"], intent="explain")])
    out = _collect(agent.stream("t", "讲讲 Other Paper"))
    assert calls == ["other"]
    assert agent.retriever.calls == []  # 不走检索
    assert "一句话总结" in out["messages"][0]
    assert agent.history("t")[-1]["content"].startswith("### Other Paper")


def test_explain_without_mention_falls_back_to_rag(make_agent):
    agent = make_agent([_qa("讲讲这个方向", intent="explain")])
    _collect(agent.stream("t", "讲讲这个方向"))
    assert agent.retriever.calls == [None]


def test_compare_two_papers_routes_to_compare(make_agent, monkeypatch):
    from paperlens.rag.compare import CompareCell, ComparedPaper, CompareResult, CompareRow

    seen = []

    def fake_compare(services, ids, focus=""):
        seen.append(ids)
        papers = [ComparedPaper(paper_id=i, title=i, short_title=i) for i in ids]
        row = CompareRow(dimension="方法", cells=[CompareCell(paper_id=i, text=i) for i in ids])
        return CompareResult(papers=papers, rows=[row], summary="差异总结")

    monkeypatch.setattr(graph_module, "compare_papers", fake_compare)
    agent = make_agent(
        [_qa("对比 Other 和 Good", mentions=["Other Paper", "Good Paper"], intent="compare")]
    )
    agent.retriever.db.upsert_paper(PaperMeta(paper_id="good", title="Good Paper"), "html", 3)
    out = _collect(agent.stream("t", "对比 Other 和 Good"))
    assert seen == [["other", "good"]]
    assert "差异总结" in out["messages"][0]


def test_compare_with_missing_paper_supplements_then_compares(make_agent, monkeypatch):
    seen = []

    def failing_compare(services, ids, focus=""):
        seen.append(ids)
        raise ValueError("模型超时")

    monkeypatch.setattr(graph_module, "compare_papers", failing_compare)
    agent = make_agent(
        [_qa("对比 Other 和 Good", mentions=["Other Paper", "Good"], intent="compare")]
    )
    out = _collect(agent.stream("t", "对比 Other 和 Good"))
    assert out["interrupt"] is not None  # Good 不在库里，先联网补充
    out = _collect(agent.resume("t", ["good"]))
    assert seen == [["other", "good"]]
    assert out["messages"][0].startswith("对比失败")  # 异常转成消息，不中断对话
