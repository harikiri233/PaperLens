"""问答 Agent：LangGraph 状态图。

    analyze ──chat──────────────────────────────→ chat → END
       │ ──discover──→ search_arxiv
       │ ──explain（点名 1 篇且在库里）──→ card → END
       │ ──compare（点名 2–5 篇且都在库里）──→ compare → END
       └──其余──→ retrieve
    retrieve ──证据充足──→ generate → END
             ──不足且未补充过──→ search_arxiv → confirm_ingest（interrupt）→ ingest → retrieve
             ──补充后仍不足 / 联网关闭──→ refuse → END
    ingest 之后按意图回到 card / compare / retrieve（discover 直接结束）

流式输出：节点通过 get_stream_writer() 写出 status / token / citations / message 事件，
人机确认通过 interrupt() 暂停，调用方用 resume() 传回用户的选择。
"""

import sqlite3
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from paperlens.agent.state import AgentState
from paperlens.ingest.arxiv_client import ArxivClient
from paperlens.ingest.pipeline import ingest_arxiv
from paperlens.ingest.schema import Chunk
from paperlens.rag.answer import Citation, check_citations, stream_answer, strip_invalid
from paperlens.rag.card import anchor_url, card_markdown, get_card
from paperlens.rag.compare import MAX_PAPERS, compare_markdown, compare_papers
from paperlens.rag.query import QueryAnalysis, analyze_query
from paperlens.rag.retriever import Retriever, ScoredChunk
from paperlens.services import Services

Event = dict[str, Any]

# 联网候选论文的最低重排分（问题 vs 标题 + 摘要）；实测无关论文基本在 0.01 以下
_CANDIDATE_MIN_SCORE = 0.05

_CHAT_SYSTEM = (
    "你是 PaperLens，一个本地论文知识库助手：可以回答知识库里论文的内容问题（带引用）、"
    "解读和对比论文、去 arXiv 找新论文。现在用户在闲聊或问与论文无关的问题，简短友好地回答；"
    "需要时提示用户可以问论文相关的问题。"
)


def _emit(kind: str, **payload: Any) -> None:
    get_stream_writer()({"type": kind, **payload})


def _history(messages: list[AnyMessage]) -> list[dict]:
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": str(m.content)}
        for m in messages
        if m.type in ("human", "ai")
    ]


def _scored(item: dict) -> ScoredChunk:
    return ScoredChunk(Chunk.model_validate(item["chunk"]), item["score"])


class PaperAgent:
    def __init__(self, services: Services) -> None:
        self.services = services
        self.settings = services.settings
        self.retriever = Retriever(services)
        path = self.settings.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self.graph = self._build().compile(checkpointer=SqliteSaver(self._conn))

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PaperAgent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ 图结构

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("analyze", self.analyze)
        g.add_node("chat", self.chat)
        g.add_node("retrieve", self.retrieve)
        g.add_node("generate", self.generate)
        g.add_node("search_arxiv", self.search_arxiv)
        g.add_node("confirm_ingest", self.confirm_ingest)
        g.add_node("ingest", self.ingest)
        g.add_node("refuse", self.refuse)
        g.add_node("card", self.card)
        g.add_node("compare", self.compare)

        g.add_edge(START, "analyze")
        g.add_conditional_edges(
            "analyze", self.route_intent, ["chat", "retrieve", "search_arxiv", "card", "compare"]
        )
        g.add_conditional_edges(
            "retrieve", self.grade_evidence, ["generate", "search_arxiv", "refuse"]
        )
        g.add_conditional_edges(
            "search_arxiv",
            lambda s: "confirm_ingest" if s.get("candidates") else "refuse",
            ["confirm_ingest", "refuse"],
        )
        g.add_conditional_edges(
            "confirm_ingest",
            lambda s: "ingest" if s.get("picked") else "refuse",
            ["ingest", "refuse"],
        )
        g.add_conditional_edges(
            "ingest", self.after_ingest, ["retrieve", "card", "compare", "refuse", END]
        )
        for node in ("chat", "generate", "refuse", "card", "compare"):
            g.add_edge(node, END)
        return g

    # ------------------------------------------------------------ 路由

    @staticmethod
    def _analysis(state: AgentState) -> QueryAnalysis:
        return QueryAnalysis.model_validate(state["analysis"])

    def _structured_route(self, state: AgentState) -> str | None:
        """解读 / 对比：点名的论文都在库里时走卡片或对比表，否则交给检索问答（会触发联网补充）。"""
        analysis = self._analysis(state)
        ids, missing = self._resolve(analysis, state.get("ingested", []))
        if missing:
            return None
        if analysis.intent == "explain" and len(ids) == 1:
            return "card"
        if analysis.intent == "compare" and 2 <= len(ids) <= MAX_PAPERS:
            return "compare"
        return None

    def route_intent(self, state: AgentState) -> str:
        intent = state["analysis"]["intent"]
        if intent == "chat":
            return "chat"
        if intent == "discover":
            return "search_arxiv"
        return self._structured_route(state) or "retrieve"

    def grade_evidence(self, state: AgentState) -> str:
        """证据判定：重排最高分达到阈值，且点名的论文都在库里，才算充足。"""
        threshold = self.settings.evidence_threshold
        score_ok = state.get("top_score", 0.0) >= threshold
        if score_ok and not state.get("missing"):
            return "generate"
        if not state.get("supplemented") and self.settings.online_fallback == "ask":
            return "search_arxiv"
        # 已经补充过：有些点名的论文仍然找不到，但已有证据足够时照常回答
        return "generate" if score_ok else "refuse"

    def after_ingest(self, state: AgentState) -> str:
        if state["analysis"]["intent"] == "discover":
            return END
        if not state.get("ingested"):
            return "refuse"
        return self._structured_route(state) or "retrieve"

    # ------------------------------------------------------------ 节点

    def analyze(self, state: AgentState) -> dict:
        messages = state["messages"]
        question = str(messages[-1].content)
        analysis = analyze_query(self.services.llm, question, _history(messages[:-1]))
        detail = f"意图 {analysis.intent}"
        if analysis.standalone_question != question:
            detail += f" · 改写为：{analysis.standalone_question}"
        if analysis.paper_mentions:
            detail += f" · 提到的论文：{', '.join(analysis.paper_mentions)}"
        _emit("status", text=f"分析问题：{detail}")
        # 每一轮开始时清空上一轮的中间结果
        return {
            "analysis": analysis.model_dump(),
            "paper_ids": [],
            "missing": [],
            "evidence": [],
            "top_score": 0.0,
            "candidates": [],
            "picked": [],
            "ingested": [],
            "supplemented": False,
            "answer": "",
            "citations": [],
        }

    def chat(self, state: AgentState) -> dict:
        messages = state["messages"]
        result = self.services.llm.chat(
            [
                {"role": "system", "content": _CHAT_SYSTEM},
                *_history(messages[:-1])[-6:],
                {"role": "user", "content": str(messages[-1].content)},
            ],
            max_tokens=500,
            temperature=0.5,
        )
        _emit("message", text=result.content)
        return {"messages": [AIMessage(result.content)], "answer": result.content}

    def _resolve(self, analysis: QueryAnalysis, ingested: list[str]) -> tuple[list[str], list[str]]:
        """点名的论文 → 库里的 paper_id；返回 (找到的 ID, 找不到的名称)。"""
        ids: list[str] = []
        missing: list[str] = []
        for mention in analysis.paper_mentions:
            found = self.services.db.find_papers(mention, limit=1)
            if found:
                if found[0].meta.paper_id not in ids:
                    ids.append(found[0].meta.paper_id)
            else:
                missing.append(mention)
        if analysis.paper_mentions and ingested:
            # 联网补充的论文标题里不一定出现用户说的简称（如 FLARE），直接并入过滤范围
            ids += [i for i in ingested if i not in ids]
            missing = []
        return ids, missing

    def retrieve(self, state: AgentState) -> dict:
        analysis = self._analysis(state)
        ids, missing = self._resolve(analysis, state.get("ingested", []))
        question = analysis.standalone_question
        top_k = self.settings.retrieve_top_k

        if analysis.intent == "compare" and len(ids) >= 2:
            # 对比类问题：在每篇论文内部分别检索，保证每篇都有证据
            per_paper = max(2, top_k // len(ids))
            evidence: list[ScoredChunk] = []
            scores: list[float] = []
            for pid in ids:
                r = self.retriever.retrieve(
                    question,
                    english_query=analysis.english_query,
                    paper_ids=[pid],
                    top_k=per_paper,
                )
                evidence += r.evidence
                scores.append(r.top_score)
            top = min(scores)
        else:
            r = self.retriever.retrieve(
                question, english_query=analysis.english_query, paper_ids=ids or None
            )
            evidence, top = r.evidence, r.top_score

        scope = f"限定 {len(ids)} 篇论文" if ids else "全库"
        note = f"；库里没有：{', '.join(missing)}" if missing else ""
        _emit("status", text=f"检索（{scope}）：{len(evidence)} 条证据，最高重排分 {top:.2f}{note}")
        return {
            "paper_ids": ids,
            "missing": missing,
            "evidence": [{"chunk": e.chunk.model_dump(), "score": e.score} for e in evidence],
            "top_score": top,
        }

    def generate(self, state: AgentState) -> dict:
        analysis = self._analysis(state)
        evidence = [_scored(e) for e in state["evidence"]]
        history = _history(state["messages"][:-1])
        tokens, citations = stream_answer(
            self.services.llm, analysis.standalone_question, evidence, history
        )
        answer = ""
        for piece in tokens:
            answer += piece
            _emit("token", text=piece)
        check = check_citations(answer, len(citations))
        answer = strip_invalid(answer, check.invalid)
        used: list[Citation] = [c for c in citations if c.n in check.used]
        _emit("citations", citations=used, invalid=check.invalid)
        return {
            "answer": answer,
            "citations": [asdict(c) for c in used],
            "messages": [AIMessage(answer)],
        }

    def card(self, state: AgentState) -> dict:
        analysis = self._analysis(state)
        ids, _ = self._resolve(analysis, state.get("ingested", []))
        record = self.services.db.get_paper(ids[0])
        cached = self.services.db.get_card(ids[0]) is not None
        _emit(
            "status",
            text="读取已缓存的解读卡片" if cached else f"通读全文生成解读卡片：{record.meta.title}",
        )
        try:
            card = get_card(self.services, ids[0])
        except Exception as exc:
            text = f"解读卡片生成失败：{type(exc).__name__}: {exc}"
            _emit("message", text=text)
            return {"messages": [AIMessage(text)], "answer": text}
        text = card_markdown(card, title=record.meta.title, url_of=lambda a: anchor_url(record, a))
        _emit("card", paper_id=ids[0], title=record.meta.title, card=card.model_dump())
        _emit("message", text=text)
        return {"paper_ids": ids, "messages": [AIMessage(text)], "answer": text}

    def compare(self, state: AgentState) -> dict:
        analysis = self._analysis(state)
        ids, _ = self._resolve(analysis, state.get("ingested", []))
        _emit("status", text=f"基于解读卡片对比 {len(ids)} 篇论文（缺卡片的先生成）")
        try:
            result = compare_papers(self.services, ids, focus=analysis.standalone_question)
        except Exception as exc:
            text = f"对比失败：{type(exc).__name__}: {exc}"
            _emit("message", text=text)
            return {"messages": [AIMessage(text)], "answer": text}
        text = compare_markdown(result)
        _emit("compare", result=result.model_dump())
        _emit("message", text=text)
        return {"paper_ids": ids, "messages": [AIMessage(text)], "answer": text}

    def search_arxiv(self, state: AgentState) -> dict:
        analysis = self._analysis(state)
        missing = state.get("missing", [])
        found = []
        try:
            with ArxivClient(self.settings) as arxiv:
                if missing:
                    _emit("status", text=f"去 arXiv 查找论文：{', '.join(missing)}")
                    for mention in missing:
                        found += arxiv.search_keywords([mention], max_results=8)
                else:
                    keywords = analysis.search_keywords or [analysis.english_query]
                    _emit("status", text=f"去 arXiv 检索：{' / '.join(keywords)}")
                    found = arxiv.search_keywords(keywords, max_results=15)
        except Exception as exc:  # 网络问题不应让整轮对话失败
            _emit("status", text=f"arXiv 检索失败：{type(exc).__name__}: {exc}")

        unique = {}
        for meta in found:
            if meta.paper_id not in unique and not self.services.db.get_paper(meta.paper_id):
                unique[meta.paper_id] = meta
        metas = list(unique.values())

        # arXiv 的关键词检索噪声很大（实测搜 "FLARE" 排第一的是联邦学习框架），
        # 用 reranker 按"问题 vs 标题 + 摘要"重新打分，过滤掉明显无关的
        candidates: list[dict] = []
        if metas:
            ranked = self.services.sf.rerank(
                analysis.standalone_question,
                [f"{m.title}\n{m.abstract}" for m in metas],
            )
            for r in ranked[:5]:
                if r.score < _CANDIDATE_MIN_SCORE:
                    continue
                m = metas[r.index]
                candidates.append(
                    {
                        "paper_id": m.paper_id,
                        "version": m.version,
                        "title": m.title,
                        "published": m.published,
                        "abstract": m.abstract[:300],
                        "score": round(r.score, 3),
                    }
                )
        _emit("status", text=f"找到 {len(candidates)} 篇不在库里的相关论文")
        return {"candidates": candidates, "supplemented": True}

    def confirm_ingest(self, state: AgentState) -> dict:
        """人机确认：暂停图的执行，等用户勾选要入库的论文。resume 时传入 paper_id 列表。"""
        discover = state["analysis"]["intent"] == "discover"
        message = (
            "在 arXiv 上找到以下论文，选择要入库的："
            if discover
            else "本地知识库里的证据不足。在 arXiv 上找到以下论文，选择要入库的，入库后我再回答："
        )
        picked = interrupt({"message": message, "candidates": state["candidates"]})
        valid = {c["paper_id"] for c in state["candidates"]}
        return {"picked": [p for p in (picked or []) if p in valid]}

    def ingest(self, state: AgentState) -> dict:
        picked = set(state["picked"])
        targets = [
            f"{c['paper_id']}{c['version']}" for c in state["candidates"] if c["paper_id"] in picked
        ]

        def report(r) -> None:
            label = {"added": "已入库", "updated": "已更新", "skipped": "已在库中"}.get(
                r.status, "入库失败"
            )
            _emit("status", text=f"{label}：{r.title[:60]}（{r.n_chunks} 块）{r.message}")

        results = ingest_arxiv(self.services, targets, on_result=report)
        ok = [r for r in results if r.status != "failed"]
        update: dict = {"ingested": [r.paper_id for r in ok]}
        if state["analysis"]["intent"] == "discover":
            if ok:
                lines = [f"已入库 {len(ok)} 篇论文，现在可以直接提问了："]
                lines += [f"- {r.title}（{r.paper_id}）" for r in ok]
                text = "\n".join(lines)
            else:
                text = "论文入库失败了，可以稍后重试。"
            _emit("message", text=text)
            update["messages"] = [AIMessage(text)]
            update["answer"] = text
        return update

    def refuse(self, state: AgentState) -> dict:
        analysis = self._analysis(state)
        if analysis.intent == "discover":
            text = (
                "好的，这次不入库新论文。"
                if state.get("candidates")
                else "arXiv 上没有找到相关的新论文（或都已经在库里了），可以换个关键词试试。"
            )
        else:
            parts = ["本地知识库里没有找到足够相关的内容，无法可靠地回答这个问题。"]
            if state.get("missing"):
                parts.append(f"库里没有这些论文：{', '.join(state['missing'])}。")
            titles: list[str] = []
            for e in state.get("evidence", []):
                title = e["chunk"]["title"]
                if title not in titles:
                    titles.append(title)
            if titles:
                parts.append(
                    "相关度最高的几篇是：" + "；".join(f"《{t}》" for t in titles[:3]) + "。"
                )
            if self.settings.online_fallback == "off":
                parts.append("（联网补充已关闭。）")
            text = "\n\n".join(parts)
        _emit("message", text=text)
        return {"messages": [AIMessage(text)], "answer": text}

    # ------------------------------------------------------------ 调用接口

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _run(self, graph_input: Any, thread_id: str) -> Iterator[Event]:
        stream = self.graph.stream(
            graph_input, self._config(thread_id), stream_mode=["custom", "updates"]
        )
        try:
            for mode, chunk in stream:
                if mode == "custom":
                    yield chunk
                elif "__interrupt__" in chunk:
                    yield {"type": "interrupt", "payload": chunk["__interrupt__"][0].value}
        except Exception as exc:  # 把错误作为一条消息返回，不中断对话
            yield {"type": "message", "text": f"出错了：{type(exc).__name__}: {exc}"}

    def stream(self, thread_id: str, question: str) -> Iterator[Event]:
        return self._run({"messages": [HumanMessage(question)]}, thread_id)

    def resume(self, thread_id: str, picked: list[str]) -> Iterator[Event]:
        return self._run(Command(resume=picked), thread_id)

    def pending_interrupt(self, thread_id: str) -> dict | None:
        """会话是否停在人机确认上（如 Web 端刷新页面后恢复）。"""
        snapshot = self.graph.get_state(self._config(thread_id))
        for task in snapshot.tasks:
            if task.interrupts:
                return task.interrupts[0].value
        return None

    def history(self, thread_id: str) -> list[dict]:
        snapshot = self.graph.get_state(self._config(thread_id))
        return _history(snapshot.values.get("messages", []))
