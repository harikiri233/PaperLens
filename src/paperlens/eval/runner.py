"""评估：A0–A4 检索消融 + 端到端问答（拒答、引用合法率、正确率、延迟、成本）。

流程（`paperlens eval`）：
1. 查询分析：每题调用一次，结果缓存到 eval/reports/analysis.json，所有配置共用同一个英文检索句。
2. 检索：每个配置对 50 题各取 top-10，记录排名和延迟。
3. 池化标注：各配置 top-10 的并集交给 LLM，对照参考答案判断哪些块包含答案；结果缓存到
   eval/judgments.json，并入 qa.jsonl 的 relevant。来源块始终算相关。
4. 指标：Hit@5、Recall@5、MRR@10（只统计可回答的 40 题），按题型拆分。
5. 端到端（--answers）：按 `paperlens ask` 的路径跑一遍：查询分析 → 检索 → 证据判定 → 生成，
   统计拒答准确率、引用编号合法率、回答正确率（LLM 对照参考答案判定）、延迟和单题成本。

A0 需要不带上下文标题的向量，单独建一个 collection（chunks_plain），只在第一次运行时嵌入。
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

from pydantic import BaseModel

from paperlens.config import Settings
from paperlens.eval.dataset import QAItem, load_items, save_items
from paperlens.eval.metrics import average, retrieval_scores
from paperlens.models import LLMClient, UsageTracker
from paperlens.rag.answer import check_citations, stream_answer
from paperlens.rag.query import QueryAnalysis, analyze_query, resolve_papers
from paperlens.rag.retriever import RetrievalOptions, Retriever
from paperlens.services import Services
from paperlens.store import VectorStore

PLAIN_COLLECTION = "chunks_plain"
TOP_N = 10  # 每个配置取多少条参与打分和池化


@dataclass(frozen=True)
class EvalConfig:
    name: str
    desc: str
    options: RetrievalOptions
    plain: bool = False  # True：使用不带上下文标题的 collection


CONFIGS: dict[str, EvalConfig] = {
    c.name: c
    for c in [
        EvalConfig(
            "A0",
            "仅稠密检索，无上下文标题",
            RetrievalOptions(use_bm25=False, use_rerank=False),
            True,
        ),
        EvalConfig("A1", "A0 + 上下文标题", RetrievalOptions(use_bm25=False, use_rerank=False)),
        EvalConfig("A2", "A1 + BM25（英文检索句）+ RRF", RetrievalOptions(use_rerank=False)),
        EvalConfig("A3", "A2 + bge-reranker 重排（完整检索）", RetrievalOptions()),
        EvalConfig("A4", "A3，但 BM25 通道用原问题", RetrievalOptions(use_english_query=False)),
    ]
}


@dataclass
class RunRecord:
    qid: str
    ranked: list[str]  # chunk_id，按得分从高到低
    top_score: float
    latency_s: float


@dataclass
class EvalPaths:
    root: Path = Path("eval")

    @property
    def qa(self) -> Path:
        return self.root / "qa.jsonl"

    @property
    def judgments(self) -> Path:
        return self.root / "judgments.json"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def analysis(self) -> Path:
        return self.reports / "analysis.json"


# ---------------------------------------------------------------- A0 用的无标题 collection


def ensure_plain_collection(services: Services, log: Callable[[str], None] = print) -> VectorStore:
    """建一个只用正文（不带上下文标题）嵌入的 collection；块数和主库一致时直接复用。"""
    s = services.settings
    store = VectorStore(s.milvus_path, PLAIN_COLLECTION, s.embedding_dim)
    total = services.vectors.count()
    if store.count() == total:
        return store
    log(f"建立 {PLAIN_COLLECTION}：重新嵌入 {total} 个块（不带上下文标题）…")
    store.client.drop_collection(PLAIN_COLLECTION)
    store.close()
    store = VectorStore(s.milvus_path, PLAIN_COLLECTION, s.embedding_dim)
    for record in services.db.list_papers():
        chunks = services.vectors.paper_chunks(record.meta.paper_id)
        texts = [c.index_text(contextual=False) for c in chunks]
        store.insert(chunks, services.sf.embed(texts), texts)
    return store


class _PlainServices(Services):
    """和主 Services 共用模型客户端，只把向量库换成无标题 collection。"""

    def __init__(self, base: Services, store: VectorStore) -> None:
        super().__init__(base.settings, tracker=base.tracker)
        self.__dict__.update(db=base.db, sf=base.sf, llm=base.llm, vectors=store)


# ---------------------------------------------------------------- 查询分析与检索


def analyze_all(
    services: Services, items: list[QAItem], cache_path: Path
) -> dict[str, QueryAnalysis]:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    for item in items:
        key = f"{item.id}:{item.question}"
        if key not in cache:
            cache[key] = analyze_query(services.llm, item.question).model_dump()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return {item.id: QueryAnalysis(**cache[f"{item.id}:{item.question}"]) for item in items}


def run_config(
    retriever: Retriever,
    config: EvalConfig,
    items: list[QAItem],
    analyses: dict[str, QueryAnalysis],
) -> list[RunRecord]:
    records = []
    for item in items:
        analysis = analyses[item.id]
        start = time.perf_counter()
        result = retriever.retrieve(
            item.question,
            english_query=analysis.english_query,
            top_k=TOP_N,
            options=config.options,
        )
        records.append(
            RunRecord(
                qid=item.id,
                ranked=[e.chunk.chunk_id for e in result.evidence],
                top_score=result.top_score,
                latency_s=time.perf_counter() - start,
            )
        )
    return records


# ---------------------------------------------------------------- 池化标注


class _Judgment(BaseModel):
    relevant: list[int]


_JUDGE_SYSTEM = """You label search results for a retrieval benchmark over research papers.
Given a question, its reference answer, and numbered passages, list the numbers of passages that
contain the information needed to answer the question (the key fact of the reference answer must
be stated in the passage, possibly in different words). A passage that is merely on the same topic,
or that only mentions the terms without giving the answer, is NOT relevant.
Output JSON: {"relevant": [numbers]} (empty list if none)."""


def judge_pool(
    llm: LLMClient,
    item: QAItem,
    chunk_ids: list[str],
    texts: dict[str, str],
) -> list[str]:
    numbered = "\n\n".join(f"[{n}] {texts[c][:2500]}" for n, c in enumerate(chunk_ids))
    user = f"Question: {item.question}\nReference answer: {item.answer}\n\nPassages:\n{numbered}"
    result = llm.complete_json(
        [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}],
        _Judgment,
        max_tokens=300,
    )
    return [chunk_ids[n] for n in result.relevant if 0 <= n < len(chunk_ids)]


def apply_pooling(
    services: Services,
    judge_llm: LLMClient,
    items: list[QAItem],
    runs: dict[str, list[RunRecord]],
    judgments_path: Path,
) -> int:
    """对每道可回答的题，把各配置 top-10 里还没判过的块交给 LLM 判定，更新 item.relevant。

    返回本次新判定的块数。判定结果按 (题号, 块) 缓存，重跑时不重复调用。
    """
    cache: dict[str, dict[str, bool]] = {}
    if judgments_path.exists():
        cache = json.loads(judgments_path.read_text(encoding="utf-8"))
    by_paper: dict[str, dict[str, str]] = {}

    def chunk_text(chunk_id: str) -> str:
        pid = chunk_id.rsplit("#", 1)[0]
        if pid not in by_paper:
            by_paper[pid] = {c.chunk_id: c.index_text() for c in services.vectors.paper_chunks(pid)}
        return by_paper[pid][chunk_id]

    new = 0
    for item in items:
        if not item.answerable:
            continue
        judged = cache.setdefault(item.id, {})
        pool: list[str] = []
        for records in runs.values():
            for r in records:
                if r.qid == item.id:
                    pool += [c for c in r.ranked if c not in pool]
        todo = [c for c in pool if c not in judged and c != item.source_chunk]
        if todo:
            texts = {c: chunk_text(c) for c in todo}
            positive = set(judge_pool(judge_llm, item, todo, texts))
            judged.update({c: c in positive for c in todo})
            new += len(todo)
        item.relevant = [item.source_chunk] + sorted(
            c for c, ok in judged.items() if ok and c != item.source_chunk
        )
    judgments_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return new


# ---------------------------------------------------------------- 指标汇总


def summarize(items: list[QAItem], records: list[RunRecord]) -> dict:
    by_id = {i.id: i for i in items}
    rows: dict[str, list[dict[str, float]]] = {"all": []}
    for r in records:
        item = by_id[r.qid]
        if not item.answerable:
            continue
        scores = retrieval_scores(r.ranked, set(item.relevant))
        rows["all"].append(scores)
        rows.setdefault(item.category, []).append(scores)
        rows.setdefault(f"lang_{item.lang}", []).append(scores)
    latencies = [r.latency_s for r in records]
    return {
        "metrics": {k: average(v) for k, v in rows.items()},
        "n": {k: len(v) for k, v in rows.items()},
        "latency_p50": median(latencies),
        "latency_mean": mean(latencies),
    }


def refusal_table(
    items: list[QAItem], records: list[RunRecord], thresholds: list[float]
) -> list[dict]:
    """按重排最高分做证据判定：不同阈值下，可回答题被误拒、不可回答题被正确拒答的比例。"""
    by_id = {i.id: i for i in items}
    rows = []
    for t in thresholds:
        ans = [r for r in records if by_id[r.qid].answerable]
        una = [r for r in records if not by_id[r.qid].answerable]
        false_refuse = sum(r.top_score < t for r in ans) / len(ans)
        true_refuse = sum(r.top_score < t for r in una) / len(una)
        correct = sum(r.top_score >= t for r in ans) + sum(r.top_score < t for r in una)
        rows.append(
            {
                "threshold": t,
                "false_refuse": false_refuse,
                "true_refuse": true_refuse,
                "accuracy": correct / len(records),
            }
        )
    return rows


# ---------------------------------------------------------------- 端到端问答


class _Grade(BaseModel):
    correct: bool
    reason: str = ""


_GRADE_SYSTEM = """You grade answers from a paper QA system against a reference answer.
The system answer is correct if it states the key fact(s) of the reference answer (wording,
language and extra correct detail do not matter). It is incorrect if it misses the key fact,
contradicts it, or says the information is not available.
Output JSON: {"correct": true/false, "reason": "<short>"}"""


@dataclass
class AnswerRecord:
    qid: str
    category: str
    refused: bool
    top_score: float
    latency_s: float
    cost_usd: float
    answer: str = ""
    n_citations: int = 0
    invalid_citations: list[int] = field(default_factory=list)
    correct: bool | None = None
    reason: str = ""


def answer_one(services: Services, judge_llm: LLMClient, item: QAItem) -> AnswerRecord:
    """和 `paperlens ask` 相同的路径：查询分析 → 点名论文过滤 → 检索 → 证据判定 → 生成。"""
    tracker = services.tracker
    cost_before = tracker.total_cost_usd
    start = time.perf_counter()
    analysis = analyze_query(services.llm, item.question)
    paper_ids = resolve_papers(services.db, analysis.paper_mentions)
    result = Retriever(services).retrieve(
        analysis.standalone_question,
        english_query=analysis.english_query,
        paper_ids=paper_ids or None,
    )
    record = AnswerRecord(
        qid=item.id,
        category=item.category,
        refused=result.top_score < services.settings.evidence_threshold,
        top_score=result.top_score,
        latency_s=0.0,
        cost_usd=0.0,
    )
    if not record.refused:
        tokens, citations = stream_answer(
            services.llm, analysis.standalone_question, result.evidence
        )
        record.answer = "".join(tokens)
        check = check_citations(record.answer, len(citations))
        record.n_citations = len(check.used) + len(check.invalid)
        record.invalid_citations = check.invalid
    record.latency_s = time.perf_counter() - start
    record.cost_usd = tracker.total_cost_usd - cost_before
    if item.answerable and not record.refused:
        grade = judge_llm.complete_json(
            [
                {"role": "system", "content": _GRADE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Question: {item.question}\nReference answer: {item.answer}\n\n"
                    f"System answer:\n{record.answer}",
                },
            ],
            _Grade,
            max_tokens=200,
        )
        record.correct, record.reason = grade.correct, grade.reason
    elif item.answerable:
        record.correct, record.reason = False, "误拒答"
    return record


def _says_not_found(answer: str) -> bool:
    head = answer[:120]
    return "未提及" in head or "没有找到" in head or "无法回答" in head


def summarize_answers(records: list[AnswerRecord]) -> dict:
    ans = [r for r in records if r.category != "unanswerable"]
    una = [r for r in records if r.category == "unanswerable"]
    answered = [r for r in records if not r.refused]
    total_cites = sum(r.n_citations for r in answered)
    invalid = sum(len(r.invalid_citations) for r in answered)
    with_cite = sum(r.n_citations > 0 for r in answered)
    return {
        "n": len(records),
        "accuracy_answerable": sum(bool(r.correct) for r in ans) / len(ans) if ans else 0.0,
        "false_refuse": sum(r.refused for r in ans) / len(ans) if ans else 0.0,
        "true_refuse": sum(r.refused for r in una) / len(una) if una else 0.0,
        # 不可回答题：证据判定拒答，或通过了判定但回答开头就说明资料未提及（提示词规则 2）
        "unanswerable_handled": sum(r.refused or _says_not_found(r.answer) for r in una) / len(una)
        if una
        else 0.0,
        "refusal_accuracy": (sum(not r.refused for r in ans) + sum(r.refused for r in una))
        / len(records),
        "citation_valid_rate": 1 - invalid / total_cites if total_cites else 1.0,
        "answers_with_citation": with_cite / len(answered) if answered else 0.0,
        "latency_p50": median(r.latency_s for r in records),
        "latency_mean": mean(r.latency_s for r in records),
        "cost_per_question_usd": mean(r.cost_usd for r in records),
    }


# ---------------------------------------------------------------- 总入口


@dataclass
class EvalOutput:
    retrieval: dict[str, dict]
    refusal: list[dict]
    answers: dict | None
    new_judgments: int
    judge_cost_usd: float


def run_eval(
    settings: Settings,
    *,
    configs: list[str] | None = None,
    answers: bool = False,
    paths: EvalPaths | None = None,
    log: Callable[[str], None] = print,
) -> EvalOutput:
    paths = paths or EvalPaths()
    items = load_items(paths.qa)
    names = configs or list(CONFIGS)
    judge_tracker = UsageTracker()
    with Services(settings) as services:
        judge_llm = LLMClient(settings, tracker=judge_tracker)
        log(f"查询分析：{len(items)} 题（有缓存时跳过）")
        analyses = analyze_all(services, items, paths.analysis)

        runs: dict[str, list[RunRecord]] = {}
        plain_store = None
        for name in names:
            config = CONFIGS[name]
            target: Services = services
            if config.plain:
                plain_store = plain_store or ensure_plain_collection(services, log)
                target = _PlainServices(services, plain_store)
            log(f"检索 {name}：{config.desc}")
            runs[name] = run_config(Retriever(target), config, items, analyses)

        new = apply_pooling(services, judge_llm, items, runs, paths.judgments)
        save_items(paths.qa, items)
        log(f"池化标注：新判定 {new} 个块")

        retrieval = {name: summarize(items, records) for name, records in runs.items()}
        refusal_source = runs.get("A3") or next(iter(runs.values()))
        refusal = refusal_table(items, refusal_source, [0.05, 0.1, 0.2, 0.3, 0.5])

        answer_summary = None
        answer_records: list[AnswerRecord] = []
        if answers:
            for item in items:
                log(f"问答 {item.id}")
                answer_records.append(answer_one(services, judge_llm, item))
            answer_summary = summarize_answers(answer_records)
        if plain_store:
            plain_store.close()

    paths.reports.mkdir(parents=True, exist_ok=True)
    dump = {
        "configs": {n: CONFIGS[n].desc for n in names},
        "retrieval": retrieval,
        "refusal": refusal,
        "runs": {n: [r.__dict__ for r in rs] for n, rs in runs.items()},
    }
    if answers:
        dump["answers"] = answer_summary
        dump["answer_records"] = [r.__dict__ for r in answer_records]
    (paths.reports / "results.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return EvalOutput(retrieval, refusal, answer_summary, new, judge_tracker.total_cost_usd)
