"""混合检索：dense + BM25 → RRF 融合 → cross-encoder 重排。

- 稠密通道用原问题（bge-m3 多语言，中文问题可以直接检索英文段落）。
- BM25 通道用英文检索句（词面匹配，中文词匹配不到英文原文）；没有英文句时退回原问题。
- explain=True 时额外跑一遍两路单独检索，返回各阶段的候选和分数，供调试和评估使用。
"""

from dataclasses import dataclass, field

from paperlens.ingest.schema import Chunk
from paperlens.services import Services
from paperlens.store import Hit


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # 重排分数（0–1）；关闭重排时为 RRF 分数


@dataclass
class RetrievalResult:
    query: str
    bm25_query: str
    evidence: list[ScoredChunk]
    dense: list[Hit] = field(default_factory=list)  # explain 时才填充
    bm25: list[Hit] = field(default_factory=list)
    fused: list[Hit] = field(default_factory=list)

    @property
    def top_score(self) -> float:
        return self.evidence[0].score if self.evidence else 0.0


@dataclass
class RetrievalOptions:
    """各项开关，供消融实验使用；默认值即完整流程。"""

    use_bm25: bool = True
    use_rerank: bool = True
    use_english_query: bool = True
    include_appendix: bool = True


class Retriever:
    def __init__(self, services: Services) -> None:
        self.services = services
        self.settings = services.settings

    def retrieve(
        self,
        query: str,
        *,
        english_query: str = "",
        paper_ids: list[str] | None = None,
        top_k: int | None = None,
        options: RetrievalOptions | None = None,
        explain: bool = False,
    ) -> RetrievalResult:
        opts = options or RetrievalOptions()
        s = self.settings
        top_k = top_k or s.retrieve_top_k
        bm25_query = english_query.strip() if opts.use_english_query and english_query else query
        vector = self.services.sf.embed([query])[0]
        store = self.services.vectors

        result = RetrievalResult(query=query, bm25_query=bm25_query, evidence=[])
        if explain:
            result.dense = store.dense_search(
                vector, limit=s.retrieve_candidates, paper_ids=paper_ids
            )
            if opts.use_bm25:
                result.bm25 = store.bm25_search(
                    bm25_query, limit=s.retrieve_candidates, paper_ids=paper_ids
                )

        if opts.use_bm25:
            fused = store.hybrid_search(
                vector,
                bm25_query,
                candidates=s.retrieve_candidates,
                limit=s.retrieve_fused,
                paper_ids=paper_ids,
            )
        else:
            fused = (
                result.dense[: s.retrieve_fused]
                if explain
                else store.dense_search(vector, limit=s.retrieve_fused, paper_ids=paper_ids)
            )
        if not opts.include_appendix:
            fused = [h for h in fused if not h.chunk.is_appendix]
        result.fused = fused

        if opts.use_rerank and fused:
            ranked = self.services.sf.rerank(
                query, [h.chunk.index_text() for h in fused], top_n=top_k
            )
            result.evidence = [ScoredChunk(fused[r.index].chunk, r.score) for r in ranked]
        else:
            result.evidence = [ScoredChunk(h.chunk, h.score) for h in fused[:top_k]]
        return result
