"""检索指标（确定性计算）。ranked 是按得分从高到低的 chunk_id，relevant 是相关块集合。"""

from collections.abc import Sequence
from statistics import mean


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """前 k 条里至少有一条相关块记 1。"""
    return 1.0 if any(c in relevant for c in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """前 k 条覆盖了多少相关块；分母取 min(|relevant|, k)，相关块多于 k 条时满分仍可达到。"""
    if not relevant:
        return 0.0
    found = sum(1 for c in ranked[:k] if c in relevant)
    return found / min(len(relevant), k)


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """第一条相关块排名的倒数；前 k 条都不相关记 0。"""
    for rank, c in enumerate(ranked[:k], start=1):
        if c in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_scores(ranked: Sequence[str], relevant: set[str]) -> dict[str, float]:
    return {
        "hit@5": hit_at_k(ranked, relevant, 5),
        "recall@5": recall_at_k(ranked, relevant, 5),
        "mrr@10": mrr_at_k(ranked, relevant, 10),
    }


def average(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: mean(r[key] for r in rows) for key in rows[0]}
