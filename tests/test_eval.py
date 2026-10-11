"""评估指标与汇总的单测（不调用模型）。"""

import pytest

from paperlens.eval.dataset import QAItem
from paperlens.eval.metrics import average, hit_at_k, mrr_at_k, recall_at_k
from paperlens.eval.runner import AnswerRecord, RunRecord, refusal_table, summarize_answers


def test_hit_recall_mrr():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert hit_at_k(ranked, {"c"}, 5) == 1.0
    assert hit_at_k(ranked, {"f"}, 5) == 0.0
    assert recall_at_k(ranked, {"b", "f"}, 5) == 0.5
    # 相关块多于 k 时分母取 k，前 k 条全相关就是满分
    assert recall_at_k(ranked[:2], {"a", "b", "x", "y"}, 2) == 1.0
    assert recall_at_k(ranked, set(), 5) == 0.0
    assert mrr_at_k(ranked, {"c", "e"}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(ranked, {"z"}, 10) == 0.0


def test_average():
    assert average([{"x": 1.0}, {"x": 0.0}]) == {"x": 0.5}
    assert average([]) == {}


def _item(qid: str, category: str) -> QAItem:
    return QAItem(id=qid, category=category, lang="en", question="q")


def test_refusal_table_thresholds():
    items = [_item("a", "fact"), _item("b", "unanswerable")]
    records = [RunRecord("a", [], 0.4, 0.1), RunRecord("b", [], 0.15, 0.1)]
    low, high = refusal_table(items, records, [0.1, 0.3])
    assert (low["false_refuse"], low["true_refuse"], low["accuracy"]) == (0.0, 0.0, 0.5)
    assert (high["false_refuse"], high["true_refuse"], high["accuracy"]) == (0.0, 1.0, 1.0)


def test_summarize_answers():
    records = [
        AnswerRecord("a", "fact", False, 0.9, 2.0, 0.001, "答案 [1][7]", 2, [7], True),
        AnswerRecord("b", "fact", True, 0.1, 1.0, 0.0005, correct=False),
        AnswerRecord("c", "unanswerable", True, 0.01, 1.0, 0.0005),
        AnswerRecord("d", "unanswerable", False, 0.5, 3.0, 0.001, "现有资料未提及 X。", 0, []),
    ]
    s = summarize_answers(records)
    assert s["accuracy_answerable"] == 0.5
    assert s["false_refuse"] == 0.5 and s["true_refuse"] == 0.5
    assert s["unanswerable_handled"] == 1.0  # d 没被拒答，但回答说明了资料未提及
    assert s["refusal_accuracy"] == 0.5
    assert s["citation_valid_rate"] == 0.5
