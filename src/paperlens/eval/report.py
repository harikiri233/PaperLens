"""把 eval/reports/results.json 渲染成 Markdown 报告（eval/reports/report.md）。"""

import json
from pathlib import Path

_CATEGORY = {
    "fact": "单篇事实",
    "term": "术语 / 缩写",
    "crosslingual": "跨语言改写",
    "lang_zh": "中文提问",
    "lang_en": "英文提问",
}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render(data: dict) -> str:
    retrieval = data["retrieval"]
    lines = ["# PaperLens 评估结果", "", "## 检索消融（可回答的 40 题）", ""]
    lines += [
        "| 配置 | 说明 | Hit@5 | Recall@5 | MRR@10 | 延迟 p50 |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in retrieval.items():
        m = r["metrics"]["all"]
        lines.append(
            f"| {name} | {data['configs'][name]} | {_pct(m['hit@5'])} | {_pct(m['recall@5'])} "
            f"| {m['mrr@10']:.3f} | {r['latency_p50']:.2f}s |"
        )
    lines += ["", "### 按题型的 Hit@5", ""]
    cats = [c for c in _CATEGORY if c in next(iter(retrieval.values()))["metrics"]]
    first = next(iter(retrieval.values()))
    head = " | ".join(f"{_CATEGORY[c]}（{first['n'][c]}）" for c in cats)
    lines += [f"| 配置 | {head} |", "|---" * (len(cats) + 1) + "|"]
    for name, r in retrieval.items():
        cells = " | ".join(_pct(r["metrics"][c]["hit@5"]) for c in cats)
        lines.append(f"| {name} | {cells} |")

    lines += [
        "",
        "## 证据判定阈值（A3 的重排最高分）",
        "",
        "| 阈值 | 可回答题被误拒 | 不可回答题被拒答 | 判定准确率 |",
        "|---|---|---|---|",
    ]
    for row in data["refusal"]:
        lines.append(
            f"| {row['threshold']} | {_pct(row['false_refuse'])} | {_pct(row['true_refuse'])} "
            f"| {_pct(row['accuracy'])} |"
        )

    answers = data.get("answers")
    if answers:
        lines += [
            "",
            "## 端到端问答（完整流程，阈值 0.3）",
            "",
            "| 指标 | 数值 |",
            "|---|---|",
            f"| 回答正确率（40 道可回答题，误拒记错） | {_pct(answers['accuracy_answerable'])} |",
            f"| 拒答准确率（50 题） | {_pct(answers['refusal_accuracy'])} |",
            f"| 不可回答题拒答率（证据判定） | {_pct(answers['true_refuse'])} |",
            f"| 不可回答题被正确处理（拒答或回答“资料未提及”） | "
            f"{_pct(answers.get('unanswerable_handled', 0))} |",
            f"| 可回答题误拒率 | {_pct(answers['false_refuse'])} |",
            f"| 引用编号合法率 | {_pct(answers['citation_valid_rate'])} |",
            f"| 带引用的回答占比 | {_pct(answers['answers_with_citation'])} |",
            f"| 延迟 p50 / 平均 | {answers['latency_p50']:.1f}s / {answers['latency_mean']:.1f}s |",
            f"| 单题成本 | ${answers['cost_per_question_usd']:.5f} |",
        ]
        wrong = [r for r in data.get("answer_records", []) if r["correct"] is False]
        if wrong:
            lines += ["", "答错 / 误拒的题：", ""]
            lines += [f"- {r['qid']}：{r['reason']}" for r in wrong]
    return "\n".join(lines) + "\n"


NOTES = """
## 方法说明

- 语料：30 篇 RAG / Agent 方向论文（`eval/corpus.txt`），2026 个块。
- 测试集 `eval/qa.jsonl`：50 题 = 单篇事实 25（中英各半）+ 术语 / 缩写 10 + 跨语言改写 5（纯中文、术语全部意译）
  + 语料外的无法回答题 10。可回答的 40 题由 LLM 依据随机抽取的块出题（`eval/build_qa.py`），
  逐题对照来源块核对参考答案，并人工改写了 10 道指代不清或类别不符的问题（`eval/review_edits.py`）。
- 相关性标注：出题的来源块，加上池化判定：五个配置 top-10 的并集交给 LLM 对照参考答案判断是否包含答案
  （判定结果缓存在 `eval/judgments.json`），平均每题 2.4 个相关块。池化只覆盖这五个配置检索到的块，
  其他块中的答案不会被计入，因此 Recall 是相对值，适合用来比较配置。
- 每个配置共用同一次查询分析的英文检索句；A0 用单独的 collection（正文直接嵌入，不带上下文标题）。
- 端到端问答走 `paperlens ask` 的路径；回答正确率由 LLM 对照参考答案判定，误拒记为答错。

## 评估中发现并修复的问题

第一次端到端评估的正确率是 92.5%。q04（“RAG 论文中 RAG-Sequence 的两种解码过程”）被误拒：查询分析抽出论文名 “RAG”，
按标题子串匹配命中了 “Leve**rag**ing Passage Retrieval…”（FiD），检索被限定在错误的论文里。修复：子串必须从单词开头匹配；
一个名称匹配到多篇论文时视为泛称，既不限定检索范围，也不当作“库里没有”去联网补充（`Database.match_paper`）。修复后 q04 答对，
正确率 95.0%。
（两次运行的单题成本 $0.00077 → $0.00039 是因为第二次落在 DeepSeek 非高峰时段、半价计费，和修复无关。）

剩下的两道误拒：q07 的 “Fusion-in-Decoder” 在库里的标题是 “Leveraging Passage Retrieval with Generative Models…”，
名称解析不到，最高重排分 0.27 略低于阈值；q35 问公式记号 $\\circ$ 的含义，重排分只有 0.06。
"""


def write_report(reports_dir: Path) -> Path:
    data = json.loads((reports_dir / "results.json").read_text(encoding="utf-8"))
    path = reports_dir / "report.md"
    path.write_text(render(data) + NOTES, encoding="utf-8")
    return path
