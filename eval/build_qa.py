"""生成候选测试题：从语料里抽块，让 LLM 依据单个块出题（之后人工审核、再做池化标注）。

用法：uv run python eval/build_qa.py  →  eval/qa_draft.jsonl

- fact 25 题（约一半中文）：单篇事实，问题要自洽，不照抄原文。
- term 10 题：围绕某个术语 / 缩写的含义或作用。
- crosslingual 5 题：纯中文提问，术语全部译成中文，考察英文检索句改写。
- unanswerable 10 题：语料之外的主题，手写在 UNANSWERABLE 里。
"""

import random
import re
import sys
from pathlib import Path

from pydantic import BaseModel

from paperlens.config import get_settings
from paperlens.eval.dataset import QAItem, save_items
from paperlens.ingest.schema import Chunk
from paperlens.services import Services

SEED = 7
OUT = Path("eval/qa_draft.jsonl")

# 语料（30 篇 RAG / Agent 论文）没有覆盖的主题；问题刻意贴近领域，避免靠主题词就能判断
UNANSWERABLE = [
    ("zh", "LoRA 微调时低秩矩阵的秩 r 一般取多少，对效果有什么影响？"),
    ("en", "How does Mamba's selective state space mechanism scale with sequence length?"),
    ("zh", "DPO 直接偏好优化相比 PPO 的 RLHF 少了哪个模型？"),
    ("en", "What quantization scheme does GPTQ use to compress LLM weights to 4 bits?"),
    ("zh", "Mixtral 8x7B 的路由器每个 token 激活几个专家？"),
    ("en", "Which tokenizer vocabulary size does Llama 3 use?"),
    ("zh", "FlashAttention 是如何通过分块减少显存读写的？"),
    ("en", "How does vLLM's PagedAttention manage the KV cache in fixed-size blocks?"),
    ("zh", "Stable Diffusion 的 VAE 把图像压缩到多少倍的潜空间？"),
    ("zh", "Constitutional AI 里用于 AI 反馈的宪法原则是怎么制定的？"),
]

_STYLE = {
    "fact": {
        "en": "Write the question in English.",
        "zh": "用中文提问；方法名、数据集名、模型名等专有名词可以保留英文。",
    },
    "term": {
        "en": "Write the question in English. The question must be about the meaning, role or "
        "definition of ONE specific term, acronym or special token that appears in the passage "
        "(e.g. what a token signals, what an acronym stands for, what a named component does).",
        "zh": "用中文提问，问题必须围绕段落里出现的某一个术语、缩写或特殊 token 的含义 / 作用"
        "（如某个缩写代表什么、某个特殊 token 表示什么、某个命名组件做什么），这个术语保留英文原文。",
    },
    "crosslingual": {
        "zh": "用纯中文提问：所有术语、方法名、数据集名都要意译成中文，问题里不能出现任何英文字母或缩写"
        "（例如 retrieval 译为“检索”，reflection token 译为“反思标记”，论文名用中文描述其做法）。",
    },
}

_SYSTEM = """You write evaluation questions for a retrieval system over a library of 30 RAG / LLM-agent papers.
You are given several passages from ONE paper. Pick the single passage that supports the most
specific, checkable question, and write one question whose answer is stated in that passage.

Requirements:
- The question must be self-contained: a reader who has not seen the passage must understand it.
  Name the paper's method / system / dataset where needed (unless the style rule forbids English).
- Ask about something specific to this paper (a design choice, number, finding, definition),
  not general knowledge.
- Paraphrase: do not copy any phrase longer than 4 words from the passage.
- The answer must be fully contained in the chosen passage. Keep the answer under 40 words.
Output JSON: {"passage": <index of the chosen passage>, "question": "...", "answer": "..."}"""


class Draft(BaseModel):
    passage: int
    question: str
    answer: str


_ACRONYM = re.compile(r"\b[A-Z][A-Za-z]*[A-Z][A-Za-z]*\b|\[[A-Z][a-z]+\]")


def body_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [
        c
        for c in chunks
        if c.kind == "text"
        and not c.is_appendix
        and len(c.content) > 600
        and not c.anchor.startswith("abstract")
        and not c.section.lower().startswith(("references", "acknowledg"))
    ]


def main() -> None:
    rng = random.Random(SEED)
    plan = (
        [("fact", "zh" if i % 2 else "en") for i in range(25)]
        + [("term", "zh" if i % 2 else "en") for i in range(10)]
        + [("crosslingual", "zh") for _ in range(5)]
    )
    items: list[QAItem] = []
    with Services(get_settings()) as services:
        papers = [r.meta.paper_id for r in services.db.list_papers()]
        rng.shuffle(papers)
        used: set[str] = set()
        for i, (category, lang) in enumerate(plan):
            pid = papers[i % len(papers)]
            pool = [
                c for c in body_chunks(services.vectors.paper_chunks(pid)) if c.chunk_id not in used
            ]
            if category == "term":
                pool = [c for c in pool if len(_ACRONYM.findall(c.content)) >= 3] or pool
            candidates = rng.sample(pool, min(4, len(pool)))
            passages = "\n\n".join(
                f"[{n}] ({c.section})\n{c.content}" for n, c in enumerate(candidates)
            )
            user = (
                f"Paper: {candidates[0].title}\n\nStyle rule: {_STYLE[category][lang]}\n\n"
                f"Passages:\n{passages}"
            )
            draft = services.llm.complete_json(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
                Draft,
                max_tokens=600,
                temperature=0.7,
            )
            chunk = candidates[min(max(draft.passage, 0), len(candidates) - 1)]
            used.add(chunk.chunk_id)
            items.append(
                QAItem(
                    id=f"q{len(items) + 1:02d}",
                    category=category,
                    lang=lang,
                    question=draft.question,
                    answer=draft.answer,
                    paper_id=pid,
                    source_chunk=chunk.chunk_id,
                    section=chunk.section,
                    relevant=[chunk.chunk_id],
                )
            )
            print(f"{items[-1].id} [{category}/{lang}] {pid} {draft.question}", file=sys.stderr)
        for lang, question in UNANSWERABLE:
            items.append(
                QAItem(
                    id=f"q{len(items) + 1:02d}",
                    category="unanswerable",
                    lang=lang,
                    question=question,
                )
            )
        save_items(OUT, items)
        print(
            f"写入 {OUT}，{len(items)} 题；花费 ${services.tracker.total_cost_usd:.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
