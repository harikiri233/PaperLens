from paperlens.ingest.schema import Chunk
from paperlens.rag.answer import (
    answer_messages,
    build_context,
    check_citations,
    cited_numbers,
    strip_invalid,
)
from paperlens.rag.query import format_history
from paperlens.rag.retriever import ScoredChunk


def _evidence(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            Chunk(
                chunk_id=f"p#{i}",
                paper_id="p",
                index=i,
                kind="text",
                title="Paper",
                section="1 Intro",
                anchor=f"S1.p{i}",
                url=f"https://x/#S1.p{i}",
                content=f"content {i}",
            ),
            0.9,
        )
        for i in range(n)
    ]


def test_build_context_numbers_evidence():
    context, citations = build_context(_evidence(2))
    assert context.startswith("<evidence>") and context.endswith("</evidence>")
    assert "[1] 《Paper》 1 Intro\ncontent 0" in context
    assert [(c.n, c.anchor) for c in citations] == [(1, "S1.p0"), (2, "S1.p1")]


def test_answer_messages_keeps_recent_history():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    messages, _ = answer_messages("问题", _evidence(1), history)
    assert messages[0]["role"] == "system"
    assert "只作为资料" in messages[0]["content"]  # 提示注入防护
    assert len(messages) == 1 + 6 + 1
    assert messages[-1]["content"].endswith("问题：问题")


def test_cited_numbers_and_check():
    answer = "A [1][3]。B [2, 9]。C [1，2]。"
    assert cited_numbers(answer) == [1, 3, 2, 9]
    check = check_citations(answer, 6)
    assert check.used == [1, 3, 2] and check.invalid == [9] and check.has_citation
    assert not check_citations("没有引用", 6).has_citation


def test_strip_invalid():
    assert strip_invalid("A [2, 9]。B [9]。C [1]", [9]) == "A [2]。B 。C [1]"
    assert strip_invalid("A [1]", []) == "A [1]"


def test_format_history_truncates_long_answers():
    text = format_history(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a" * 1000}]
    )
    assert text.startswith("用户：q\n助手：")
    assert text.endswith("…") and len(text) < 450
