from paperlens.ingest.chunker import approx_tokens, chunk_paper, split_sentences
from paperlens.ingest.schema import Block, PaperMeta, ParsedPaper, Section


def _paper(sections: list[Section]) -> ParsedPaper:
    return ParsedPaper(
        meta=PaperMeta(paper_id="2401.00001", version="v1", title="Toy"),
        source="html",
        sections=sections,
    )


def _para(anchor: str, n_chars: int) -> Block:
    sentence = "This is a sentence about retrieval. "
    return Block(anchor=anchor, text=(sentence * (n_chars // len(sentence) + 1))[:n_chars].strip())


def test_merges_short_paragraphs_within_section():
    paper = _paper(
        [Section(anchor="S1", path=["1 Intro"], blocks=[_para("S1.p1", 400), _para("S1.p2", 400)])]
    )
    chunks = chunk_paper(paper, target_tokens=400)
    assert len(chunks) == 1
    assert chunks[0].anchor == "S1.p1"
    assert chunks[0].url.endswith("#S1.p1")


def test_never_crosses_section_boundary():
    paper = _paper(
        [
            Section(anchor="S1", path=["1 Intro"], blocks=[_para("S1.p1", 200)]),
            Section(anchor="S2", path=["2 Method"], blocks=[_para("S2.p1", 200)]),
        ]
    )
    chunks = chunk_paper(paper, target_tokens=400)
    assert [c.section for c in chunks] == ["1 Intro", "2 Method"]
    assert [c.chunk_id for c in chunks] == ["2401.00001#0", "2401.00001#1"]


def test_splits_to_target_size():
    paper = _paper(
        [Section(anchor="S1", path=["1 Intro"], blocks=[_para(f"S1.p{i}", 1000) for i in range(6)])]
    )
    chunks = chunk_paper(paper, target_tokens=400, max_tokens=600)
    assert all(approx_tokens(c.content) <= 600 for c in chunks)
    assert len(chunks) >= 3


def test_long_paragraph_split_by_sentences():
    paper = _paper([Section(anchor="S1", path=["1 Intro"], blocks=[_para("S1.p1", 6000)])])
    chunks = chunk_paper(paper, target_tokens=400, max_tokens=600)
    assert len(chunks) > 1
    assert all(c.anchor == "S1.p1" for c in chunks)
    assert all(approx_tokens(c.content) <= 600 for c in chunks)


def test_tables_are_own_chunks_and_big_tables_split():
    rows = "\n".join(f"| model{i} | {i} |" for i in range(400))
    table = Block(anchor="S1.T1", kind="table", text=f"Table 1: Big.\n| a | b |\n|---|---|\n{rows}")
    paper = _paper(
        [
            Section(
                anchor="S1",
                path=["1 Intro"],
                blocks=[_para("S1.p1", 200), table, _para("S1.p2", 200)],
            )
        ]
    )
    chunks = chunk_paper(paper, target_tokens=400, max_tokens=600)
    kinds = [c.kind for c in chunks]
    assert kinds[0] == "text" and kinds[-1] == "text"
    table_chunks = [c for c in chunks if c.kind == "table"]
    assert len(table_chunks) > 1
    assert all(c.content.startswith("Table 1: Big.") for c in table_chunks)
    assert all(approx_tokens(c.content) <= 600 for c in table_chunks)


def test_contextual_header():
    paper = _paper(
        [Section(anchor="S1", path=["1 Intro", "1.1 Motivation"], blocks=[_para("S1.p1", 100)])]
    )
    chunk = chunk_paper(paper)[0]
    assert chunk.index_text().startswith("Toy | 1 Intro > 1.1 Motivation\n")
    assert not chunk.index_text(contextual=False).startswith("Toy |")


def test_split_sentences():
    assert split_sentences("A b. C d? E (f). $x$ is y.") == ["A b.", "C d?", "E (f).", "$x$ is y."]
