"""存储测试：SQLite 与 Milvus Lite（临时库文件，4 维向量）。"""

import pytest

from paperlens.ingest.schema import Chunk, PaperMeta
from paperlens.store import Database, VectorStore
from paperlens.store.vectors import paper_filter


def _meta(pid: str, title: str) -> PaperMeta:
    return PaperMeta(paper_id=pid, version="v1", title=title, published="2023-10-17")


def test_database_upsert_find_and_cards(tmp_path):
    db = Database(tmp_path / "p.db")
    db.upsert_paper(_meta("2310.11511", "Self-RAG: Learning to Retrieve"), "html", 78)
    db.upsert_paper(_meta("2401.15884", "Corrective Retrieval Augmented Generation"), "pdf", 40)
    db.upsert_paper(_meta("2310.11511", "Self-RAG: Learning to Retrieve"), "html", 80)

    assert db.get_paper("2310.11511").n_chunks == 80
    assert len(db.list_papers()) == 2
    assert db.find_papers("self rag")[0].meta.paper_id == "2310.11511"  # 忽略大小写与连字符
    assert db.find_papers("SELF-RAG")[0].meta.paper_id == "2310.11511"
    assert db.find_papers("corrective retrieval")[0].meta.paper_id == "2401.15884"
    assert db.find_papers("2401.15884")[0].meta.paper_id == "2401.15884"
    assert db.find_papers("GraphRAG") == []
    assert db.find_papers("x") == []

    db.save_card("2310.11511", {"method": "reflection tokens"}, "deepseek-flash")
    assert db.get_card("2310.11511") == {"method": "reflection tokens"}
    assert db.delete_paper("2310.11511")
    assert db.get_card("2310.11511") is None  # 外键级联删除
    db.close()


def test_paper_filter_escapes_quotes():
    assert paper_filter(None) == ""
    assert paper_filter(['a"b']) == 'paper_id in ["a\\"b"]'


def _chunk(pid: str, i: int, content: str) -> Chunk:
    return Chunk(
        chunk_id=f"{pid}#{i}",
        paper_id=pid,
        index=i,
        kind="text",
        title=pid,
        section="1 Intro",
        anchor=f"S1.p{i}",
        url="",
        content=content,
    )


@pytest.fixture
def store(tmp_path):
    vs = VectorStore(tmp_path / "milvus.db", "chunks", dim=4)
    chunks = [
        _chunk("a", 0, "Self-RAG trains reflection tokens to decide retrieval"),
        _chunk("a", 1, "We evaluate on PopQA and TriviaQA"),
        _chunk("b", 0, "PPO clips the probability ratio for policy updates"),
    ]
    vectors = [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0]]
    vs.insert(chunks, vectors, [c.index_text() for c in chunks])
    yield vs
    vs.close()


def test_dense_bm25_and_hybrid(store):
    assert store.count() == 3
    dense = store.dense_search([1, 0, 0, 0], limit=2)
    assert [h.chunk.chunk_id for h in dense] == ["a#0", "a#1"]

    bm25 = store.bm25_search("policy ratio", limit=3)
    assert bm25[0].chunk.chunk_id == "b#0"

    # 稠密指向 a，BM25 指向 b：RRF 融合后两边的第一名都在前两位
    fused = store.hybrid_search([1, 0, 0, 0], "policy ratio", candidates=3, limit=3)
    assert {h.chunk.chunk_id for h in fused[:2]} == {"a#0", "b#0"}


def test_filter_and_delete(store):
    hits = store.dense_search([0, 1, 0, 0], limit=3, paper_ids=["a"])
    assert {h.chunk.paper_id for h in hits} == {"a"}
    assert [c.chunk_id for c in store.paper_chunks("a")] == ["a#0", "a#1"]
    store.delete_paper("a")
    assert store.count() == 1


def test_reopen_loads_collection(tmp_path):
    """实测过的坑：重新打开库文件后 collection 是 released 状态，不 load 会检索失败。"""
    path = tmp_path / "milvus.db"
    vs = VectorStore(path, "chunks", dim=4)
    vs.insert([_chunk("a", 0, "hello world")], [[1, 0, 0, 0]], ["hello world"])
    vs.close()
    reopened = VectorStore(path, "chunks", dim=4)
    assert reopened.dense_search([1, 0, 0, 0], limit=1)[0].chunk.chunk_id == "a#0"
    reopened.close()
