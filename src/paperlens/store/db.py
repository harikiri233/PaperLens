"""SQLite：论文元数据与解读卡片。

只有两张表，直接用标准库 sqlite3。所有查询都用参数化语句。
"""

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from paperlens.ingest.schema import PaperMeta, Source

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id    TEXT PRIMARY KEY,
    version     TEXT NOT NULL DEFAULT '',
    title       TEXT NOT NULL,
    authors     TEXT NOT NULL DEFAULT '[]',
    abstract    TEXT NOT NULL DEFAULT '',
    categories  TEXT NOT NULL DEFAULT '[]',
    published   TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    added_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cards (
    paper_id    TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    card        TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


class PaperRecord(BaseModel):
    meta: PaperMeta
    source: Source
    n_chunks: int
    added_at: str


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：FastAPI 的后台任务在其他线程里访问
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self):
        # Web 服务里问答、入库、生成卡片在不同线程共用这个连接，写事务串行执行
        with self._lock, self._conn:
            yield self._conn

    # ---- papers ----

    def upsert_paper(self, meta: PaperMeta, source: Source, n_chunks: int) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO papers (paper_id, version, title, authors, abstract, categories,
                                    published, source, n_chunks, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    version=excluded.version, title=excluded.title, authors=excluded.authors,
                    abstract=excluded.abstract, categories=excluded.categories,
                    published=excluded.published, source=excluded.source,
                    n_chunks=excluded.n_chunks
                """,
                (
                    meta.paper_id,
                    meta.version,
                    meta.title,
                    json.dumps(meta.authors, ensure_ascii=False),
                    meta.abstract,
                    json.dumps(meta.categories),
                    meta.published,
                    source,
                    n_chunks,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> PaperRecord:
        return PaperRecord(
            meta=PaperMeta(
                paper_id=row["paper_id"],
                version=row["version"],
                title=row["title"],
                authors=json.loads(row["authors"]),
                abstract=row["abstract"],
                categories=json.loads(row["categories"]),
                published=row["published"],
            ),
            source=row["source"],
            n_chunks=row["n_chunks"],
            added_at=row["added_at"],
        )

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        row = self._conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return self._record(row) if row else None

    def list_papers(self) -> list[PaperRecord]:
        rows = self._conn.execute("SELECT * FROM papers ORDER BY added_at DESC, paper_id")
        return [self._record(r) for r in rows]

    def delete_paper(self, paper_id: str) -> bool:
        with self._tx() as conn:
            return conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,)).rowcount > 0

    def _scored_matches(self, mention: str) -> list[tuple[int, PaperRecord]]:
        mention = mention.strip()
        if not mention:
            return []
        exact = self.get_paper(mention)
        if exact:
            return [(0, exact)]
        key = _normalize(mention)
        if len(key) < 3:
            return []
        scored: list[tuple[int, PaperRecord]] = []
        for record in self.list_papers():
            title = record.meta.title
            short_part = title.split(":", 1)[0]
            if key == _normalize(short_part):
                scored.append((0, record))
            elif _word_substring(key, short_part) or key == _acronym(title):
                scored.append((1, record))
            elif _word_substring(key, title):
                scored.append((2, record))
        scored.sort(key=lambda x: x[0])
        return scored

    def find_papers(self, mention: str, *, limit: int = 3) -> list[PaperRecord]:
        """按标题模糊匹配论文，用于把问题里点名的论文（如 "Self-RAG"）解析成 ID。

        先精确匹配 ID，再匹配标题冒号前的系统名或首字母缩写（"Corrective Retrieval Augmented
        Generation" → CRAG），最后做子串匹配；忽略大小写和连字符。
        子串必须从单词开头开始匹配：否则 "RAG" 会命中 "Leve-rag-ing Passage Retrieval…"（评估时发现）。
        """
        return [r for _, r in self._scored_matches(mention)[:limit]]

    def match_paper(self, mention: str) -> tuple[PaperRecord | None, bool]:
        """返回 (最匹配的论文, 是否有歧义)。

        最佳档次有多篇论文时视为泛称（如库里有五篇标题带 RAG 的论文时说 "RAG"），
        调用方不应据此限定检索范围，也不应当成"库里没有"去联网补充。
        """
        scored = self._scored_matches(mention)
        if not scored:
            return None, False
        best = scored[0][0]
        ambiguous = best > 0 and sum(1 for tier, _ in scored if tier == best) > 1
        return (None if ambiguous else scored[0][1]), ambiguous

    # ---- cards ----

    def get_card(self, paper_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT card FROM cards WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return json.loads(row["card"]) if row else None

    def save_card(self, paper_id: str, card: dict, model: str) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO cards (paper_id, card, model, created_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    card=excluded.card, model=excluded.model, created_at=excluded.created_at
                """,
                (
                    paper_id,
                    json.dumps(card, ensure_ascii=False),
                    model,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )


def _acronym(title: str) -> str:
    """标题冒号前部分中大写开头单词的首字母，小写返回。"""
    words = re.split(r"[\s\-]+", title.split(":", 1)[0])
    return "".join(w[0] for w in words if w[:1].isupper()).lower()


def _word_substring(key: str, text: str) -> bool:
    """key（已规范化）是否从 text 的某个单词开头起出现，可以跨越多个单词（"selfrag" 匹配 "Self-RAG"）。"""
    words = [_normalize(w) for w in re.split(r"[^0-9A-Za-z]+", text) if w]
    return any("".join(words[i:]).startswith(key) for i in range(len(words)))


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())
