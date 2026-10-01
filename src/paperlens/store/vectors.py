"""Milvus：稠密向量 + 内置 BM25 稀疏向量，一个 collection 完成混合检索。

实测（pymilvus 3.0.2 + milvus-lite 3.2.1）：
- BM25 Function 由 text 字段自动生成 sparse 向量，插入时不需要传 sparse。
- Lite 版分词器只支持 standard / jieba，不支持 english 词干化。
- 同一个库文件只能被一个进程打开。
- 重新打开库文件后 collection 是 released 状态，要先 load_collection。
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from paperlens.ingest.schema import Chunk

_OUTPUT_FIELDS = [
    "paper_id",
    "idx",
    "kind",
    "title",
    "section",
    "anchor",
    "url",
    "content",
    "is_appendix",
]


@dataclass
class Hit:
    chunk: Chunk
    score: float


def _quote(value: str) -> str:
    """过滤表达式里的字符串字面量：用 JSON 转义，避免引号和反斜杠破坏表达式。"""
    return json.dumps(value)


def paper_filter(paper_ids: list[str] | None) -> str:
    if not paper_ids:
        return ""
    return f"paper_id in [{', '.join(_quote(p) for p in paper_ids)}]"


class VectorStore:
    def __init__(self, uri: Path | str, collection: str, dim: int) -> None:
        if isinstance(uri, Path):
            uri.parent.mkdir(parents=True, exist_ok=True)
            uri = str(uri)
        self.client = MilvusClient(uri)
        self.collection = collection
        self.dim = dim
        if not self.client.has_collection(collection):
            self._create()
        # 实测：新进程重新打开库文件后 collection 处于 released 状态，需要先 load 才能检索
        self.client.load_collection(collection)

    def close(self) -> None:
        self.client.close()

    def _create(self) -> None:
        schema = self.client.create_schema(auto_id=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("paper_id", DataType.VARCHAR, max_length=64)
        schema.add_field("idx", DataType.INT64)
        schema.add_field("kind", DataType.VARCHAR, max_length=16)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("section", DataType.VARCHAR, max_length=2048)
        schema.add_field("anchor", DataType.VARCHAR, max_length=256)
        schema.add_field("url", DataType.VARCHAR, max_length=512)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("is_appendix", DataType.BOOL)
        # BM25 的输入：上下文标题 + 正文
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "standard", "filter": ["lowercase"]},
        )
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        index = self.client.prepare_index_params()
        index.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        self.client.create_collection(self.collection, schema=schema, index_params=index)

    # ---- 写入 ----

    def insert(self, chunks: list[Chunk], vectors: list[list[float]], texts: list[str]) -> None:
        rows = [
            {
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "idx": c.index,
                "kind": c.kind,
                "title": c.title[:1000],
                "section": c.section[:2000],
                "anchor": c.anchor[:250],
                "url": c.url[:500],
                "content": c.content[:60000],
                "is_appendix": c.is_appendix,
                "text": t[:60000],
                "dense": v,
            }
            for c, v, t in zip(chunks, vectors, texts, strict=True)
        ]
        for start in range(0, len(rows), 256):
            self.client.insert(self.collection, rows[start : start + 256])

    def delete_paper(self, paper_id: str) -> None:
        self.client.delete(self.collection, filter=f"paper_id == {_quote(paper_id)}")

    def count(self) -> int:
        result = self.client.query(self.collection, filter="", output_fields=["count(*)"])
        return int(result[0]["count(*)"])

    def paper_chunks(self, paper_id: str) -> list[Chunk]:
        rows = self.client.query(
            self.collection,
            filter=f"paper_id == {_quote(paper_id)}",
            output_fields=["chunk_id", *_OUTPUT_FIELDS],
        )
        return sorted((self._chunk(r) for r in rows), key=lambda c: c.index)

    # ---- 检索 ----

    @staticmethod
    def _chunk(entity: dict) -> Chunk:
        return Chunk(
            chunk_id=entity["chunk_id"],
            paper_id=entity["paper_id"],
            index=entity["idx"],
            kind=entity["kind"],
            title=entity["title"],
            section=entity["section"],
            anchor=entity["anchor"],
            url=entity["url"],
            content=entity["content"],
            is_appendix=entity["is_appendix"],
        )

    def _hits(self, result) -> list[Hit]:
        hits = []
        for h in result[0]:
            # 实测：主键按字段名返回（h["chunk_id"]），entity 里也带一份；没有名为 "id" 的键
            entity = dict(h["entity"])
            entity.setdefault("chunk_id", h.get("chunk_id") or h.get("id"))
            hits.append(Hit(self._chunk(entity), float(h["distance"])))
        return hits

    def dense_search(
        self, vector: list[float], *, limit: int, paper_ids: list[str] | None = None
    ) -> list[Hit]:
        result = self.client.search(
            self.collection,
            data=[vector],
            anns_field="dense",
            limit=limit,
            filter=paper_filter(paper_ids),
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
        return self._hits(result)

    def bm25_search(
        self, query: str, *, limit: int, paper_ids: list[str] | None = None
    ) -> list[Hit]:
        result = self.client.search(
            self.collection,
            data=[query],
            anns_field="sparse",
            limit=limit,
            filter=paper_filter(paper_ids),
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "BM25"},
        )
        return self._hits(result)

    def hybrid_search(
        self,
        vector: list[float],
        query: str,
        *,
        candidates: int,
        limit: int,
        paper_ids: list[str] | None = None,
        rrf_k: int = 60,
    ) -> list[Hit]:
        """稠密 + BM25 两路检索，用 RRF 按排名融合（分数 = Σ 1 / (k + rank)）。"""
        expr = paper_filter(paper_ids)
        requests = [
            AnnSearchRequest(
                data=[vector], anns_field="dense", param={}, limit=candidates, expr=expr
            ),
            AnnSearchRequest(
                data=[query], anns_field="sparse", param={}, limit=candidates, expr=expr
            ),
        ]
        result = self.client.hybrid_search(
            self.collection,
            requests,
            RRFRanker(rrf_k),
            limit=limit,
            output_fields=_OUTPUT_FIELDS,
        )
        return self._hits(result)
