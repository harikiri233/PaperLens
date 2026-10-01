"""入库流程：获取 → 解析 → 分块 → 嵌入 → 写入 Milvus 与 SQLite。

- 幂等：以 arXiv 基础 ID 为主键，已入库的论文默认跳过；force=True 或发现新版本时先删旧块再入库。
- 解析结果保存为 data/parsed/{paper_id}.json，生成解读卡片时直接读取全文。
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from paperlens.ingest.arxiv_client import ArxivClient
from paperlens.ingest.chunker import chunk_paper
from paperlens.ingest.html_parser import looks_usable, parse_html
from paperlens.ingest.pdf_parser import parse_pdf
from paperlens.ingest.schema import PaperMeta, ParsedPaper
from paperlens.services import Services

Status = Literal["added", "updated", "skipped", "failed"]


@dataclass
class IngestResult:
    paper_id: str
    title: str
    status: Status
    source: str = ""
    n_chunks: int = 0
    message: str = ""


def parsed_path(services: Services, paper_id: str) -> Path:
    return services.settings.parsed_dir / f"{paper_id.replace('/', '_')}.json"


def load_parsed(services: Services, paper_id: str) -> ParsedPaper | None:
    path = parsed_path(services, paper_id)
    if not path.exists():
        return None
    return ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))


def fetch_and_parse(arxiv: ArxivClient, meta: PaperMeta) -> ParsedPaper:
    """HTML 优先；没有 HTML 版或解析结果太少时回退到 PDF。"""
    html = arxiv.fetch_html(meta.paper_id, meta.version)
    if html is not None:
        paper = parse_html(html, meta)
        if looks_usable(paper):
            return paper
    return parse_pdf(arxiv.fetch_pdf(meta.paper_id, meta.version), meta)


def index_paper(services: Services, paper: ParsedPaper) -> int:
    """分块、嵌入并写入存储，返回块数。调用前旧数据需要已经删除。"""
    settings = services.settings
    chunks = chunk_paper(
        paper,
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
    )
    if not chunks:
        raise ValueError("解析结果里没有可入库的内容")
    texts = [c.index_text() for c in chunks]
    vectors = services.sf.embed(texts)
    services.vectors.insert(chunks, vectors, texts)

    path = parsed_path(services, paper.meta.paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paper.model_dump_json(), encoding="utf-8")
    services.db.upsert_paper(paper.meta, paper.source, len(chunks))
    return len(chunks)


def remove_paper(services: Services, paper_id: str) -> None:
    services.vectors.delete_paper(paper_id)
    services.db.delete_paper(paper_id)
    parsed_path(services, paper_id).unlink(missing_ok=True)


def ingest_arxiv(
    services: Services,
    ids: list[str],
    *,
    force: bool = False,
    arxiv: ArxivClient | None = None,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    own_client = arxiv is None
    arxiv = arxiv or ArxivClient(services.settings)
    results: list[IngestResult] = []

    def report(result: IngestResult) -> None:
        results.append(result)
        if on_result:
            on_result(result)

    try:
        metas = arxiv.fetch_meta(ids)
        for meta in metas:
            existing = services.db.get_paper(meta.paper_id)
            if existing and not force and existing.meta.version == meta.version:
                report(
                    IngestResult(
                        meta.paper_id,
                        meta.title,
                        "skipped",
                        existing.source,
                        existing.n_chunks,
                        "已在库中",
                    )
                )
                continue
            try:
                paper = fetch_and_parse(arxiv, meta)
                if existing:
                    services.vectors.delete_paper(meta.paper_id)
                n = index_paper(services, paper)
                status: Status = "updated" if existing else "added"
                report(IngestResult(meta.paper_id, meta.title, status, paper.source, n))
            except Exception as exc:  # 单篇失败不影响其余论文
                report(
                    IngestResult(
                        meta.paper_id, meta.title, "failed", message=f"{type(exc).__name__}: {exc}"
                    )
                )
    finally:
        if own_client:
            arxiv.close()
    return results


def ingest_pdf(services: Services, path: Path, *, title: str = "") -> IngestResult:
    """入库本地 PDF。paper_id 由文件内容的哈希生成，同一个文件重复上传会覆盖。"""
    paper = parse_pdf(path)
    if title:
        paper.meta.title = title
    existing = services.db.get_paper(paper.meta.paper_id)
    if existing:
        services.vectors.delete_paper(paper.meta.paper_id)
    n = index_paper(services, paper)
    return IngestResult(
        paper.meta.paper_id, paper.meta.title, "updated" if existing else "added", "pdf", n
    )
