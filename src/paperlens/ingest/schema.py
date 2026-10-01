"""论文的统一数据结构：元数据、解析结果（HTML 和 PDF 共用）与分块。"""

import re
from typing import Literal

from pydantic import BaseModel, Field

BlockKind = Literal["text", "table", "figure"]
Source = Literal["html", "pdf"]


class PaperMeta(BaseModel):
    paper_id: str  # arXiv 基础 ID（不带版本号）；本地上传的 PDF 为 local-xxxxxxxxxx
    version: str = ""  # 如 "v1"；本地 PDF 为空
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    categories: list[str] = Field(default_factory=list)
    published: str = ""  # YYYY-MM-DD

    @property
    def is_arxiv(self) -> bool:
        return not self.paper_id.startswith("local-")

    @property
    def year(self) -> int | None:
        return int(self.published[:4]) if self.published[:4].isdigit() else None

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.paper_id}{self.version}" if self.is_arxiv else ""


def source_url(paper_id: str, version: str, source: Source, anchor: str) -> str:
    """原文跳转链接：HTML 跳到段落锚点，PDF 跳到页码；本地 PDF 没有在线链接。"""
    if paper_id.startswith("local-"):
        return ""
    if source == "html":
        base = f"https://arxiv.org/html/{paper_id}{version}"
    else:
        base = f"https://arxiv.org/pdf/{paper_id}{version}"
    return f"{base}#{anchor}" if anchor else base


class Block(BaseModel):
    anchor: str  # HTML 元素 id（如 S3.SS2.p4、S3.T1）；PDF 为 "page=5"
    kind: BlockKind = "text"
    text: str


class Section(BaseModel):
    anchor: str
    path: list[str]  # ["3 Method", "3.2 Retrieval Tokens"]
    is_appendix: bool = False
    blocks: list[Block] = Field(default_factory=list)

    @property
    def path_str(self) -> str:
        return " > ".join(self.path) if self.path else "正文"


class ParsedPaper(BaseModel):
    meta: PaperMeta
    source: Source
    sections: list[Section]

    def url(self, anchor: str) -> str:
        return source_url(self.meta.paper_id, self.meta.version, self.source, anchor)

    def count(self, kind: BlockKind) -> int:
        return sum(1 for s in self.sections for b in s.blocks if b.kind == kind)

    @property
    def n_formulas(self) -> int:
        """正文中以 $...$ 保留的 LaTeX 公式个数（只统计 HTML 来源）。"""
        return sum(len(re.findall(r"\$[^$]+\$", b.text)) for s in self.sections for b in s.blocks)

    @property
    def n_chars(self) -> int:
        return sum(len(b.text) for s in self.sections for b in s.blocks)

    def full_text(self, *, include_appendix: bool = False) -> str:
        """按章节拼接的全文，用于长上下文场景（解读卡片）。"""
        parts: list[str] = []
        for section in self.sections:
            if section.is_appendix and not include_appendix:
                continue
            parts.append(f"## {section.path_str} [#{section.anchor}]")
            parts.extend(b.text for b in section.blocks)
        return "\n\n".join(parts)

    def to_markdown(self) -> str:
        m = self.meta
        lines = [f"# {m.title}", ""]
        if m.authors:
            lines.append(f"作者：{', '.join(m.authors)}  ")
        lines.append(f"来源：{self.source}  ")
        if m.is_arxiv:
            lines.append(f"链接：{m.abs_url}")
        lines.append("")
        for section in self.sections:
            depth = min(len(section.path), 4) or 1
            tag = "（附录）" if section.is_appendix else ""
            lines += [
                f"{'#' * (depth + 1)} {section.path[-1] if section.path else '正文'}{tag}",
                "",
            ]
            for block in section.blocks:
                label = {"table": "**[表格]** ", "figure": "**[图注]** "}.get(block.kind, "")
                lines += [f"{label}{block.text}", "", f"<sub>#{block.anchor}</sub>", ""]
        return "\n".join(lines)


class Chunk(BaseModel):
    chunk_id: str  # {paper_id}#{序号}
    paper_id: str
    index: int
    kind: BlockKind
    title: str
    section: str  # 章节路径，如 "3 Method > 3.2 Retrieval Tokens"
    anchor: str
    url: str
    content: str
    is_appendix: bool = False

    @property
    def header(self) -> str:
        """上下文标题：告诉嵌入模型和 BM25 这段话出自哪篇论文的哪一部分。"""
        return f"{self.title} | {self.section}"

    def index_text(self, *, contextual: bool = True) -> str:
        return f"{self.header}\n{self.content}" if contextual else self.content
