"""PDF 解析：arXiv 没有 HTML 版时的回退，也用于用户上传的本地 PDF。

用 pymupdf4llm 按页转 Markdown，再按标题行切章节；每个段落的锚点是页码（page=5），
引用链接只能跳到 PDF 的对应页。参考文献之后的内容全部丢弃。
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

import pymupdf4llm

from paperlens.ingest.schema import Block, PaperMeta, ParsedPaper, Section

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_BOLD_HEADING = re.compile(r"^\*\*(\d+(?:\.\d+)*\.?\s+[A-Z][^*]{2,80})\*\*\s*$")
_REFERENCES = re.compile(r"^(references|bibliography)$", re.IGNORECASE)
_SKIP = re.compile(r"^acknowledge?ments?$", re.IGNORECASE)
_NUMBER = re.compile(r"^([\dA-H]+(?:\.\d+)*)\.?\s")


def _clean_heading(text: str) -> str:
    return re.sub(r"[*_`]+", "", text).strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"<sup>.*?</sup>", "", text)
    text = re.sub(r"(?<!\*)\*\*(?!\*)|(?<!_)_(?!_)", "", text)  # 去掉粗体 / 斜体标记
    return re.sub(r"[ \t]+", " ", text).strip()


def local_paper_id(path: Path) -> str:
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"local-{digest}"


def _repeated_lines(pages: list[str]) -> set[str]:
    """出现在一半以上页面的短行视为页眉页脚（如 "Preprint."、会议名）。"""
    counter: Counter[str] = Counter()
    for text in pages:
        lines = {line.strip() for line in text.splitlines() if 0 < len(line.strip()) < 80}
        counter.update(lines)
    threshold = max(3, len(pages) // 2)
    return {line for line, n in counter.items() if n >= threshold}


def parse_pdf(path: Path, meta: PaperMeta | None = None) -> ParsedPaper:
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    texts = [p["text"] for p in pages]
    noise = _repeated_lines(texts)

    sections: list[Section] = []
    path_stack: list[tuple[str, str]] = []  # (编号, 标题)；没有编号的标题编号为空
    in_appendix = False
    skipping = True  # 第一个标题（论文题目）与摘要之间是作者信息，不入库
    first_heading = ""
    done = False

    def push(title: str) -> None:
        """按编号维护章节路径："3.2 xxx" 挂在 "3 xxx" 下；上级标题漏识别时也不会挂错。"""
        nonlocal path_stack
        match = _NUMBER.match(title)
        number = match.group(1) if match else ""
        if not number:
            path_stack = [("", title)]
            return
        parent = number.rsplit(".", 1)[0] if "." in number else ""
        kept = []
        for num, t in path_stack:
            if num and parent and (parent == num or parent.startswith(num + ".")):
                kept.append((num, t))
        path_stack = [*kept, (number, title)]

    def current(page: int) -> Section:
        section_path = [t for _, t in path_stack]
        if sections and sections[-1].path == section_path:
            return sections[-1]
        anchor = f"page={page}"
        sections.append(Section(anchor=anchor, path=section_path, is_appendix=in_appendix))
        return sections[-1]

    for page_no, text in enumerate(texts, start=1):
        if done:
            break
        for para in re.split(r"\n\s*\n", text):
            lines = [line for line in para.splitlines() if line.strip() not in noise]
            if not lines:
                continue
            head = _HEADING.match(lines[0]) or _BOLD_HEADING.match(lines[0])
            if head:
                title = _clean_heading(head.group(head.lastindex or 1))
                if not first_heading:
                    first_heading = title
                    lines = lines[1:]
                    if not lines:
                        continue
                    head = None
                elif _REFERENCES.match(title):
                    done = True
                    break
                else:
                    skipping = bool(_SKIP.match(title))
                    if title.lower().startswith("appendix"):
                        in_appendix = True
                    if not skipping:
                        push(title)
                    lines = lines[1:]
                    if not lines:
                        continue
            if skipping:
                continue
            body = _clean_text("\n".join(lines))
            if len(body) < 20:  # 页码、孤立的图表编号等
                continue
            kind = "table" if body.lstrip().startswith("|") else "text"
            current(page_no).blocks.append(Block(anchor=f"page={page_no}", kind=kind, text=body))

    if meta is None:
        meta = PaperMeta(
            paper_id=local_paper_id(path),
            title=first_heading.title() if first_heading.isupper() else first_heading or path.stem,
        )
    elif not meta.title:
        meta = meta.model_copy(update={"title": first_heading or path.stem})
    return ParsedPaper(meta=meta, source="pdf", sections=[s for s in sections if s.blocks])
