"""分块：章节内合并相邻段落到目标长度，超长段落按句子切分；不跨章节。

- 表格、图注各自独立成块，不和正文合并。
- 块的锚点取第一个段落的锚点，引用时跳到这一段。
- token 数按字符数 / 4 近似（英文论文），不引入分词器。
"""

import re

from paperlens.ingest.schema import Block, Chunk, ParsedPaper, Section

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[$])")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text) if s.strip()]


def _split_long(text: str, max_tokens: int, overlap: int = 1) -> list[str]:
    """把超长文本按句子切成不超过 max_tokens 的片段，相邻片段重叠 overlap 句。"""
    sentences = split_sentences(text)
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        # 单句就超长（如大表格、公式堆）：按字符硬切
        if approx_tokens(sentence) > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []
            step = max_tokens * 4
            pieces.extend(sentence[i : i + step] for i in range(0, len(sentence), step))
            continue
        if current and approx_tokens(" ".join([*current, sentence])) > max_tokens:
            pieces.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_paper(
    paper: ParsedPaper, *, target_tokens: int = 400, max_tokens: int = 600
) -> list[Chunk]:
    meta = paper.meta
    chunks: list[Chunk] = []

    def emit(section: Section, block: Block, text: str) -> None:
        """block 是这个块的第一个段落，决定锚点和类型。"""
        chunks.append(
            Chunk(
                chunk_id=f"{meta.paper_id}#{len(chunks)}",
                paper_id=meta.paper_id,
                index=len(chunks),
                kind=block.kind,
                title=meta.title,
                section=section.path_str,
                anchor=block.anchor,
                url=paper.url(block.anchor),
                content=text,
                is_appendix=section.is_appendix,
            )
        )

    for section in paper.sections:
        buffer: list[Block] = []
        for block in section.blocks:
            is_text = block.kind == "text"
            too_long = approx_tokens(block.text) > max_tokens
            merged = "\n\n".join([*(b.text for b in buffer), block.text])
            # 遇到表格 / 图注、超长段落，或合并后超过目标长度时，先把缓冲区输出成一个块
            if buffer and (not is_text or too_long or approx_tokens(merged) > target_tokens):
                emit(section, buffer[0], "\n\n".join(b.text for b in buffer))
                buffer = []
            if is_text and not too_long:
                buffer.append(block)
            elif is_text:
                for piece in _split_long(block.text, target_tokens):
                    emit(section, block, piece)
            else:
                for piece in _split_table(block.text, max_tokens):
                    emit(section, block, piece)
        if buffer:
            emit(section, buffer[0], "\n\n".join(b.text for b in buffer))
    return chunks


def _split_table(text: str, max_tokens: int) -> list[str]:
    """大表格按行切，每一片都带上表注、表头和分隔行（前 3 行）。"""
    if approx_tokens(text) <= max_tokens:
        return [text]
    lines = text.splitlines()
    head, rows = lines[:3], lines[3:]
    size = max(1, len(rows) // (approx_tokens(text) // max_tokens + 1))
    pieces: list[str] = []
    for i in range(0, len(rows), size):
        piece = "\n".join(head + rows[i : i + size])
        # 表注特别长或单行特别宽时，仍可能超长，按字符硬切兜底
        step = max_tokens * 4
        pieces.extend(piece[j : j + step] for j in range(0, len(piece), step))
    return pieces
