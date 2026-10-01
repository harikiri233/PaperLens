"""arXiv HTML（LaTeXML 生成）解析。

结构约定（2026-09 实测 Self-RAG / Attention / RAG 三篇论文）：
- 章节：section.ltx_section / ltx_subsection / ltx_subsubsection / ltx_paragraph / ltx_appendix，
  标题是第一个子元素 h2–h6.ltx_title，id 如 S3、S3.SS2、A1。
- 段落：div.ltx_para（id 如 S3.SS2.p4）；列表项里还会嵌套 ltx_para，只取最外层。
- 表格：figure.ltx_table；图：figure.ltx_figure（只保留图注）。
- 公式：math[alttext] 保留为 $LaTeX$；脚注：.ltx_note；参考文献：section.ltx_bibliography。
"""

import copy
import re

from bs4 import BeautifulSoup, NavigableString, Tag

from paperlens.ingest.schema import Block, PaperMeta, ParsedPaper, Section

_SECTION_CLASSES = {
    "ltx_section",
    "ltx_subsection",
    "ltx_subsubsection",
    "ltx_paragraph",
    "ltx_appendix",
}
_SKIP_TITLES = re.compile(r"^(acknowledge?ments?|references|bibliography)\b", re.IGNORECASE)


def _classes(tag: Tag) -> set[str]:
    return set(tag.get("class") or [])


def _is_drawing(alt: str) -> bool:
    """公式里嵌了 TikZ 图时，LaTeXML 会把整段 pgf 绘图命令放进 alttext（实测一个公式 2 万多字符）。"""
    return len(alt) > 500 or "\\pgf" in alt or "lxSVG" in alt


def _norm(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # get_text(" ") 会在内联元素两侧补空格，这里把标点前的多余空格去掉
    return re.sub(r"\s+([,.;:!?)\]])", r"\1", text).replace("( ", "(").replace("[ ", "[")


def element_text(tag: Tag) -> str:
    """元素的纯文本：公式换成 $LaTeX$，脚注保留内容，去掉图表（它们单独成块）。"""
    tag = copy.copy(tag)
    for fig in tag.select("figure.ltx_table, figure.ltx_figure"):
        fig.decompose()
    for math in tag.select("math"):
        alt = re.sub(r"\s+", " ", math.get("alttext") or "").strip()
        if _is_drawing(alt):
            math.replace_with(NavigableString(" [formula] "))
        else:
            math.replace_with(NavigableString(f" ${alt}$ " if alt else " "))
    for note in tag.select(".ltx_note"):
        content = note.select_one(".ltx_note_content")
        if content is None:
            note.decompose()
            continue
        for mark in content.select(".ltx_note_mark, .ltx_tag_note"):
            mark.decompose()
        note.replace_with(NavigableString(f" (Footnote: {_norm(content.get_text(' '))}) "))
    return _norm(tag.get_text(" "))


def _heading(section: Tag) -> str:
    for child in section.children:
        if isinstance(child, Tag) and child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return element_text(child)
    return ""


def table_to_markdown(figure: Tag) -> str:
    caption_tag = figure.find("figcaption")
    caption = element_text(caption_tag) if caption_tag else ""
    rows: list[list[str]] = []
    for tr in figure.select("tr"):
        cells = [element_text(td) for td in tr.find_all(["td", "th"], recursive=False)]
        if any(cells):
            rows.append([c.replace("|", "\\|") for c in cells])
    if not rows:
        return caption
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return f"{caption}\n" + "\n".join(lines) if caption else "\n".join(lines)


class _Walker:
    def __init__(self) -> None:
        self.sections: list[Section] = []

    def _current(self, anchor: str, path: list[str], is_appendix: bool) -> Section:
        """返回当前章节；同一章节的连续内容追加到同一个 Section 里。"""
        if self.sections and self.sections[-1].anchor == anchor:
            return self.sections[-1]
        section = Section(anchor=anchor, path=path, is_appendix=is_appendix)
        self.sections.append(section)
        return section

    def add(self, block: Block, anchor: str, path: list[str], is_appendix: bool) -> None:
        if block.text:
            self._current(anchor, path, is_appendix).blocks.append(block)

    def walk(self, node: Tag, anchor: str, path: list[str], is_appendix: bool) -> None:
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            classes = _classes(child)
            if child.name == "section" and classes & _SECTION_CLASSES:
                title = _heading(child)
                if _SKIP_TITLES.match(re.sub(r"^[\dA-Z.]+\s+", "", title)):
                    continue
                self.walk(
                    child,
                    child.get("id") or anchor,
                    [*path, title] if title else path,
                    is_appendix or "ltx_appendix" in classes,
                )
            elif child.name == "section" and "ltx_bibliography" in classes:
                continue
            elif child.name == "div" and "ltx_para" in classes:
                self.add(
                    Block(anchor=child.get("id") or anchor, text=element_text(child)),
                    anchor,
                    path,
                    is_appendix,
                )
                # 段落里嵌着的表格 / 图单独成块
                for fig in child.select("figure.ltx_table, figure.ltx_figure"):
                    self._add_figure(fig, anchor, path, is_appendix)
            elif child.name == "figure" and classes & {"ltx_table", "ltx_figure"}:
                self._add_figure(child, anchor, path, is_appendix)
            elif child.name in ("div", "span", "article"):
                self.walk(child, anchor, path, is_appendix)

    def _add_figure(self, fig: Tag, anchor: str, path: list[str], is_appendix: bool) -> None:
        if fig.find_parent("figure") is not None:  # 子图：由外层 figure 处理
            return
        if "ltx_table" in _classes(fig):
            block = Block(anchor=fig.get("id") or anchor, kind="table", text=table_to_markdown(fig))
        else:
            caption = fig.find("figcaption", recursive=False) or fig.find("figcaption")
            text = element_text(caption) if caption else ""
            block = Block(anchor=fig.get("id") or anchor, kind="figure", text=text)
        self.add(block, anchor, path, is_appendix)


def parse_html(html: str, meta: PaperMeta) -> ParsedPaper:
    soup = BeautifulSoup(html, "lxml")
    if not meta.title:
        title_tag = soup.select_one("h1.ltx_title_document")
        if title_tag is not None:
            meta = meta.model_copy(update={"title": element_text(title_tag)})
    article = soup.select_one("article.ltx_document") or soup.body or soup
    walker = _Walker()

    abstract = article.select_one("div.ltx_abstract")
    if abstract is not None:
        paragraphs = abstract.select("p.ltx_p") or [abstract]
        for p in paragraphs:
            walker.add(
                Block(anchor=p.get("id") or "abstract", text=element_text(p)),
                abstract.get("id") or "abstract",
                ["Abstract"],
                False,
            )
        abstract.decompose()

    for tag in article.select(
        "div.ltx_authors, div.ltx_dates, div.ltx_classification, div.ltx_keywords, "
        "h1.ltx_title_document, section.ltx_bibliography"
    ):
        tag.decompose()
    walker.walk(article, "body", [], False)
    sections = [s for s in walker.sections if s.blocks]
    return ParsedPaper(meta=meta, source="html", sections=sections)


def looks_usable(paper: ParsedPaper) -> bool:
    """HTML 解析结果是否可用：段落太少说明转换失败或只有摘要，需要回退到 PDF。"""
    return paper.count("text") >= 8 and paper.n_chars >= 5000
