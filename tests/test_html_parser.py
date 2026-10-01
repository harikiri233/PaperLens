"""HTML 解析测试：用一段仿 LaTeXML 结构的 HTML（不提交真实论文全文）。"""

from paperlens.ingest.html_parser import looks_usable, parse_html, table_to_markdown
from paperlens.ingest.schema import PaperMeta

HTML = """
<html><body><div class="ltx_page_main"><article class="ltx_document">
<h1 class="ltx_title ltx_title_document">Toy-RAG: A Test Paper</h1>
<div class="ltx_authors">Alice, Bob</div>
<div class="ltx_abstract" id="abstract1"><h6>Abstract</h6>
  <p class="ltx_p" id="abstract1.1">We propose Toy-RAG.<span class="ltx_note ltx_role_footnote">
  <sup class="ltx_note_mark">1</sup><span class="ltx_note_outer"><span class="ltx_note_content">
  <sup class="ltx_note_mark">1</sup><span class="ltx_tag ltx_tag_note">1</span>Code at example.org.
  </span></span></span></p>
</div>
<section class="ltx_section" id="S1"><h2 class="ltx_title">1 Introduction</h2>
  <div class="ltx_para" id="S1.p1"><p class="ltx_p">Retrieval helps <cite>(Lewis et al., 2020)</cite>.</p></div>
  <div class="ltx_para" id="S1.p2"><p class="ltx_p">Let <math alttext="\\mathcal{R}">R</math> be a retriever.</p></div>
</section>
<section class="ltx_section" id="S2"><h2 class="ltx_title">2 Method</h2>
  <section class="ltx_subsection" id="S2.SS1"><h3 class="ltx_title">2.1 Scoring</h3>
    <div class="ltx_para" id="S2.SS1.p1"><p class="ltx_p">We score with
      <math alttext="\\hbox{\\pgfpicture \\lxSVG@begingroup@ lots of drawing}">x</math>.</p>
      <ul class="ltx_itemize"><li class="ltx_item"><div class="ltx_para" id="S2.I1.i1.p1">
        <p class="ltx_p">first item</p></div></li></ul>
    </div>
    <figure class="ltx_table" id="S2.T1"><figcaption>Table 1: Results.</figcaption>
      <table><tr><th>Model</th><th>Acc</th></tr><tr><td>Toy|RAG</td><td>90</td></tr></table>
    </figure>
    <figure class="ltx_figure" id="S2.F1"><img src="x.png"/><figcaption>Figure 1: Overview.</figcaption></figure>
  </section>
</section>
<section class="ltx_section" id="Sx1"><h2 class="ltx_title">Acknowledgements</h2>
  <div class="ltx_para" id="Sx1.p1"><p class="ltx_p">Thanks to everyone.</p></div>
</section>
<section class="ltx_bibliography" id="bib"><h2>References</h2>
  <ul><li class="ltx_bibitem">Lewis et al. 2020.</li></ul>
</section>
<section class="ltx_appendix" id="A1"><h2 class="ltx_title">Appendix A Details</h2>
  <div class="ltx_para" id="A1.p1"><p class="ltx_p">More details here.</p></div>
</section>
</article></div></body></html>
"""


def _parse():
    return parse_html(HTML, PaperMeta(paper_id="2401.00001", version="v2", title=""))


def test_sections_paths_and_anchors():
    paper = _parse()
    assert paper.meta.title == "Toy-RAG: A Test Paper"
    anchors = [s.anchor for s in paper.sections]
    assert anchors == ["abstract1", "S1", "S2.SS1", "A1"]
    assert paper.sections[2].path == ["2 Method", "2.1 Scoring"]
    assert paper.sections[-1].is_appendix
    assert not paper.sections[1].is_appendix


def test_skips_bibliography_and_acknowledgements():
    text = paper_text = "\n".join(b.text for s in _parse().sections for b in s.blocks)
    assert "Thanks to everyone" not in paper_text
    assert "Lewis et al. 2020." not in text


def test_math_footnote_and_drawing():
    paper = _parse()
    blocks = {b.anchor: b for s in paper.sections for b in s.blocks}
    assert "$\\mathcal{R}$" in blocks["S1.p2"].text
    assert "(Footnote: Code at example.org.)" in blocks["abstract1.1"].text
    # TikZ 绘图命令不进入正文
    assert "[formula]" in blocks["S2.SS1.p1"].text
    assert "pgfpicture" not in blocks["S2.SS1.p1"].text
    # 列表里嵌套的段落并入外层段落，不重复成块
    assert "first item" in blocks["S2.SS1.p1"].text
    assert "S2.I1.i1.p1" not in blocks


def test_tables_and_figures_are_separate_blocks():
    paper = _parse()
    blocks = {b.anchor: b for s in paper.sections for b in s.blocks}
    assert blocks["S2.T1"].kind == "table"
    assert "| Model | Acc |" in blocks["S2.T1"].text
    assert "Toy\\|RAG" in blocks["S2.T1"].text  # 单元格里的竖线要转义
    assert blocks["S2.F1"].kind == "figure"
    assert blocks["S2.F1"].text == "Figure 1: Overview."
    assert "Table 1" not in blocks["S2.SS1.p1"].text


def test_urls_point_to_html_anchor():
    paper = _parse()
    assert paper.url("S2.T1") == "https://arxiv.org/html/2401.00001v2#S2.T1"


def test_looks_usable_rejects_tiny_documents():
    assert not looks_usable(_parse())


def test_table_without_rows_returns_caption():
    from bs4 import BeautifulSoup

    fig = BeautifulSoup("<figure><figcaption>Table 9: Empty.</figcaption></figure>", "lxml").figure
    assert table_to_markdown(fig) == "Table 9: Empty."
