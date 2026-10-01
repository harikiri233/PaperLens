from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from paperlens.ingest.arxiv_client import (
    ArxivClient,
    ArxivError,
    build_query,
    parse_arxiv_id,
    parse_feed,
)


def _feed(*entries: tuple[str, str]) -> str:
    body = "".join(
        f"""<entry><id>http://arxiv.org/abs/{pid}</id><title>{title}
        </title><summary>Abstract of {title}.</summary><published>2023-10-17T18:18:32Z</published>
        <author><name>Ada</name></author><author><name>Bob</name></author>
        <category term="cs.CL"/><category term="cs.AI"/></entry>"""
        for pid, title in entries
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{body}</feed>'


@pytest.fixture
def fast(settings):
    settings.arxiv_delay_s = 0
    return settings


def test_parse_arxiv_id():
    assert parse_arxiv_id("2310.11511") == ("2310.11511", "")
    assert parse_arxiv_id("https://arxiv.org/abs/2310.11511v2") == ("2310.11511", "v2")
    assert parse_arxiv_id("arxiv.org/pdf/cs/0112017v1") == ("cs/0112017", "v1")
    with pytest.raises(ArxivError):
        parse_arxiv_id("not an id")


def test_parse_feed():
    papers = parse_feed(_feed(("2310.11511v1", "Self-RAG: Learning")))
    p = papers[0]
    assert (p.paper_id, p.version, p.title) == ("2310.11511", "v1", "Self-RAG: Learning")
    assert p.authors == ["Ada", "Bob"]
    assert p.categories == ["cs.CL", "cs.AI"]
    assert p.published == "2023-10-17" and p.year == 2023


def test_build_query_quotes_phrases_and_strips_syntax():
    assert build_query(["RAG", "dense retrieval"]) == 'all:RAG AND all:"dense retrieval"'
    assert build_query(['a"b', "c:d"], op="OR") == 'all:"a b" OR all:"c d"'


def test_fetch_meta_keeps_order_and_reports_missing(fast):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=_feed(("2005.11401v4", "RAG"), ("2310.11511v1", "Self-RAG"))
        )

    with ArxivClient(fast, transport=httpx.MockTransport(handler)) as arxiv:
        metas = arxiv.fetch_meta(["2310.11511", "2005.11401v4"])
        assert [m.title for m in metas] == ["Self-RAG", "RAG"]
        with pytest.raises(ArxivError, match="1706.03762"):
            arxiv.fetch_meta(["1706.03762"])


def test_search_keywords_falls_back_to_or(fast):
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        q = parse_qs(urlparse(str(request.url)).query)["search_query"][0]
        queries.append(q)
        return httpx.Response(200, text=_feed() if " AND " in q else _feed(("2401.00001v1", "Hit")))

    with ArxivClient(fast, transport=httpx.MockTransport(handler)) as arxiv:
        found = arxiv.search_keywords(["rag evaluation", "benchmark"])
    assert [p.title for p in found] == ["Hit"]
    assert queries == [
        'all:"rag evaluation" AND all:benchmark',
        'all:"rag evaluation" OR all:benchmark',
    ]


def test_fetch_html_missing_and_cache(fast):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.startswith("/html/2401.00002"):
            return httpx.Response(404)
        return httpx.Response(200, text='<div class="ltx_page_main">ok</div>')

    with ArxivClient(fast, transport=httpx.MockTransport(handler)) as arxiv:
        assert arxiv.fetch_html("2401.00002") is None
        assert "ltx_page_main" in arxiv.fetch_html("2401.00001", "v1")
        arxiv.fetch_html("2401.00001", "v1")  # 第二次读缓存
    assert calls.count("/html/2401.00001v1") == 1


def test_fetch_pdf_rejects_non_pdf(fast):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text="<html>captcha</html>"))
    with ArxivClient(fast, transport=transport) as arxiv, pytest.raises(ArxivError):
        arxiv.fetch_pdf("2401.00001")
    assert not (fast.raw_dir / "2401.00001.pdf").exists()
