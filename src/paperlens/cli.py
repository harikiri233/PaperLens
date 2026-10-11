"""命令行入口。"""

import re
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from paperlens import __version__
from paperlens.config import get_settings
from paperlens.doctor import CheckResult, run_all
from paperlens.models import UsageTracker
from paperlens.models.usage import USD_TO_CNY

app = typer.Typer(help="PaperLens：论文检索与分析 Agent", no_args_is_help=True)
console = Console()

_STYLE = {
    "ok": ("[green]通过[/green]", "green"),
    "warn": ("[yellow]注意[/yellow]", "yellow"),
    "fail": ("[red]失败[/red]", "red"),
    "skip": ("[dim]跳过[/dim]", "dim"),
}


@app.callback()
def main() -> None:
    """PaperLens 命令行工具。"""


@app.command()
def version() -> None:
    """显示版本号。"""
    console.print(f"paperlens {__version__}")


def _print_usage(tracker: UsageTracker, title: str = "本次用量") -> None:
    usage = tracker.snapshot()
    if not usage:
        return
    table = Table(title=title, show_lines=False)
    for col in ("模型", "调用", "输入 tokens", "缓存命中", "输出 tokens", "费用"):
        table.add_column(col, justify="right" if col != "模型" else "left")
    for model, u in usage.items():
        cost = "未知单价" if u.cost_usd is None else f"${u.cost_usd:.6f}"
        table.add_row(
            model,
            str(u.calls),
            str(u.input_tokens),
            str(u.cached_input_tokens),
            str(u.output_tokens),
            cost,
        )
    console.print(table)
    total = tracker.total_cost_usd
    console.print(f"合计约 ${total:.6f}（约 ¥{total * USD_TO_CNY:.4f}）")


def _print_result(result: CheckResult) -> None:
    label, color = _STYLE[result.status]
    latency = f"  [dim]{result.latency_s:.2f}s[/dim]" if result.latency_s is not None else ""
    console.print(f"{label}  [bold]{result.name}[/bold]  {result.summary}{latency}")
    for line in result.details:
        console.print(f"      [{color}]│[/{color}] {line}", highlight=False)


@app.command()
def doctor(
    skip_arxiv: bool = typer.Option(False, "--skip-arxiv", help="跳过 arXiv 连通性检查"),
) -> None:
    """检查配置、LLM、Embedding、Rerank 与 arXiv 的连通性。"""
    settings = get_settings()
    tracker = UsageTracker()
    console.rule("PaperLens 环境自检")
    results = run_all(settings, tracker, skip_arxiv=skip_arxiv, on_result=_print_result)
    _print_usage(tracker, "本次自检的模型用量")

    failed = [r.name for r in results if r.status == "fail"]
    if failed:
        console.print(f"[red]有 {len(failed)} 项失败：{', '.join(failed)}[/red]")
        raise typer.Exit(code=1)
    console.print("[green]自检结束，没有失败项。[/green]")


# ---------------------------------------------------------------- M1：解析


@app.command()
def parse(
    target: str = typer.Argument(..., help="arXiv ID / 链接，或本地 PDF 路径"),
    output: Path | None = typer.Option(None, "--output", "-o", help="导出 Markdown 的路径"),
    show_blocks: bool = typer.Option(False, "--blocks", help="列出每个段落的锚点和开头"),
) -> None:
    """解析单篇论文（不入库），显示章节树与统计信息。"""
    from paperlens.ingest.arxiv_client import ArxivClient, ArxivError
    from paperlens.ingest.pdf_parser import parse_pdf
    from paperlens.ingest.pipeline import fetch_and_parse

    settings = get_settings()
    path = Path(target)
    try:
        if path.suffix.lower() == ".pdf" and path.exists():
            paper = parse_pdf(path)
        else:
            with ArxivClient(settings) as arxiv:
                meta = arxiv.fetch_meta([target])[0]
                console.print(f"[dim]已获取元数据：{meta.title}（{meta.published}）[/dim]")
                paper = fetch_and_parse(arxiv, meta)
    except ArxivError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    m = paper.meta
    console.rule(m.title)
    console.print(
        f"ID：{m.paper_id}{m.version}   来源：[bold]{paper.source}[/bold]   "
        f"作者：{', '.join(m.authors[:4])}{' 等' if len(m.authors) > 4 else ''}"
    )
    console.print(
        f"章节 {len(paper.sections)} 个，段落 {paper.count('text')}，表格 {paper.count('table')}，"
        f"图注 {paper.count('figure')}，公式 {paper.n_formulas}，正文约 {paper.n_chars // 4} tokens"
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("锚点", style="cyan", no_wrap=True)
    table.add_column("章节")
    table.add_column("块数", justify="right")
    for section in paper.sections:
        path_str = section.path_str + (" [dim]（附录）[/dim]" if section.is_appendix else "")
        table.add_row(section.anchor, path_str, str(len(section.blocks)))
    console.print(table)
    if show_blocks:
        for section in paper.sections:
            for block in section.blocks:
                console.print(
                    f"[cyan]{block.anchor}[/cyan] [dim]{block.kind}[/dim] {block.text[:100]}",
                    highlight=False,
                )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(paper.to_markdown(), encoding="utf-8")
        console.print(f"[green]已导出 Markdown：{output}[/green]")


# ---------------------------------------------------------------- M2：入库与检索


def _read_ids(ids: list[str], file: Path | None) -> list[str]:
    result = list(ids)
    if file:
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                result.append(line)
    return result


@app.command()
def ingest(
    ids: list[str] = typer.Argument(None, help="arXiv ID 或链接，可以写多个"),
    file: Path | None = typer.Option(None, "--file", "-f", help="每行一个 arXiv ID 的文件"),
    search: str | None = typer.Option(None, "--search", "-s", help="按关键词在 arXiv 检索后入库"),
    limit: int = typer.Option(5, "--limit", help="--search 时最多入库几篇"),
    pdf: Path | None = typer.Option(None, "--pdf", help="入库本地 PDF"),
    force: bool = typer.Option(False, "--force", help="已入库的论文也重新入库"),
) -> None:
    """论文入库：获取 → 解析 → 分块 → 嵌入 → 写入 Milvus 与 SQLite。"""
    from paperlens.ingest.arxiv_client import ArxivClient
    from paperlens.ingest.pipeline import IngestResult, ingest_arxiv, ingest_pdf
    from paperlens.services import Services

    settings = get_settings()
    colors = {"added": "green", "updated": "cyan", "skipped": "dim", "failed": "red"}
    labels = {"added": "入库", "updated": "更新", "skipped": "跳过", "failed": "失败"}

    def show(r: IngestResult) -> None:
        c = colors[r.status]
        extra = f"{r.source}，{r.n_chunks} 块" if r.n_chunks else ""
        msg = f"  {r.message}" if r.message else ""
        console.print(
            f"[{c}]{labels[r.status]}[/{c}]  {r.paper_id}  {r.title[:60]}  [dim]{extra}{msg}[/dim]",
            highlight=False,
        )

    with Services(settings) as services:
        if pdf:
            show(ingest_pdf(services, pdf))
        targets = _read_ids(ids or [], file)
        with ArxivClient(settings) as arxiv:
            if search:
                found = arxiv.search(search, max_results=limit)
                console.print(f"[dim]arXiv 检索到 {len(found)} 篇[/dim]")
                targets += [f"{m.paper_id}{m.version}" for m in found]
            if targets:
                console.print(
                    f"[dim]共 {len(targets)} 篇，arXiv 请求间隔 "
                    f"{settings.arxiv_delay_s:.0f} 秒[/dim]"
                )
                results = ingest_arxiv(services, targets, force=force, arxiv=arxiv, on_result=show)
                failed = sum(1 for r in results if r.status == "failed")
                console.print(
                    f"完成：{len(results) - failed} 篇成功，{failed} 篇失败；"
                    f"库中共 {services.vectors.count()} 个块"
                )
        if not pdf and not targets:
            console.print("[yellow]没有要入库的论文：请给出 ID、--file、--search 或 --pdf[/yellow]")
        _print_usage(services.tracker)


@app.command()
def papers() -> None:
    """列出知识库中的论文。"""
    from paperlens.store import Database

    settings = get_settings()
    db = Database(settings.db_path)
    records = db.list_papers()
    if not records:
        console.print("知识库是空的，先用 paperlens ingest 入库论文。")
        return
    table = Table(title=f"知识库：{len(records)} 篇论文")
    for col in ("ID", "标题", "年份", "来源", "块数"):
        table.add_column(col)
    for r in records:
        table.add_row(
            r.meta.paper_id, r.meta.title[:70], str(r.meta.year or ""), r.source, str(r.n_chunks)
        )
    console.print(table)


@app.command()
def chunks(
    paper_id: str = typer.Argument(..., help="论文 ID"),
    full: bool = typer.Option(False, "--full", help="显示完整内容"),
) -> None:
    """查看一篇论文的分块结果。"""
    from paperlens.services import Services

    with Services(get_settings()) as services:
        items = services.vectors.paper_chunks(paper_id)
    if not items:
        console.print(f"[yellow]库中没有 {paper_id} 的分块[/yellow]")
        raise typer.Exit(code=1)
    console.rule(f"{items[0].title}：{len(items)} 个块")
    for c in items:
        tokens = len(c.content) // 4
        console.print(
            f"[cyan]#{c.index}[/cyan] [dim]{c.kind} · ~{tokens} tokens · "
            f"{c.section} · {c.anchor}[/dim]",
            highlight=False,
        )
        console.print(
            c.content if full else c.content[:160].replace("\n", " ") + "…", highlight=False
        )


def _hit_rows(title: str, hits, score_fmt: str = "{:.4f}") -> Table:
    table = Table(title=title, title_justify="left", show_lines=False, expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("分数", justify="right", width=8)
    table.add_column("论文 · 章节", ratio=2)
    table.add_column("内容", ratio=3)
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        table.add_row(
            str(i),
            score_fmt.format(h.score),
            f"{c.title[:30]} · {c.section[-40:]}",
            c.content[:90].replace("\n", " "),
        )
    return table


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="检索问题"),
    english: str | None = typer.Option(
        None, "--english", "-e", help="BM25 用的英文检索句；不填时由 LLM 改写"
    ),
    paper: list[str] = typer.Option(None, "--paper", "-p", help="只在这些论文里检索"),
    explain: bool = typer.Option(False, "--explain", help="并排显示 dense / BM25 / RRF / 重排"),
    top: int = typer.Option(8, "--top", help="每个阶段显示几条"),
) -> None:
    """只检索不生成，查看混合检索各阶段的结果。"""
    from paperlens.rag.query import analyze_query
    from paperlens.rag.retriever import Retriever
    from paperlens.services import Services

    with Services(get_settings()) as services:
        if english is None:
            analysis = analyze_query(services.llm, query)
            english = analysis.english_query
            console.print(f"[dim]英文检索句（BM25 用）：{english}[/dim]")
        result = Retriever(services).retrieve(
            query, english_query=english, paper_ids=paper or None, explain=explain
        )
        if explain:
            console.print(_hit_rows("① 稠密检索（bge-m3，COSINE）", result.dense[:top]))
            console.print(_hit_rows("② BM25（Milvus 内置）", result.bm25[:top]))
            console.print(_hit_rows("③ RRF 融合（k=60）", result.fused[:top]))
        console.print(_hit_rows("④ 重排（bge-reranker-v2-m3）", result.evidence[:top], "{:.3f}"))
        threshold = services.settings.evidence_threshold
        verdict = "充足" if result.top_score >= threshold else "不足"
        console.print(f"最高重排分 {result.top_score:.3f}，阈值 {threshold} → 证据{verdict}")
        _print_usage(services.tracker)


# ---------------------------------------------------------------- M3：带引用问答


def _print_citations(citations, used: list[int]) -> None:
    if not used:
        return
    console.print()
    console.print("[bold]引用[/bold]")
    by_n = {c.n: c for c in citations}
    for n in used:
        c = by_n[n]
        link = f"[link={c.url}]{c.url}[/link]" if c.url else "（本地 PDF）"
        console.print(f"[{n}] {c.title[:50]} · {c.section[-50:]}\n    {link}", highlight=False)


@app.command()
def ask(
    question: str = typer.Argument(..., help="问题，中英文均可"),
    paper: list[str] = typer.Option(None, "--paper", "-p", help="只在这些论文里检索"),
) -> None:
    """单轮带引用问答（不经过 Agent，直接走检索 + 生成）。"""
    from paperlens.rag.answer import check_citations, stream_answer
    from paperlens.rag.query import analyze_query, resolve_papers
    from paperlens.rag.retriever import Retriever
    from paperlens.services import Services

    with Services(get_settings()) as services:
        analysis = analyze_query(services.llm, question)
        paper_ids = list(paper or []) or resolve_papers(services.db, analysis.paper_mentions)
        console.print(
            f"[dim]英文检索句：{analysis.english_query}"
            f"{'；限定论文：' + ', '.join(paper_ids) if paper_ids else ''}[/dim]"
        )
        result = Retriever(services).retrieve(
            analysis.standalone_question,
            english_query=analysis.english_query,
            paper_ids=paper_ids or None,
        )
        if result.top_score < services.settings.evidence_threshold:
            console.print(
                f"[yellow]本地知识库里没有找到足够相关的内容（最高重排分 "
                f"{result.top_score:.3f}）。可以用 paperlens chat 让 Agent 去 arXiv 找论文。"
                f"[/yellow]"
            )
            raise typer.Exit()
        tokens, citations = stream_answer(
            services.llm, analysis.standalone_question, result.evidence
        )
        answer = ""
        for piece in tokens:
            answer += piece
            console.print(piece, end="", highlight=False, markup=False)
        console.print()
        check = check_citations(answer, len(citations))
        if check.invalid:
            console.print(f"[yellow]回答中有越界的引用编号 {check.invalid}，已忽略[/yellow]")
        _print_citations(citations, check.used)
        _print_usage(services.tracker)


# ---------------------------------------------------------------- M5：解读卡片与多篇对比


def _resolve_one(services, name: str) -> str:
    """命令行参数可以是 paper_id，也可以是标题里的系统名（如 Self-RAG）。"""
    found = services.db.find_papers(name, limit=1)
    if not found:
        console.print(f"[red]知识库里没有找到论文：{name}[/red]")
        raise typer.Exit(code=1)
    return found[0].meta.paper_id


@app.command()
def card(
    paper: str = typer.Argument(..., help="论文 ID 或标题里的名称，如 2310.11511 / Self-RAG"),
    refresh: bool = typer.Option(False, "--refresh", help="忽略缓存，重新生成"),
    output: Path | None = typer.Option(None, "--output", "-o", help="导出 Markdown 的路径"),
) -> None:
    """单篇解读卡片：整篇全文送进长上下文，抽取问题 / 方法 / 实验 / 结果 / 局限并标出处。"""
    from paperlens.rag.card import CardError, anchor_url, card_markdown, get_card
    from paperlens.services import Services

    with Services(get_settings()) as services:
        pid = _resolve_one(services, paper)
        record = services.db.get_paper(pid)
        cached = services.db.get_card(pid) is not None and not refresh
        console.print(
            f"[dim]{'读取缓存的卡片' if cached else '通读全文生成卡片'}：{record.meta.title}[/dim]"
        )
        try:
            result = get_card(services, pid, refresh=refresh)
        except CardError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        text = card_markdown(
            result, title=record.meta.title, url_of=lambda a: anchor_url(record, a)
        )
        console.print(Markdown(text))
        if output:
            output.write_text(text, encoding="utf-8")
            console.print(f"[green]已导出：{output}[/green]")
        _print_usage(services.tracker)


@app.command()
def compare(
    papers_: list[str] = typer.Argument(..., metavar="PAPERS...", help="2–5 篇论文的 ID 或名称"),
    focus: str = typer.Option("", "--focus", help="对比的关注点，如“检索时机的决策方式”"),
    output: Path | None = typer.Option(None, "--output", "-o", help="导出 Markdown 的路径"),
) -> None:
    """多篇对比：基于各篇的解读卡片生成对比表（缺卡片的先并行生成）。"""
    from paperlens.rag.card import CardError
    from paperlens.rag.compare import compare_markdown, compare_papers
    from paperlens.services import Services

    with Services(get_settings()) as services:
        ids = [_resolve_one(services, name) for name in papers_]
        missing = [i for i in ids if services.db.get_card(i) is None]
        if missing:
            console.print(f"[dim]先生成 {len(missing)} 张解读卡片：{', '.join(missing)}[/dim]")
        try:
            result = compare_papers(services, ids, focus=focus)
        except CardError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        table = Table(show_lines=True, expand=True)
        table.add_column("维度", style="bold", ratio=1)
        for p in result.papers:
            table.add_column(f"{p.short_title}\n[dim]{p.paper_id}[/dim]", ratio=2)
        for row in result.rows:
            table.add_row(row.dimension, *(c.text for c in row.cells))
        console.print(table)
        console.print(f"[bold]总结[/bold]：{result.summary}", highlight=False)
        if output:
            output.write_text(compare_markdown(result), encoding="utf-8")
            console.print(f"[green]已导出：{output}[/green]")
        _print_usage(services.tracker)


# ---------------------------------------------------------------- M4：Agent 多轮对话


def _render_answer(text: str) -> None:
    console.print(Markdown(text))


@app.command()
def chat(
    thread: str = typer.Option("default", "--thread", "-t", help="会话 ID，相同 ID 可以接着聊"),
    new: bool = typer.Option(False, "--new", help="开启一个新会话"),
) -> None:
    """多轮对话（LangGraph Agent）：路由、追问、证据不足时询问是否联网补充论文。"""
    import uuid

    from paperlens.agent.graph import PaperAgent
    from paperlens.services import Services

    thread_id = uuid.uuid4().hex[:8] if new else thread
    settings = get_settings()
    with Services(settings) as services, PaperAgent(services) as agent:
        console.rule(f"PaperLens 对话 · 会话 {thread_id}")
        history = agent.history(thread_id)
        if history:
            console.print(f"[dim]已恢复 {len(history) // 2} 轮历史对话[/dim]")
        console.print("[dim]输入问题开始对话；/new 开新会话，/cost 查看花费，/quit 退出[/dim]")
        while True:
            try:
                question = console.input("\n[bold cyan]你[/bold cyan]：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question in ("/quit", "/exit", "/q"):
                break
            if question == "/new":
                thread_id = uuid.uuid4().hex[:8]
                console.print(f"[dim]已开启新会话 {thread_id}[/dim]")
                continue
            if question == "/cost":
                _print_usage(services.tracker)
                continue

            events = agent.stream(thread_id, question)
            while True:
                interrupt = _consume(events)
                if interrupt is None:
                    break
                choice = _ask_candidates(interrupt)
                events = agent.resume(thread_id, choice)
        _print_usage(services.tracker, "本次对话用量")


def _consume(events) -> dict | None:
    """打印 Agent 的事件流；遇到人机确认时返回 interrupt 的内容。"""
    started = False
    answer = ""
    for event in events:
        kind = event["type"]
        if kind == "status":
            console.print(f"[dim]· {event['text']}[/dim]", highlight=False)
        elif kind == "token":
            if not started:
                console.print("\n[bold green]PaperLens[/bold green]：", end="")
                started = True
            answer += event["text"]
            console.print(event["text"], end="", highlight=False, markup=False)
        elif kind == "message":
            console.print("\n[bold green]PaperLens[/bold green]：")
            _render_answer(event["text"])
        elif kind == "citations":
            if started:
                console.print()
            _print_citations(event["citations"], [c.n for c in event["citations"]])
        elif kind == "interrupt":
            return event["payload"]
    if started and not answer.endswith("\n"):
        console.print()
    return None


def _ask_candidates(payload: dict) -> list[str]:
    console.print(f"\n[yellow]{payload['message']}[/yellow]")
    candidates = payload["candidates"]
    for i, c in enumerate(candidates, start=1):
        console.print(
            f"  [cyan]{i}[/cyan]. {c['title'][:80]} [dim]({c['paper_id']}, "
            f"{c['published'][:4]})[/dim]",
            highlight=False,
        )
    raw = console.input("输入要入库的序号（如 1,3；直接回车表示都不要）：").strip()
    picked: list[str] = []
    for part in re.split(r"[,，\s]+", raw):
        if part.isdigit() and 1 <= int(part) <= len(candidates):
            picked.append(candidates[int(part) - 1]["paper_id"])
    return picked


# ---------------------------------------------------------------- M6：Web 服务


@app.command()
def serve(
    host: str | None = typer.Option(
        None, "--host", help="监听地址，默认读取 API_HOST（127.0.0.1）"
    ),
    port: int | None = typer.Option(None, "--port", help="端口，默认读取 API_PORT（8000）"),
) -> None:
    """启动 REST API（FastAPI + SSE）。Milvus Lite 单进程限制：服务运行时不要再跑 ingest 等命令。"""
    import uvicorn

    from paperlens.api.app import create_app

    settings = get_settings()
    host = host or settings.api_host
    port = port or settings.api_port
    if host not in ("127.0.0.1", "localhost", "::1") and settings.api_key is None:
        console.print(
            "[yellow]警告：服务监听在非本机地址，但没有设置 API_KEY，任何能访问这个端口的人"
            "都可以调用接口（消耗你的模型额度、删除论文）。建议在 .env 里设置 API_KEY。[/yellow]"
        )
    console.print(f"PaperLens API：http://{host}:{port}  接口文档：http://{host}:{port}/docs")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", help="Streamlit 端口"),
    api: str = typer.Option(
        "", "--api", help="后端地址，默认读取 PAPERLENS_API 或 http://127.0.0.1:8000"
    ),
) -> None:
    """启动 Streamlit 界面（需要先运行 paperlens serve）。"""
    import os
    import subprocess
    import sys

    settings = get_settings()
    env = dict(os.environ)
    env.setdefault("PAPERLENS_API", f"http://127.0.0.1:{settings.api_port}")
    if api:
        env["PAPERLENS_API"] = api
    if settings.api_key is not None:
        env.setdefault("API_KEY", settings.api_key.get_secret_value())
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).parent / "ui" / "app.py"),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    console.print(f"界面：http://127.0.0.1:{port}  后端：{env['PAPERLENS_API']}")
    raise typer.Exit(subprocess.call(cmd, env=env))


# ---------------------------------------------------------------- M8：评估


@app.command("eval")
def eval_cmd(
    config: list[str] = typer.Option(None, "--config", "-c", help="只跑这些配置，如 -c A0 -c A3"),
    answers: bool = typer.Option(False, "--answers", help="同时跑端到端问答（调用 LLM，约 ¥0.5）"),
) -> None:
    """在 eval/qa.jsonl 上评估检索（A0–A4 消融）与问答，报告写到 eval/reports/。"""
    from paperlens.eval.report import write_report
    from paperlens.eval.runner import run_eval

    out = run_eval(
        get_settings(),
        configs=list(config or []) or None,
        answers=answers,
        log=lambda m: console.print(f"[dim]{m}[/dim]"),
    )
    table = Table(title="检索消融（可回答的 40 题）")
    for col in ("配置", "Hit@5", "Recall@5", "MRR@10", "术语题 Hit@5", "中文题 Hit@5", "延迟 p50"):
        table.add_column(col, justify="left" if col == "配置" else "right")
    for name, r in out.retrieval.items():
        m = r["metrics"]
        table.add_row(
            name,
            f"{m['all']['hit@5']:.1%}",
            f"{m['all']['recall@5']:.1%}",
            f"{m['all']['mrr@10']:.3f}",
            f"{m.get('term', {}).get('hit@5', 0):.1%}",
            f"{m.get('lang_zh', {}).get('hit@5', 0):.1%}",
            f"{r['latency_p50']:.2f}s",
        )
    console.print(table)
    if out.answers:
        a = out.answers
        console.print(
            f"问答：正确率 {a['accuracy_answerable']:.1%} · 拒答准确率 {a['refusal_accuracy']:.1%}"
            f" · 引用合法率 {a['citation_valid_rate']:.1%} · 延迟 p50 {a['latency_p50']:.1f}s"
            f" · 单题 ${a['cost_per_question_usd']:.5f}"
        )
    path = write_report(Path("eval/reports"))
    console.print(f"报告：{path}（评审用 LLM 花费 ${out.judge_cost_usd:.4f}）")


if __name__ == "__main__":
    app()
