"""多篇对比：先拿到每篇的解读卡片（没有就并行生成），再基于卡片生成对比表。

对比时只把卡片送进模型，不送全文：5 篇论文的卡片加起来也只有几千 token，
比把 5 篇全文塞进上下文便宜得多，维度也更整齐。
"""

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from paperlens.ingest.pipeline import load_parsed
from paperlens.rag.card import CardError, PaperCard, generate_card
from paperlens.services import Services

MIN_PAPERS, MAX_PAPERS = 2, 5

_SYSTEM = """你是论文对比助手。<cards> 中是几篇论文的解读卡片（JSON），按 paper_id 区分。生成一张中文对比表，输出 JSON。

要求：
- rows：5–8 个对比维度，如"要解决的问题""核心思路""是否需要训练""检索 / 推理方式""实验任务与数据集""主要结果""局限"；
  选最能区分这几篇论文的维度。用户给了关注点时，优先围绕关注点选维度。
- 每个维度的 cells 必须为每篇论文各写一格（paper_id 与输入一致），每格不超过 50 字；卡片里没有的信息写"未提及"。
- summary：3–5 句话，总结共同点、关键差异，以及各自适合的场景。
- 只根据卡片内容填写，不要编造数字。"""


class CompareCell(BaseModel):
    paper_id: str
    text: str


class CompareRow(BaseModel):
    dimension: str
    cells: list[CompareCell] = Field(default_factory=list)


class Comparison(BaseModel):
    rows: list[CompareRow]
    summary: str


class ComparedPaper(BaseModel):
    paper_id: str
    title: str
    short_title: str


class CompareResult(BaseModel):
    papers: list[ComparedPaper]
    rows: list[CompareRow]  # cells 已按 papers 的顺序对齐
    summary: str


def short_title(title: str) -> str:
    """表头用的简称：取冒号前的系统名（如 "Self-RAG"），太长时截断。"""
    head = title.split(":", 1)[0].strip()
    return head if len(head) <= 40 else head[:37] + "…"


def ensure_cards(services: Services, paper_ids: list[str]) -> dict[str, PaperCard]:
    """读取缓存的卡片；缺的并行生成（只调模型），生成完在当前线程统一写库。"""
    cards: dict[str, PaperCard] = {}
    todo = []
    for pid in paper_ids:
        cached = services.db.get_card(pid)
        if cached:
            cards[pid] = PaperCard.model_validate(cached)
            continue
        paper = load_parsed(services, pid)
        if paper is None:
            raise CardError(f"找不到 {pid} 的解析结果，请用 --force 重新入库")
        todo.append(paper)
    if todo:
        _ = services.llm  # cached_property 不是线程安全的，先在当前线程创建客户端
        with ThreadPoolExecutor(max_workers=min(len(todo), MAX_PAPERS)) as pool:
            generated = list(pool.map(lambda p: generate_card(services, p), todo))
        for paper, card in zip(todo, generated, strict=True):
            services.db.save_card(
                paper.meta.paper_id, card.model_dump(), services.settings.llm_model
            )
            cards[paper.meta.paper_id] = card
    return cards


def _align(rows: list[CompareRow], paper_ids: list[str]) -> list[CompareRow]:
    """按论文顺序重排单元格；模型漏写的格子补"未提及"，多写的丢掉。"""
    aligned = []
    for row in rows:
        by_id = {c.paper_id: c.text for c in row.cells}
        aligned.append(
            CompareRow(
                dimension=row.dimension,
                cells=[CompareCell(paper_id=p, text=by_id.get(p, "未提及")) for p in paper_ids],
            )
        )
    return aligned


def compare_papers(services: Services, paper_ids: list[str], *, focus: str = "") -> CompareResult:
    paper_ids = list(dict.fromkeys(paper_ids))  # 去重并保持顺序
    if not MIN_PAPERS <= len(paper_ids) <= MAX_PAPERS:
        raise CardError(f"对比需要 {MIN_PAPERS}–{MAX_PAPERS} 篇论文，当前 {len(paper_ids)} 篇")
    papers = []
    for pid in paper_ids:
        record = services.db.get_paper(pid)
        if record is None:
            raise CardError(f"知识库里没有论文 {pid}")
        title = record.meta.title
        papers.append(ComparedPaper(paper_id=pid, title=title, short_title=short_title(title)))

    cards = ensure_cards(services, paper_ids)
    blocks = [
        f'<card paper_id="{p.paper_id}" title="{p.title}">\n'
        f"{cards[p.paper_id].model_dump_json(exclude_none=True)}\n</card>"
        for p in papers
    ]
    user = "<cards>\n" + "\n".join(blocks) + "\n</cards>"
    if focus:
        user += f"\n\n用户的关注点：{focus}"
    comparison = services.llm.complete_json(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        Comparison,
        max_tokens=2500,
    )
    return CompareResult(
        papers=papers, rows=_align(comparison.rows, paper_ids), summary=comparison.summary
    )


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def compare_markdown(result: CompareResult) -> str:
    header = "| 维度 | " + " | ".join(_cell(p.short_title) for p in result.papers) + " |"
    sep = "|---" * (len(result.papers) + 1) + "|"
    lines = [header, sep]
    for row in result.rows:
        lines.append(
            f"| **{_cell(row.dimension)}** | " + " | ".join(_cell(c.text) for c in row.cells) + " |"
        )
    lines += ["", f"**总结**：{result.summary}"]
    return "\n".join(lines) + "\n"
