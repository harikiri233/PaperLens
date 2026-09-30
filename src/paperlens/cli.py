"""命令行入口。"""

import typer
from rich.console import Console
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

    usage = tracker.snapshot()
    if usage:
        table = Table(title="本次自检的模型用量", show_lines=False)
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

    failed = [r.name for r in results if r.status == "fail"]
    if failed:
        console.print(f"[red]有 {len(failed)} 项失败：{', '.join(failed)}[/red]")
        raise typer.Exit(code=1)
    console.print("[green]自检结束，没有失败项。[/green]")


if __name__ == "__main__":
    app()
