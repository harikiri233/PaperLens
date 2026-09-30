"""环境自检：逐项检查配置、模型接口和 arXiv 的连通性。"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import httpx
import openai
from pydantic import BaseModel

from paperlens.config import Settings, mask_secret
from paperlens.models import LLMClient, SiliconFlowClient, UsageTracker, cosine

Status = Literal["ok", "warn", "fail", "skip"]

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_HTML = "https://arxiv.org/html/2310.11511v1"  # Self-RAG，用来确认 HTML 版可访问


@dataclass
class CheckResult:
    name: str
    status: Status
    summary: str
    details: list[str] = field(default_factory=list)
    latency_s: float | None = None


class _Probe(BaseModel):
    """LLM JSON 模式的探测结构。"""

    answer: str
    confidence: float


def check_config(settings: Settings) -> CheckResult:
    details = [
        f"LLM：{settings.llm_model} @ {settings.llm_base_url}，Key {mask_secret(settings.llm_api_key)}",
        f"Embedding：{settings.embedding_model}，Rerank：{settings.rerank_model}",
        f"硅基流动：{settings.siliconflow_base_url}，Key {mask_secret(settings.siliconflow_api_key)}",
        f"数据目录：{settings.data_dir.resolve()}",
    ]
    missing = [
        name
        for name, value in (
            ("LLM_API_KEY", settings.llm_api_key),
            ("SILICONFLOW_API_KEY", settings.siliconflow_api_key),
        )
        if value is None
    ]
    if missing:
        return CheckResult("配置", "warn", f"缺少 {', '.join(missing)}，相关检查会跳过", details)
    return CheckResult("配置", "ok", "配置已加载", details)


def check_llm(settings: Settings, tracker: UsageTracker) -> CheckResult:
    if settings.llm_api_key is None:
        return CheckResult("LLM", "skip", "未配置 LLM_API_KEY")
    try:
        llm = LLMClient(settings, tracker=tracker)
        start = time.perf_counter()
        chat = llm.chat(
            [{"role": "user", "content": "用一句话说明什么是检索增强生成（RAG）。"}],
            max_tokens=200,
        )
        probe = llm.complete_json(
            [
                {
                    "role": "user",
                    "content": "问题：Self-RAG 论文提出的特殊 token 叫什么？给出简短回答和 0–1 的把握程度。",
                }
            ],
            _Probe,
            max_tokens=200,
        )
        latency = time.perf_counter() - start
    except openai.AuthenticationError:
        return CheckResult("LLM", "fail", "鉴权失败（HTTP 401），请检查 LLM_API_KEY。")
    except Exception as exc:  # 自检需要把任何错误都展示出来
        return CheckResult("LLM", "fail", f"{type(exc).__name__}: {exc}")
    details = [
        f"服务端模型：{chat.model}",
        f"普通对话：{chat.content.strip()[:80]}",
        f"  输入 {chat.input_tokens} tokens（缓存命中 {chat.cached_input_tokens}），输出 {chat.output_tokens}，"
        f"{chat.latency_s:.2f}s",
        f"JSON 模式：{probe.model_dump_json()}",
    ]
    return CheckResult("LLM", "ok", "普通对话与 JSON 模式均正常", details, latency)


def check_embedding_and_rerank(settings: Settings, tracker: UsageTracker) -> list[CheckResult]:
    if settings.siliconflow_api_key is None:
        return [
            CheckResult("Embedding", "skip", "未配置 SILICONFLOW_API_KEY"),
            CheckResult("Rerank", "skip", "未配置 SILICONFLOW_API_KEY"),
        ]
    query_zh = "如何让模型判断什么时候需要检索？"
    docs = [
        "Self-RAG trains a language model to decide when to retrieve by generating reflection tokens.",
        "Proximal policy optimization clips the probability ratio to stabilize policy updates.",
        "We evaluate on image classification benchmarks such as ImageNet.",
    ]
    results: list[CheckResult] = []
    with SiliconFlowClient(settings, tracker=tracker) as sf:
        try:
            start = time.perf_counter()
            vectors = sf.embed([query_zh, *docs])
            latency = time.perf_counter() - start
            sims = [cosine(vectors[0], v) for v in vectors[1:]]
            details = [f"向量维度：{len(vectors[0])}", f"中文问题：{query_zh}"]
            details += [f"  相似度 {s:.3f}  {d[:60]}" for s, d in zip(sims, docs, strict=True)]
            best = max(range(len(sims)), key=sims.__getitem__)
            status: Status = "ok" if best == 0 else "warn"
            summary = "跨语言相似度排序符合预期" if best == 0 else "最相关的英文段落没有排在第一"
            results.append(CheckResult("Embedding", status, summary, details, latency))
        except Exception as exc:
            results.append(CheckResult("Embedding", "fail", f"{type(exc).__name__}: {exc}"))
        try:
            start = time.perf_counter()
            ranked = sf.rerank(query_zh, docs)
            latency = time.perf_counter() - start
            details = [f"  分数 {r.score:.3f}  {docs[r.index][:60]}" for r in ranked]
            status = "ok" if ranked and ranked[0].index == 0 else "warn"
            summary = "重排结果符合预期" if status == "ok" else "最相关的段落没有排在第一"
            results.append(CheckResult("Rerank", status, summary, details, latency))
        except Exception as exc:
            results.append(CheckResult("Rerank", "fail", f"{type(exc).__name__}: {exc}"))
    return results


def check_arxiv(transport: httpx.BaseTransport | None = None) -> CheckResult:
    """检查 arXiv API 和 HTML 页面能否访问。两次请求之间间隔 3 秒，遵守 arXiv 的访问限制。"""
    details: list[str] = []
    try:
        with httpx.Client(timeout=20, follow_redirects=True, transport=transport) as http:
            start = time.perf_counter()
            api = http.get(ARXIV_API, params={"id_list": "2310.11511", "max_results": 1})
            api_latency = time.perf_counter() - start
            api.raise_for_status()
            found = "<entry>" in api.text
            details.append(
                f"API：HTTP {api.status_code}，{api_latency:.2f}s，{'找到' if found else '未找到'}测试论文"
            )
            time.sleep(3)
            start = time.perf_counter()
            html = http.get(ARXIV_HTML)
            html_latency = time.perf_counter() - start
            html.raise_for_status()
            has_para = "ltx_para" in html.text
            details.append(
                f"HTML：HTTP {html.status_code}，{html_latency:.2f}s，{len(html.content) // 1024} KB，"
                f"{'含' if has_para else '不含'} LaTeXML 段落标记"
            )
    except Exception as exc:
        details.append("如果网络需要代理，请在 .env 或环境变量中设置 HTTPS_PROXY")
        return CheckResult("arXiv", "fail", f"{type(exc).__name__}: {exc}", details)
    status: Status = "ok" if found and has_para else "warn"
    summary = "API 与 HTML 均可访问" if status == "ok" else "可以连通，但返回内容不符合预期"
    return CheckResult("arXiv", status, summary, details, api_latency + html_latency)


def run_all(
    settings: Settings,
    tracker: UsageTracker,
    *,
    skip_arxiv: bool = False,
    on_result: Callable[[CheckResult], None] | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    def add(result: CheckResult) -> None:
        results.append(result)
        if on_result:
            on_result(result)

    add(check_config(settings))
    add(check_llm(settings, tracker))
    for r in check_embedding_and_rerank(settings, tracker):
        add(r)
    if skip_arxiv:
        add(CheckResult("arXiv", "skip", "已通过 --skip-arxiv 跳过"))
    else:
        add(check_arxiv())
    return results
