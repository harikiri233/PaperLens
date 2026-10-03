"""PaperLens Web 界面（Streamlit）。所有操作都通过 HTTP 调用 PaperLens API。

启动：先 `paperlens serve`，再 `paperlens ui`（或 `streamlit run src/paperlens/ui/app.py`）。
"""

import os
import re
import time
import uuid

import pandas as pd
import streamlit as st

from paperlens.api.client import APIError, PaperLensClient

st.set_page_config(page_title="PaperLens", page_icon="📚", layout="wide")

API_URL = os.environ.get("PAPERLENS_API", "http://127.0.0.1:8000")
USD_TO_CNY = 7.1


@st.cache_resource
def get_client() -> PaperLensClient:
    return PaperLensClient(API_URL, api_key=os.environ.get("API_KEY") or None)


client = get_client()


def call(fn, *args, **kwargs):
    """调用 API；出错时在页面上提示并返回 None。"""
    try:
        return fn(*args, **kwargs)
    except APIError as exc:
        st.error(f"请求失败（{exc.status}）：{exc.detail}")
    except Exception as exc:  # 连接不上后端等
        st.error(f"无法连接 PaperLens API（{API_URL}）：{type(exc).__name__}: {exc}")
    return None


def paper_options() -> dict[str, str]:
    """paper_id → 显示名称"""
    papers = call(client.papers) or []
    return {p["paper_id"]: f"{p['title'][:70]}（{p['paper_id']}）" for p in papers}


# ================================================================ 问答页


def linkify(answer: str, citations: list[dict]) -> str:
    """把回答里的 [n] 变成指向原文段落的链接。"""
    urls = {c["n"]: c["url"] for c in citations if c.get("url")}

    def fix(match: re.Match) -> str:
        parts = re.split(r"\s*[,，]\s*", match.group(1))
        return "".join(f"[[{p}]]({urls[int(p)]})" if int(p) in urls else f"[{p}]" for p in parts)

    return re.sub(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]", fix, answer)


def render_message(msg: dict) -> None:
    if msg.get("steps"):
        with st.expander(f"执行过程（{len(msg['steps'])} 步）", expanded=False):
            for step in msg["steps"]:
                st.caption(step)
    st.markdown(linkify(msg["content"], msg.get("citations", [])))
    if msg.get("citations"):
        with st.expander(f"引用 {len(msg['citations'])} 条", expanded=False):
            for c in msg["citations"]:
                link = f"[原文]({c['url']})" if c.get("url") else "本地 PDF"
                st.markdown(f"**[{c['n']}]** {c['title']} · *{c['section']}* · {link}")
                st.caption(c["snippet"])


def run_stream(events) -> None:
    """渲染一次 Agent 的事件流，结束后把消息存进会话。"""
    msg = {"role": "assistant", "content": "", "citations": [], "steps": []}
    with st.chat_message("assistant"):
        status = st.status("分析问题…", expanded=False)
        body = st.empty()
        try:
            for event in events:
                kind = event["type"]
                if kind == "status":
                    msg["steps"].append(event["text"])
                    status.write(event["text"])
                    status.update(label=event["text"][:80])
                elif kind == "token":
                    msg["content"] += event["text"]
                    body.markdown(msg["content"] + "▌")
                elif kind == "message":
                    msg["content"] = event["text"]
                    body.markdown(msg["content"])
                elif kind == "citations":
                    msg["citations"] = event["citations"]
                elif kind == "interrupt":
                    st.session_state.pending = event["payload"]
        except APIError as exc:
            msg["content"] = msg["content"] or f"请求失败（{exc.status}）：{exc.detail}"
        except Exception as exc:
            msg["content"] = msg["content"] or f"无法连接 PaperLens API：{exc}"
        status.update(label="完成", state="complete")
        body.empty()
        if msg["content"]:
            render_message(msg)
    if msg["content"]:
        st.session_state.messages.append(msg)


def load_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    data = call(client.history, thread_id) or {}
    st.session_state.messages = data.get("messages", [])
    st.session_state.pending = data.get("pending")


def chat_page() -> None:
    if "thread_id" not in st.session_state:
        load_thread(uuid.uuid4().hex[:8])

    with st.sidebar:
        st.subheader("会话")
        st.code(st.session_state.thread_id, language=None)
        col1, col2 = st.columns(2)
        if col1.button("新会话", use_container_width=True):
            load_thread(uuid.uuid4().hex[:8])
            st.rerun()
        other = st.text_input("恢复会话 ID", placeholder="如 9771382c")
        if col2.button("恢复", use_container_width=True, disabled=not other.strip()):
            load_thread(other.strip())
            st.rerun()
        stats = call(client.stats)
        if stats:
            cost = stats["total_cost_usd"]
            st.caption(f"服务累计花费约 ${cost:.4f}（¥{cost * USD_TO_CNY:.3f}）")

    st.title("💬 论文问答")
    if not st.session_state.messages:
        st.info(
            "可以这样问：\n"
            "- Self-RAG 的反思 token 有哪几种？（带引用的问答）\n"
            "- 讲讲 ReAct 这篇论文（解读卡片）\n"
            "- 对比一下 Self-RAG 和 CRAG（对比表）\n"
            "- 帮我找几篇关于 GraphRAG 的论文（联网找论文并入库）"
        )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_message(msg)

    pending = st.session_state.get("pending")
    if pending:
        with st.chat_message("assistant"), st.form("confirm"):
            st.warning(pending["message"])
            picked = []
            for c in pending["candidates"]:
                label = f"**{c['title']}**（{c['paper_id']}，{c['published'][:4]}，相关度 {c['score']:.2f}）"
                if st.checkbox(label, key=f"pick-{c['paper_id']}"):
                    picked.append(c["paper_id"])
                st.caption(c["abstract"])
            ok = st.form_submit_button("入库选中的论文", type="primary")
            skip = st.form_submit_button("都不要")
        if ok or skip:
            st.session_state.pending = None
            run_stream(client.resume(st.session_state.thread_id, picked if ok else []))
            st.rerun()

    question = st.chat_input("问点什么…", disabled=bool(pending))
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        run_stream(client.chat(question, st.session_state.thread_id))
        st.rerun()


# ================================================================ 论文库页


def jobs_panel() -> None:
    jobs = call(client.jobs) or []
    if not jobs:
        return
    st.subheader("入库任务")
    running = any(j["status"] in ("queued", "running") for j in jobs)
    labels = {"queued": "排队中", "running": "进行中", "done": "完成", "failed": "失败"}
    for j in jobs[:5]:
        results = j["results"]
        ok = sum(1 for r in results if r["status"] != "failed")
        line = f"**{labels[j['status']]}** · {j['description'][:80]} · 已处理 {len(results)} 篇（成功 {ok}）"
        if j["error"]:
            line += f" · {j['error']}"
        st.markdown(line)
        for r in results:
            if r["status"] == "failed":
                st.caption(f"❌ {r['paper_id']} {r['title'][:60]}：{r['message']}")
    if running:
        st.caption("任务进行中，页面每 3 秒刷新一次…")
        time.sleep(3)
        st.rerun()


def library_page() -> None:
    st.title("📚 论文库")
    add_ids, add_search, add_pdf = st.tabs(["按 arXiv ID 入库", "在 arXiv 上搜索", "上传 PDF"])
    with add_ids, st.form("ingest-ids"):
        raw = st.text_area("arXiv ID 或链接，每行一个", placeholder="2310.11511\n2401.15884")
        force = st.checkbox("已在库里的也重新入库")
        if st.form_submit_button("入库", type="primary"):
            ids = [x.strip() for x in re.split(r"[\s,，]+", raw) if x.strip()]
            if ids and call(client.ingest, ids, force=force):
                st.success(f"已提交 {len(ids)} 篇，后台入库中")
    with add_search:
        q = st.text_input("关键词", placeholder="graph retrieval augmented generation")
        if st.button("搜索", disabled=not q.strip()):
            st.session_state.arxiv_results = call(client.arxiv_search, q.strip(), 10) or []
        results = st.session_state.get("arxiv_results", [])
        if results:
            with st.form("ingest-search"):
                picked = []
                for r in results:
                    tag = "（已在库中）" if r["in_library"] else ""
                    if st.checkbox(
                        f"**{r['title']}**{tag} · {r['paper_id']} · {r['published'][:4]}",
                        key=f"s-{r['paper_id']}",
                        disabled=r["in_library"],
                    ):
                        picked.append(f"{r['paper_id']}{r['version']}")
                    st.caption(r["abstract"][:300] + "…")
                submitted = st.form_submit_button("入库选中的论文", type="primary")
                if submitted and picked and call(client.ingest, picked):
                    st.success(f"已提交 {len(picked)} 篇")
    with add_pdf:
        file = st.file_uploader("选择 PDF", type=["pdf"])
        clicked = st.button("上传并入库", type="primary", disabled=file is None)
        if clicked and file and call(client.upload, file.name, file.getvalue()):
            st.success("已上传，后台解析入库中")

    jobs_panel()

    papers = call(client.papers) or []
    st.subheader(f"已入库 {len(papers)} 篇")
    if not papers:
        return
    df = pd.DataFrame(papers)[["paper_id", "title", "year", "source", "n_chunks", "added_at"]]
    df.columns = ["ID", "标题", "年份", "来源", "块数", "入库时间"]
    st.dataframe(df, hide_index=True)
    by_id = {p["paper_id"]: p for p in papers}
    pid = st.selectbox(
        "查看论文",
        list(by_id),
        index=None,
        format_func=lambda i: f"{by_id[i]['title'][:80]}（{i}）",
        placeholder="选择一篇论文查看摘要和解读卡片",
    )
    if pid is None:
        return
    paper = by_id[pid]
    st.divider()
    st.subheader(paper["title"])
    meta = f"{', '.join(paper['authors'][:5])} · {paper['published']} · {paper['source']}"
    if paper["url"]:
        meta += f" · [arXiv]({paper['url']})"
    st.caption(meta)
    col1, col2, col3 = st.columns([1, 1, 4])
    show = col1.button("解读卡片", type="primary")
    refresh = col2.button("重新生成")
    if col3.button("从库中删除", help="删除论文的分块、元数据和卡片"):
        st.session_state.confirm_delete = paper["paper_id"]
    if st.session_state.get("confirm_delete") == paper["paper_id"]:
        st.warning(f"确定删除 {paper['paper_id']}？删除后需要重新入库。")
        if st.button("确定删除"):
            call(client.delete_paper, paper["paper_id"])
            st.session_state.confirm_delete = None
            st.rerun()
    if show or refresh:
        with st.spinner("通读全文生成卡片（约 10 秒）…" if refresh else "加载卡片…"):
            card = call(client.card, paper["paper_id"], refresh=refresh)
        if card:
            st.markdown(card["markdown"])
    else:
        with st.expander("摘要", expanded=True):
            st.write(paper["abstract"] or "（无摘要）")


# ================================================================ 对比页


def compare_page() -> None:
    st.title("⚖️ 多篇对比")
    options = paper_options()
    picked = st.multiselect(
        "选择 2–5 篇论文", list(options), format_func=options.get, max_selections=5
    )
    focus = st.text_input("关注点（可选）", placeholder="如：何时触发检索、如何处理检索质量")
    if st.button("生成对比表", type="primary", disabled=len(picked) < 2):
        with st.spinner("读取解读卡片（缺的先生成），再生成对比表…"):
            result = call(client.compare, picked, focus)
        if result:
            st.session_state.compare_result = result
    result = st.session_state.get("compare_result")
    if result:
        columns = {p["paper_id"]: p["short_title"] for p in result["papers"]}
        table = pd.DataFrame(
            [
                {
                    "维度": row["dimension"],
                    **{columns[c["paper_id"]]: c["text"] for c in row["cells"]},
                }
                for row in result["rows"]
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True, row_height=80)
        st.markdown(f"**总结**：{result['summary']}")
        st.download_button(
            "下载 Markdown", result["markdown"], file_name="compare.md", mime="text/markdown"
        )


# ================================================================ 检索调试页


def search_page() -> None:
    st.title("🔍 检索调试")
    st.caption("只检索不生成：并排查看稠密检索、BM25、RRF 融合和重排四个阶段的结果。")
    options = paper_options()
    query = st.text_input("问题", placeholder="Self-RAG 的反思 token 有哪几种？")
    scope = st.multiselect("限定论文（可选）", list(options), format_func=options.get)
    if st.button("检索", type="primary", disabled=not query.strip()):
        with st.spinner("检索中…"):
            st.session_state.search_result = call(
                client.search, query.strip(), paper_ids=scope or None
            )
    r = st.session_state.get("search_result")
    if not r:
        return
    verdict = "充足" if r["top_score"] >= r["threshold"] else "不足"
    st.markdown(
        f"BM25 检索句：`{r['bm25_query']}` · 最高重排分 **{r['top_score']:.3f}**"
        f"（阈值 {r['threshold']}，证据{verdict}）"
    )
    stages = [
        ("① 稠密（bge-m3）", r["dense"]),
        ("② BM25", r["bm25"]),
        ("③ RRF 融合", r["fused"]),
        ("④ 重排", r["reranked"]),
    ]
    for col, (label, hits) in zip(st.columns(4), stages, strict=True):
        with col:
            st.markdown(f"**{label}**")
            for i, h in enumerate(hits[:8], start=1):
                st.markdown(
                    f"{i}. `{h['score']:.3f}` [{h['title'][:28]}]({h['url']}) · {h['section'][-30:]}"
                )
                st.caption(h["content"][:120])


# ================================================================ 导航

health = call(client.health)
with st.sidebar:
    st.markdown("## 📚 PaperLens")
    if health:
        st.caption(f"知识库 {health['papers']} 篇 · 模型 {health['model']}")

pages = st.navigation(
    [
        st.Page(chat_page, title="问答", icon="💬", default=True),
        st.Page(library_page, title="论文库", icon="📚", url_path="library"),
        st.Page(compare_page, title="对比", icon="⚖️", url_path="compare"),
        st.Page(search_page, title="检索调试", icon="🔍", url_path="search"),
    ]
)
pages.run()
