# PaperLens

论文检索与分析 Agent：arXiv 论文知识库 + 混合检索 RAG + LangGraph 编排。

> 当前进度：**M0–M6 已完成**（解析入库、混合检索、带引用问答、多轮对话 Agent、解读卡片与多篇对比、FastAPI + Streamlit）。
> 项目功能、架构和使用案例见 [docs/PaperLens_项目讲解.md](docs/PaperLens_项目讲解.md)。

## 环境要求

- Linux 或 WSL2（Milvus Lite 不支持原生 Windows）
- Python 3.11+，[uv](https://docs.astral.sh/uv/)
- DeepSeek API Key、硅基流动 API Key

## 快速开始

```bash
cd paperlens
uv sync                                      # 安装依赖（按 uv.lock 锁定版本）
cp .env.example .env                         # 填写 LLM_API_KEY 与 SILICONFLOW_API_KEY
uv run paperlens doctor                      # 环境自检
uv run paperlens ingest -f eval/seed_papers.txt   # 入库 15 篇种子论文（约 2 分钟）
uv run paperlens chat                        # 开始对话
```

Web 界面（两个终端）：

```bash
uv run paperlens serve    # API：http://127.0.0.1:8000（接口文档 /docs）
uv run paperlens ui       # 界面：http://127.0.0.1:8501
```

## 命令

| 命令 | 作用 |
|---|---|
| `paperlens parse 2310.11511 -o out.md` | 解析单篇论文（不入库），显示章节树，导出 Markdown |
| `paperlens ingest <ID...>` / `-f 文件` / `-s 关键词` / `--pdf 路径` | 入库；已入库的自动跳过，`--force` 强制重建 |
| `paperlens papers` | 列出知识库 |
| `paperlens chunks <ID>` | 查看一篇论文的分块 |
| `paperlens search "问题" --explain` | 只检索，并排显示稠密 / BM25 / RRF / 重排四个阶段 |
| `paperlens ask "问题"` | 单轮带引用问答 |
| `paperlens chat [--new] [-t 会话ID]` | 多轮对话 Agent；证据不足时询问是否去 arXiv 补充论文 |
| `paperlens card <ID 或简称> [--refresh]` | 单篇解读卡片（整篇长上下文抽取，带出处，缓存） |
| `paperlens compare <论文...> [--focus 关注点]` | 2–5 篇论文对比表（基于解读卡片） |
| `paperlens serve` / `paperlens ui` | 启动 REST API / Streamlit 界面 |

## 工作流程

```
入库：arXiv HTML（失败时回退 PDF）→ 章节树 + 段落锚点 → 章节内分块 + 上下文标题
      → bge-m3 嵌入 → Milvus（稠密向量 + 内置 BM25）+ SQLite（元数据）

问答：查询分析（意图 / 指代消解 / 英文检索句 / 点名论文）
      → dense（原问题）+ BM25（英文检索句）→ RRF → bge-reranker 重排
      → 证据判定 ──充足──→ 带 [n] 引用的流式回答（链接跳到 arXiv 原文段落）
                 └─不足──→ arXiv 检索 → 重排过滤候选 → 用户确认（interrupt）→ 入库 → 重新检索
```

## 配置

`.env` 里除了两个 Key，常用的可调项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EVIDENCE_THRESHOLD` | 0.3 | 重排最高分低于它视为证据不足 |
| `ONLINE_FALLBACK` | ask | `ask`：证据不足时询问是否联网补充；`off`：直接拒答 |
| `RETRIEVE_TOP_K` | 6 | 送入生成的证据条数 |
| `CHUNK_TARGET_TOKENS` | 400 | 分块目标长度 |
| `API_HOST` / `API_PORT` | 127.0.0.1 / 8000 | API 监听地址 |
| `API_KEY` | 空 | 设置后所有接口需带 `X-API-Key` 头；对外暴露服务时务必设置 |

## 开发

```bash
uv run pytest -q          # 单元测试（模型、arXiv 全部用替身，不访问网络）
uv run ruff check . && uv run ruff format --check .
```

## 目录

```
src/paperlens/
├── config.py / services.py / cli.py / doctor.py
├── models/        # LLM（流式、JSON 模式重试）、Embedding / Rerank、用量统计
├── ingest/        # arxiv_client、html_parser、pdf_parser、chunker、pipeline
├── store/         # db.py（SQLite）、vectors.py（Milvus 混合检索）
├── rag/           # query、retriever、answer（带引用生成）、card（解读卡片）、compare（多篇对比）
├── agent/         # state.py、graph.py（LangGraph 状态图）
├── api/           # app.py（FastAPI + SSE）、client.py（HTTP 客户端）
└── ui/app.py      # Streamlit 界面
docs/                  # 项目讲解、简历介绍、架构图与界面截图
eval/seed_papers.txt   # 种子论文清单
```

## 已知限制

- Milvus Lite 同一个库文件只能被一个进程打开：`serve` 或 `chat` 运行时不能同时用 CLI 跑 `ingest`（Web 模式下在界面里入库）。
- API 默认不鉴权、只监听本机；改成对外监听时请设置 `API_KEY`。
- PDF 回退只能定位到页码，引用链接跳到 PDF 对应页。
- 查询分析失败时按普通问题处理，BM25 通道退回用原问题。

---

由 ClawsGO Science Agent 协助开发。
