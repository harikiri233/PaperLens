# PaperLens

论文检索与分析 Agent：arXiv 论文知识库 + 混合检索 RAG + LangGraph 编排。

> 当前进度：**M0 工程骨架与模型连通**。后续里程碑见 `../paperlens_design/PaperLens_总体设计.md` §11。

## 环境要求

- Linux 或 WSL2（Milvus Lite 不支持原生 Windows）
- Python 3.11+，[uv](https://docs.astral.sh/uv/)
- DeepSeek API Key、硅基流动 API Key

## 快速开始

```bash
cd paperlens
uv sync                  # 安装依赖（按 uv.lock 锁定版本）
cp .env.example .env     # 然后填写 LLM_API_KEY 与 SILICONFLOW_API_KEY
uv run paperlens doctor  # 环境自检
```

## M0 验证

`uv run paperlens doctor` 会依次检查：

| 检查项 | 通过标准 |
|---|---|
| 配置 | 两个 API Key 都已读取（只显示掩码） |
| LLM | 普通对话有回复；JSON 模式返回的结果通过 Pydantic 校验 |
| Embedding | 返回 1024 维向量；中文问题与语义最相关的英文段落相似度最高 |
| Rerank | 同一个中文问题下，最相关的英文段落得分最高 |
| arXiv | API 和 HTML 页面都能访问 |

最后会打印本次自检的 token 用量和估算费用（通常不到 ¥0.01）。

没有网络或还没有 Key 时，可以先跑单元测试（全部使用替身，不访问网络）：

```bash
uv run pytest -q
uv run ruff check .
```

访问 arXiv 需要代理时，在 `.env` 或 shell 中设置 `HTTPS_PROXY`。

## 目录

```
src/paperlens/
├── config.py          # 配置（.env + 环境变量）
├── models/
│   ├── llm.py         # OpenAI 兼容 LLM 客户端：思考模式开关、JSON 模式重试、用量统计
│   ├── siliconflow.py # Embedding / Rerank 客户端：分批、重试
│   └── usage.py       # token 与费用统计（含 DeepSeek 高峰/非高峰计价）
├── doctor.py          # 环境自检
└── cli.py             # 命令行入口
```

---

由 ClawsGO Science Agent 协助开发。
