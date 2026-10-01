"""查询理解：一次 JSON 调用完成意图识别、指代消解、英文检索句和论文点名抽取。"""

from typing import Literal

from pydantic import BaseModel, Field

from paperlens.models import LLMClient
from paperlens.models.llm import Message
from paperlens.store import Database

Intent = Literal["qa", "explain", "compare", "discover", "chat"]

_SYSTEM = """你是论文知识库助手的查询分析模块。根据对话历史和用户的最新问题，输出 JSON。

字段说明：
- intent：
  - qa：关于论文内容的具体问题（方法、实验、结果、定义、对比某个细节等）
  - explain：要求整体解读 / 总结某一篇论文（如"讲讲 Self-RAG 这篇论文"）
  - compare：要求对比两篇及以上论文的整体异同（如"对比 Self-RAG 和 CRAG"）
  - discover：要求寻找、推荐新论文（如"帮我找几篇关于 RAG 评估的论文"）
  - chat：打招呼、闲聊、与论文无关的问题
- standalone_question：结合历史改写成不依赖上下文的完整问题，消解"它""这篇""第二个方法"等指代；用用户的语言。
- english_query：用于关键词检索的英文句子，保留术语、缩写及其全称；用户本来就用英文提问时可以照抄。
- paper_mentions：问题中点名的论文名称或 arXiv ID（如 "Self-RAG"、"2310.11511"），结合历史补全指代；没有则为空列表。
- search_keywords：去 arXiv 搜论文用的 2–4 个英文关键词或短语，只保留主题词，不要疑问词和泛泛的词（如 ["diffusion model", "image generation"]）。"""


class QueryAnalysis(BaseModel):
    intent: Intent = "qa"
    standalone_question: str
    english_query: str = ""
    paper_mentions: list[str] = Field(default_factory=list)
    search_keywords: list[str] = Field(default_factory=list)


def format_history(messages: list[Message], *, max_turns: int = 6) -> str:
    """把最近几轮对话拼成文本；助手的长回答截断，只保留开头用于指代消解。"""
    recent = messages[-max_turns * 2 :]
    lines = []
    for m in recent:
        role = "用户" if m["role"] == "user" else "助手"
        content = str(m["content"])
        if m["role"] != "user" and len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def analyze_query(
    llm: LLMClient, question: str, history: list[Message] | None = None
) -> QueryAnalysis:
    context = format_history(history or [])
    user = f"对话历史：\n{context or '（无）'}\n\n最新问题：{question}"
    try:
        return llm.complete_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            QueryAnalysis,
            max_tokens=400,
        )
    except Exception:
        # 分析失败不影响问答：按普通问题处理，BM25 用原问题
        return QueryAnalysis(intent="qa", standalone_question=question, english_query=question)


def resolve_papers(db: Database, mentions: list[str]) -> list[str]:
    """把点名的论文解析成库里的 paper_id，每个名称取最匹配的一篇；按出现顺序去重。"""
    ids: list[str] = []
    for mention in mentions:
        for record in db.find_papers(mention, limit=1):
            if record.meta.paper_id not in ids:
                ids.append(record.meta.paper_id)
    return ids
