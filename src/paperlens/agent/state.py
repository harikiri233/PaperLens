"""Agent 状态。

除 messages 外都只存 JSON 友好的基本类型（dict / list / str），
这样 SqliteSaver 持久化和恢复时不依赖自定义类的序列化。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    analysis: dict  # QueryAnalysis.model_dump()
    paper_ids: list[str]  # 点名论文解析出的 ID，用作检索过滤
    missing: list[str]  # 点名了但库里没有的论文
    evidence: list[dict]  # [{"chunk": Chunk.model_dump(), "score": float}]
    top_score: float
    candidates: list[dict]  # arXiv 候选论文
    picked: list[str]  # 用户勾选入库的 paper_id
    ingested: list[str]  # 本轮新入库的 paper_id
    supplemented: bool  # 本轮是否已经联网补充过（最多一次，防止循环）
    answer: str
    citations: list[dict]
