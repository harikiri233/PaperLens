"""测试集：eval/qa.jsonl，一行一题。

相关性标注到块（chunk_id）：出题时的来源块，加上池化判定（各配置 top-10 的并集交给 LLM 判断
是否包含答案）认定的块。无法回答的题 relevant 为空。
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["fact", "term", "crosslingual", "unanswerable"]


class QAItem(BaseModel):
    id: str
    category: Category
    lang: Literal["zh", "en"]
    question: str
    answer: str = ""  # 参考答案（人工核对用，不参与打分）
    paper_id: str = ""  # 出题来源论文；无法回答的题为空
    source_chunk: str = ""  # 出题来源块
    section: str = ""  # 来源块所在章节路径
    relevant: list[str] = Field(default_factory=list)  # 相关块 chunk_id（含来源块）

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def load_items(path: Path) -> list[QAItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [QAItem.model_validate_json(line) for line in lines if line.strip()]


def save_items(path: Path, items: list[QAItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(item.model_dump_json() for item in items)
    path.write_text(text + "\n", encoding="utf-8")
