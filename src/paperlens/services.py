"""把配置、存储和模型客户端组装在一起，供 CLI、Agent 和后续的 API 共用。

Milvus Lite 同一个库文件只能被一个进程打开，所以一个进程里只创建一个 Services。
"""

from functools import cached_property

from paperlens.config import Settings
from paperlens.models import LLMClient, SiliconFlowClient, UsageTracker
from paperlens.store import Database, VectorStore


class Services:
    def __init__(self, settings: Settings, *, tracker: UsageTracker | None = None) -> None:
        self.settings = settings
        self.tracker = tracker or UsageTracker()

    @cached_property
    def db(self) -> Database:
        return Database(self.settings.db_path)

    @cached_property
    def vectors(self) -> VectorStore:
        return VectorStore(
            self.settings.milvus_path, self.settings.collection, self.settings.embedding_dim
        )

    @cached_property
    def llm(self) -> LLMClient:
        return LLMClient(self.settings, tracker=self.tracker)

    @cached_property
    def sf(self) -> SiliconFlowClient:
        return SiliconFlowClient(self.settings, tracker=self.tracker)

    def close(self) -> None:
        for name in ("db", "vectors", "sf"):
            if name in self.__dict__:
                self.__dict__[name].close()

    def __enter__(self) -> "Services":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
