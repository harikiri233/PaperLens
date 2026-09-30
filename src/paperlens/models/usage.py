"""模型调用的 token 用量与费用统计。"""

import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime

USD_TO_CNY = 7.1  # 仅用于显示近似的人民币金额


@dataclass(frozen=True)
class Price:
    """每百万 token 的美元单价（高峰时段）。"""

    input_cache_hit: float
    input_cache_miss: float
    output: float
    off_peak_factor: float = 1.0  # 非高峰时段的价格系数，1.0 表示不区分时段


# DeepSeek 价格来自 https://api-docs.deepseek.com/quick_start/pricing（2026-09-30 查询）。
# 硅基流动的 BAAI/bge-m3 与 BAAI/bge-reranker-v2-m3 为免费模型。
_DEEPSEEK_FLASH = Price(
    input_cache_hit=0.006, input_cache_miss=0.3, output=1.2, off_peak_factor=0.5
)
PRICES: dict[str, Price] = {
    "deepseek-flash": _DEEPSEEK_FLASH,
    "deepseek-v4-flash": _DEEPSEEK_FLASH,  # 旧名称，仍按 Flash 计费
    "BAAI/bge-m3": Price(0.0, 0.0, 0.0),
    "BAAI/bge-reranker-v2-m3": Price(0.0, 0.0, 0.0),
}


def is_deepseek_peak(at: datetime) -> bool:
    """DeepSeek 高峰时段：周一至周五 UTC 01:00–04:00 与 06:00–10:00。

    法定节假日同样按非高峰计费，这里没有处理节假日，所以节假日的估算会偏高。
    """
    t = at.astimezone(UTC)
    if t.weekday() >= 5:
        return False
    return 1 <= t.hour < 4 or 6 <= t.hour < 10


def estimate_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    at: datetime | None = None,
) -> float | None:
    """估算一次调用的美元费用；单价未知时返回 None。"""
    price = PRICES.get(model)
    if price is None:
        return None
    miss = max(input_tokens - cached_input_tokens, 0)
    cost = (
        cached_input_tokens * price.input_cache_hit
        + miss * price.input_cache_miss
        + output_tokens * price.output
    ) / 1_000_000
    if price.off_peak_factor != 1.0 and not is_deepseek_peak(at or datetime.now(UTC)):
        cost *= price.off_peak_factor
    return cost


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None 表示该模型单价未知


class UsageTracker:
    """按模型累计调用次数、token 与费用。线程安全，可在多个客户端之间共享。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, ModelUsage] = {}

    def record(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        at: datetime | None = None,
    ) -> float | None:
        cost = estimate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            at=at,
        )
        with self._lock:
            usage = self._by_model.setdefault(model, ModelUsage())
            usage.calls += 1
            usage.input_tokens += input_tokens
            usage.cached_input_tokens += cached_input_tokens
            usage.output_tokens += output_tokens
            if cost is None or usage.cost_usd is None:
                usage.cost_usd = None
            else:
                usage.cost_usd += cost
        return cost

    def snapshot(self) -> dict[str, ModelUsage]:
        with self._lock:
            return {model: replace(usage) for model, usage in self._by_model.items()}

    @property
    def total_cost_usd(self) -> float:
        """所有单价已知模型的费用合计。"""
        return sum(u.cost_usd for u in self.snapshot().values() if u.cost_usd is not None)
