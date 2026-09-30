from datetime import UTC, datetime

import pytest

from paperlens.models.usage import UsageTracker, estimate_cost, is_deepseek_peak

PEAK = datetime(2026, 9, 30, 2, 0, tzinfo=UTC)  # 周三 UTC 02:00
OFF_PEAK = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)  # 周三 UTC 12:00
WEEKEND = datetime(2026, 10, 3, 2, 0, tzinfo=UTC)  # 周六


def test_peak_hours():
    assert is_deepseek_peak(PEAK)
    assert is_deepseek_peak(datetime(2026, 9, 30, 9, 59, tzinfo=UTC))
    assert not is_deepseek_peak(datetime(2026, 9, 30, 4, 0, tzinfo=UTC))
    assert not is_deepseek_peak(OFF_PEAK)
    assert not is_deepseek_peak(WEEKEND)


def test_cost_peak_and_off_peak():
    # 100 万未命中输入 + 100 万输出，高峰：0.3 + 1.2
    peak = estimate_cost("deepseek-flash", input_tokens=1_000_000, output_tokens=1_000_000, at=PEAK)
    assert peak == pytest.approx(1.5)
    off = estimate_cost(
        "deepseek-flash", input_tokens=1_000_000, output_tokens=1_000_000, at=OFF_PEAK
    )
    assert off == pytest.approx(0.75)


def test_cost_uses_cache_hits():
    cost = estimate_cost(
        "deepseek-flash", input_tokens=1_000_000, cached_input_tokens=1_000_000, at=PEAK
    )
    assert cost == pytest.approx(0.006)


def test_unknown_model_cost_is_none():
    assert estimate_cost("some-unknown-model", input_tokens=10) is None


def test_tracker_accumulates():
    tracker = UsageTracker()
    tracker.record("deepseek-flash", input_tokens=1000, output_tokens=100, at=PEAK)
    tracker.record("deepseek-flash", input_tokens=500, output_tokens=50, at=PEAK)
    tracker.record("unknown", input_tokens=10)
    snap = tracker.snapshot()
    assert snap["deepseek-flash"].calls == 2
    assert snap["deepseek-flash"].input_tokens == 1500
    assert snap["unknown"].cost_usd is None
    assert tracker.total_cost_usd == pytest.approx((1500 * 0.3 + 150 * 1.2) / 1_000_000)
