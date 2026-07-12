from __future__ import annotations

import pytest

from LOCKED.schemas import Decision


@pytest.fixture
def log_root(tmp_path):
    """临时 LOG 区根目录,测试间互不污染。"""
    d = tmp_path / "LOG"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def make_decision():
    def _make(**overrides):
        base = dict(
            ts=1_700_000_000_000,
            symbol="BTC/USDT:USDT",
            action="open_long",
            target_notional_pct=50.0,
            leverage=3,
            thesis="基于H1:BTC波动率处于历史低位,预期突破概率上升",
            falsifier="若4h收盘价跌破前低支撑位则判断此假设失效",
            horizon="12h",
            branch="main",
        )
        base.update(overrides)
        return Decision(**base)

    return _make
