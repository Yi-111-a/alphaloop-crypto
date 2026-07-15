"""
tests/test_carry_v1_funding.py —— carry_v1 双模式升级(真carry优先 / 低波动
代理回退)验收测试。

覆盖范围:
  1. 合成"近10天>=30条、显著为正"的资金费率历史 -> 断言产出 open_short
     (仓位方向与费率符号相反,做空吃正费率)。
  2. 合成"近10天>=30条、显著为负"的资金费率历史 -> 断言产出 open_long
     (谨慎做多吃负费率),且比正费率分支更保守(仓位/杠杆更低)。
  3. 资金费率数据不足(条数不够/整个symbol没有key/费率接近零)时,decide()
     的输出必须与直接调用 _fallback(ctx) 逐字段相等——回退行为与升级前的
     v1完全一致,不能因为本次升级而在数据不足时产生新的行为分叉。
  4. 真carry分支的输出同样满足 Simulator 合法性校验(与 tests/test_policies.py
     里 _validate_decision 等价的规则),以及确定性(同一ctx连续调用两次
     逐字段相同)。

复用 tests/test_policies.py 里已有的合成K线构造工具 (_make_bars)、Simulator
合法性校验 helper (_validate_decision) 和 max_leverage 常量,避免重复造轮子
(tests/ 是一个真正的package,见 tests/__init__.py,可以互相import)。
"""
from __future__ import annotations

import pandas as pd
import pytest

from ASSET.strategy.policies import StrategyContext, load_policy
from tests.test_policies import _MAX_LEVERAGE, _make_bars, _validate_decision

_INTERVAL_MS = 4 * 3600 * 1000  # 与项目4h K线周期一致,和test_policies.py同口径
_FUNDING_INTERVAL_MS = 8 * 3600 * 1000  # 真实资金费率结算周期(8h/次)

_SYMBOL = "AAA/USDT:USDT"

# 与carry_v1.py里的_RANGE_EDGE贴边pattern手法一致(见test_policies.py::
# _flat_ctx),末尾落在pattern[-1](贴近区间下沿)——用来在"数据不足回退"
# 场景下同时触发代理模式的均值回归信号,验证回退行为不是空转。
_FLAT_PATTERN = [0.0, 0.3, 0.6, 0.9, 0.6, 0.3, 0.0, -0.3, -0.6, -0.9]


def _flat_bars(n: int = 80) -> pd.DataFrame:
    assert n % len(_FLAT_PATTERN) == 0
    return _make_bars(n, base_price=100.0, drift_pct=0.0, flat_pattern=_FLAT_PATTERN)


def _make_funding_df(mean_rate: float, ts: int, n: int = 30, interval_ms: int = _FUNDING_INTERVAL_MS) -> pd.DataFrame:
    """构造n条资金费率记录,固定值mean_rate,时间戳升序、最后一条恰好等于ts
    (满足"只包含<=ctx.ts的记录"这条时间边界约定)。"""
    rows = [
        {"timestamp": ts - (n - 1 - i) * interval_ms, "funding_rate": mean_rate}
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=["timestamp", "funding_rate"])


def _make_ctx(recent_bars: dict[str, pd.DataFrame], recent_funding: dict[str, pd.DataFrame]) -> StrategyContext:
    snapshot = {symbol: {"last": float(df["close"].iloc[-1])} for symbol, df in recent_bars.items()}
    last_ts = max(int(df["timestamp"].iloc[-1]) for df in recent_bars.values())
    return StrategyContext(
        ts=last_ts + _INTERVAL_MS,
        positions={},
        snapshot=snapshot,
        recent_bars=recent_bars,
        memory_context=[],
        recent_funding=recent_funding,
    )


# ---------------------------------------------------------------------------
# 1. 显著正费率 -> open_short
# ---------------------------------------------------------------------------


def test_carry_v1_opens_short_on_significant_positive_funding():
    bars = _flat_bars(30)
    ctx_ts_probe = int(bars["timestamp"].iloc[-1]) + _INTERVAL_MS
    funding = _make_funding_df(mean_rate=0.0002, ts=ctx_ts_probe)  # 远超0.005%阈值
    ctx = _make_ctx({_SYMBOL: bars}, {_SYMBOL: funding})

    module = load_policy("carry_v1")
    decisions = module.decide(ctx)

    assert len(decisions) == 1
    d = decisions[0]
    assert d.symbol == _SYMBOL
    assert d.action == "open_short"
    assert d.falsifier_condition.startswith("price>")
    assert "资金费率" in d.thesis

    errors = _validate_decision(d, {_SYMBOL})
    assert errors == [], f"real-carry short decision failed validation: {errors}"


# ---------------------------------------------------------------------------
# 2. 显著负费率 -> open_long(谨慎,比做空方向更保守)
# ---------------------------------------------------------------------------


def test_carry_v1_opens_cautious_long_on_significant_negative_funding():
    bars = _flat_bars(30)
    ctx_ts_probe = int(bars["timestamp"].iloc[-1]) + _INTERVAL_MS
    funding = _make_funding_df(mean_rate=-0.0002, ts=ctx_ts_probe)
    ctx = _make_ctx({_SYMBOL: bars}, {_SYMBOL: funding})

    module = load_policy("carry_v1")
    decisions = module.decide(ctx)

    assert len(decisions) == 1
    d = decisions[0]
    assert d.symbol == _SYMBOL
    assert d.action == "open_long"
    assert d.falsifier_condition.startswith("price<")

    errors = _validate_decision(d, {_SYMBOL})
    assert errors == [], f"real-carry cautious-long decision failed validation: {errors}"

    # "谨慎"做多的具体量化:仓位/杠杆不应超过做空吃正费率那一侧。
    short_decision = module.decide(
        _make_ctx({_SYMBOL: bars}, {_SYMBOL: _make_funding_df(mean_rate=0.0002, ts=ctx_ts_probe)})
    )[0]
    assert d.target_notional_pct <= short_decision.target_notional_pct
    assert d.leverage <= short_decision.leverage


# ---------------------------------------------------------------------------
# 3. 费率接近零 -> 不触发真carry,回退代理逻辑
# ---------------------------------------------------------------------------


def test_carry_v1_near_zero_funding_falls_back_to_proxy():
    bars = _flat_bars(80)
    ctx_ts_probe = int(bars["timestamp"].iloc[-1]) + _INTERVAL_MS
    funding = _make_funding_df(mean_rate=0.000001, ts=ctx_ts_probe)  # 远低于0.005%阈值
    ctx = _make_ctx({_SYMBOL: bars}, {_SYMBOL: funding})

    module = load_policy("carry_v1")
    decisions = module.decide(ctx)
    fallback_decisions = module._fallback(ctx)

    assert decisions == fallback_decisions
    assert decisions, "expected proxy fallback to trade on the flat/range-bound synthetic data"


# ---------------------------------------------------------------------------
# 4. 数据不足(条数不够 / 整个symbol缺key)-> 回退代理逻辑,与旧行为一致
# ---------------------------------------------------------------------------


def test_carry_v1_falls_back_when_funding_record_count_insufficient():
    bars = _flat_bars(80)
    ctx_ts_probe = int(bars["timestamp"].iloc[-1]) + _INTERVAL_MS
    sparse_funding = _make_funding_df(mean_rate=0.0005, ts=ctx_ts_probe, n=5)  # 远少于30条门槛
    ctx = _make_ctx({_SYMBOL: bars}, {_SYMBOL: sparse_funding})

    module = load_policy("carry_v1")
    decisions = module.decide(ctx)
    fallback_decisions = module._fallback(ctx)

    assert decisions == fallback_decisions
    assert decisions, "expected proxy fallback to trade on the flat/range-bound synthetic data"


def test_carry_v1_falls_back_when_recent_funding_missing_entirely():
    bars = _flat_bars(80)
    ctx = _make_ctx({_SYMBOL: bars}, {})  # 该symbol在recent_funding里完全没有key

    module = load_policy("carry_v1")
    decisions = module.decide(ctx)
    fallback_decisions = module._fallback(ctx)

    assert decisions == fallback_decisions
    assert decisions


def test_carry_v1_treats_missing_recent_funding_attribute_like_empty_dict():
    """StrategyContext.recent_funding 有 default_factory=dict 兜底,但carry_v1
    读取时仍然用 getattr(ctx, 'recent_funding', {}) 兜底——用一个没有这个字段
    的裸对象(鸭子类型,模拟 LOCKED/backtest_engine.py 早期/其它调用方可能
    传入的更窄的ctx)验证这条防御性写法真的生效,不会 AttributeError。"""
    bars = _flat_bars(80)

    class _BareCtx:
        def __init__(self, ts, snapshot, recent_bars):
            self.ts = ts
            self.snapshot = snapshot
            self.recent_bars = recent_bars
            # 故意不设置 recent_funding 属性

    ts = int(bars["timestamp"].iloc[-1]) + _INTERVAL_MS
    bare_ctx = _BareCtx(ts=ts, snapshot={_SYMBOL: {"last": float(bars["close"].iloc[-1])}}, recent_bars={_SYMBOL: bars})

    module = load_policy("carry_v1")
    decisions = module.decide(bare_ctx)
    assert decisions, "expected proxy fallback to still work when recent_funding attribute is absent"


# ---------------------------------------------------------------------------
# 5. 确定性:真carry分支同一ctx连续调用两次,逐字段相同
# ---------------------------------------------------------------------------


def test_carry_v1_real_carry_branch_is_deterministic():
    bars = _flat_bars(30)
    ctx_ts_probe = int(bars["timestamp"].iloc[-1]) + _INTERVAL_MS
    funding = _make_funding_df(mean_rate=0.0002, ts=ctx_ts_probe)
    ctx = _make_ctx({_SYMBOL: bars}, {_SYMBOL: funding})

    module = load_policy("carry_v1")
    decisions_1 = module.decide(ctx)
    decisions_2 = module.decide(ctx)
    assert decisions_1 == decisions_2


# ---------------------------------------------------------------------------
# 6. 多symbol候选:优先选均值费率绝对值最大的那个(信号最强)
# ---------------------------------------------------------------------------


def test_carry_v1_picks_symbol_with_strongest_funding_signal_among_candidates():
    bars_a = _flat_bars(30)
    bars_b = _make_bars(30, base_price=50.0, drift_pct=0.0, flat_pattern=_FLAT_PATTERN)
    ctx_ts_probe = int(bars_a["timestamp"].iloc[-1]) + _INTERVAL_MS

    funding_weak = _make_funding_df(mean_rate=0.00006, ts=ctx_ts_probe)  # 刚过阈值,信号弱
    funding_strong = _make_funding_df(mean_rate=0.0005, ts=ctx_ts_probe)  # 信号强得多

    ctx = _make_ctx(
        {"AAA/USDT:USDT": bars_a, "BBB/USDT:USDT": bars_b},
        {"AAA/USDT:USDT": funding_weak, "BBB/USDT:USDT": funding_strong},
    )

    module = load_policy("carry_v1")
    decisions = module.decide(ctx)

    assert len(decisions) == 1
    assert decisions[0].symbol == "BBB/USDT:USDT"
    assert decisions[0].action == "open_short"
