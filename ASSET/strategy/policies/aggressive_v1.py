"""
ASSET/strategy/policies/aggressive_v1.py —— "进取战术"的确定性代码翻译。

原战术文字(scripts/ignite.py _DEFAULT_EVO_TACTICS["evo/20260714-aggressive"])
核心思想:在假设置信度足够时,愿意用更高杠杆、更集中的仓位捕捉机会,不要求
像main分支那样保守分散;但仍必须为每笔非hold决策写出明确的falsifier_condition
并严格执行,进取不等于不设止损。

本实现与 momentum_v1 的关键差异,正是要体现"单信号确认即可、仓位/杠杆更高"
这句话:
  - 确认门槛更低(_MOMENTUM_THRESHOLD 比 momentum_v1 更小):单一动量信号
    只要略超噪声水平就足够进场,不像 conservative_v1 那样要求多信号互相印证。
  - 杠杆(_LEVERAGE)与目标仓位(_TARGET_NOTIONAL_PCT)都显著高于 momentum_v1。
  - 止损依然是ATR倍数、依然会产出 falsifier_condition——"进取"体现在信号
    门槛和仓位大小上,不是取消止损纪律。
"""
from __future__ import annotations

import pandas as pd

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

_MOMENTUM_LOOKBACK = 10  # 更短的观察窗口,反应更快、进场门槛更低
_ATR_WINDOW = 14
_ATR_MULT = 1.5  # 止损收得比momentum_v1更紧,配合更高杠杆控制单笔风险敞口
_MOMENTUM_THRESHOLD = 0.012  # 单一信号即可确认,阈值明显低于momentum_v1的3%
_LEVERAGE = 5  # 五个种子策略里的较高档(spec要求整体仍保守在1-5区间内)
_TARGET_NOTIONAL_PCT = 35.0  # 更集中的仓位
_HORIZON = "8h"

REQUIRED_HISTORY_BARS = max(_MOMENTUM_LOOKBACK + 1, _ATR_WINDOW + 1)

DESCRIPTION = (
    "进取战术(aggressive):单一动量信号确认即可用更高杠杆、更集中的仓位"
    "捕捉机会,不要求像main/conservative分支那样多信号互相印证。逐symbol"
    "计算最近10根K线收益率,取|收益率|最大且超过1.2%阈值(明显低于momentum_v1"
    "的3%)的symbol顺势建仓,杠杆5倍、目标名义仓位35%净值(均高于momentum_v1)。"
    "止损(falsifier_condition)仍设在入场价反向1.5倍ATR处——进取只体现在"
    "信号门槛和仓位规模上,不等于取消止损纪律。无标的达到阈值时本周期"
    "不操作(返回空列表)。"
)


def _momentum(df: "pd.DataFrame", n: int) -> float:
    closes = df["close"]
    latest = float(closes.iloc[-1])
    base = float(closes.iloc[-1 - n])
    if base == 0:
        return 0.0
    return (latest - base) / base


def _atr(df: "pd.DataFrame", n: int) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tail = tr.tail(n).dropna()
    if len(tail) == 0:
        return 0.0
    return float(tail.mean())


def decide(ctx: StrategyContext) -> list[Decision]:
    best_symbol: str | None = None
    best_abs_mom = 0.0
    best_mom = 0.0

    for symbol in sorted(ctx.snapshot.keys()):
        df = ctx.recent_bars.get(symbol)
        if df is None or len(df) < REQUIRED_HISTORY_BARS:
            continue
        mom = _momentum(df, _MOMENTUM_LOOKBACK)
        if abs(mom) > best_abs_mom:
            best_abs_mom = abs(mom)
            best_mom = mom
            best_symbol = symbol

    if best_symbol is None or best_abs_mom < _MOMENTUM_THRESHOLD:
        return []

    df = ctx.recent_bars[best_symbol]
    last_close = float(df["close"].iloc[-1])
    atr = _atr(df, _ATR_WINDOW)

    if best_mom > 0:
        action = "open_long"
        stop_price = last_close - _ATR_MULT * atr
        falsifier_condition = f"price<{stop_price:.6f}"
        direction_word = "上涨"
    else:
        action = "open_short"
        stop_price = last_close + _ATR_MULT * atr
        falsifier_condition = f"price>{stop_price:.6f}"
        direction_word = "下跌"

    thesis = (
        f"{best_symbol}最近{_MOMENTUM_LOOKBACK}根K线收益率为{best_mom:.2%},"
        f"单一动量信号已超过{_MOMENTUM_THRESHOLD:.1%}的进取型确认阈值,"
        f"判断短期{direction_word}动能已经成型,用更高杠杆、更集中仓位"
        f"捕捉这次机会,不等待第二个独立信号印证。"
    )
    falsifier = (
        f"若{best_symbol}价格反向运行至入场价{last_close:.6f}反向"
        f"{_ATR_MULT:g}倍ATR({atr:.6f})对应的{stop_price:.6f}价位,"
        f"视为本次单信号确认的动能假设已经证伪,进取不等于不设止损,"
        f"应立即平仓离场,不死扛。"
    )

    return [
        Decision(
            ts=ctx.ts,
            symbol=best_symbol,
            action=action,
            target_notional_pct=_TARGET_NOTIONAL_PCT,
            leverage=_LEVERAGE,
            thesis=thesis,
            falsifier=falsifier,
            horizon=_HORIZON,
            falsifier_condition=falsifier_condition,
        )
    ]
