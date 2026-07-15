"""
ASSET/strategy/policies/conservative_v1.py —— "保守战术"的确定性代码翻译。

原战术文字(scripts/ignite.py _DEFAULT_EVO_TACTICS["evo/20260714-conservative"])
核心思想:只在多条独立假设互相印证、且没有明显反向风险信号时才建仓;仓位
规模、杠杆都应明显低于main分支;宁可错过机会也不放大不确定性下的敞口。

本实现把"多条独立假设互相印证"具体化为两个独立信号必须同向:
  1. 动量信号:最近 _MOMENTUM_LOOKBACK 根K线收益率的符号与幅度(阈值明显低于
     momentum_v1,但要求下面第2条也同时成立才会真正建仓——单独任何一个信号
     都不够)。
  2. 趋势位置信号:最新收盘价相对 _SMA_WINDOW 根K线均线的位置(高于均线视为
     多头结构、低于均线视为空头结构)。

只有当两者方向一致(动量为正 且 价格在均线上方 -> 多头;动量为负 且 价格在
均线下方 -> 空头)时才建仓,否则视为"没有明显反向风险信号"这条件不成立,
本周期不操作。杠杆、仓位都取五个种子策略里最低的一档。
"""
from __future__ import annotations

import pandas as pd

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

_MOMENTUM_LOOKBACK = 20
_SMA_WINDOW = 50  # 均线窗口,用于判断"趋势结构"这第二个独立信号
_ATR_WINDOW = 14
_ATR_MULT = 2.5  # 止损留出比aggressive更宽的余地,配合更低杠杆/更小仓位
_MOMENTUM_THRESHOLD = 0.015  # 动量信号本身的确认门槛(需与均线信号同时成立)
_LEVERAGE = 2  # 五个种子策略里的最低档之一
_TARGET_NOTIONAL_PCT = 8.0  # 明显低于momentum/aggressive
_HORIZON = "1d"

REQUIRED_HISTORY_BARS = max(_MOMENTUM_LOOKBACK + 1, _SMA_WINDOW, _ATR_WINDOW + 1)

DESCRIPTION = (
    "保守战术(conservative):只在两条独立信号同时印证、且没有明显反向"
    "风险时才建小仓位,否则整体观望。信号1=最近20根K线动量(符号+幅度"
    "超过1.5%阈值);信号2=最新收盘价相对50根K线均线的位置(高于/低于)。"
    "只有动量方向与均线相对位置一致(动量为正且价格在均线上方->多头;"
    "动量为负且价格在均线下方->空头)才建仓,任一信号缺席或矛盾则本周期"
    "不操作(返回空列表)。杠杆2倍、目标名义仓位8%净值,均为五个种子策略"
    "里最低档之一,宁可错过机会也不放大不确定性下的敞口。"
)


def _momentum(df: "pd.DataFrame", n: int) -> float:
    closes = df["close"]
    latest = float(closes.iloc[-1])
    base = float(closes.iloc[-1 - n])
    if base == 0:
        return 0.0
    return (latest - base) / base


def _sma(df: "pd.DataFrame", n: int) -> float:
    return float(df["close"].tail(n).mean())


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
        if abs(mom) < _MOMENTUM_THRESHOLD:
            continue  # 信号1本身都不够格,不必再算信号2

        last_close = float(df["close"].iloc[-1])
        sma = _sma(df, _SMA_WINDOW)

        # 信号1、信号2必须同向,才算"多条独立假设互相印证"。
        if mom > 0 and last_close <= sma:
            continue
        if mom < 0 and last_close >= sma:
            continue

        if abs(mom) > best_abs_mom:
            best_abs_mom = abs(mom)
            best_mom = mom
            best_symbol = symbol

    if best_symbol is None:
        return []

    df = ctx.recent_bars[best_symbol]
    last_close = float(df["close"].iloc[-1])
    sma = _sma(df, _SMA_WINDOW)
    atr = _atr(df, _ATR_WINDOW)

    if best_mom > 0:
        action = "open_long"
        stop_price = last_close - _ATR_MULT * atr
        falsifier_condition = f"price<{stop_price:.6f}"
        structure_word = "均线上方(多头结构)"
    else:
        action = "open_short"
        stop_price = last_close + _ATR_MULT * atr
        falsifier_condition = f"price>{stop_price:.6f}"
        structure_word = "均线下方(空头结构)"

    thesis = (
        f"{best_symbol}最近{_MOMENTUM_LOOKBACK}根K线动量为{best_mom:.2%}"
        f"(超过{_MOMENTUM_THRESHOLD:.1%}确认阈值),且最新收盘价{last_close:.6f}"
        f"位于{_SMA_WINDOW}根均线{sma:.6f}的{structure_word},两条独立信号"
        f"互相印证、没有明显反向风险,才小仓位、低杠杆谨慎建仓。"
    )
    falsifier = (
        f"若{best_symbol}价格反向运行至入场价{last_close:.6f}反向"
        f"{_ATR_MULT:g}倍ATR({atr:.6f})对应的{stop_price:.6f}价位,"
        f"视为两条独立信号中至少一条已经失效,保守假设证伪,应立即平仓,"
        f"不因为仓位小就拖延止损。"
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
