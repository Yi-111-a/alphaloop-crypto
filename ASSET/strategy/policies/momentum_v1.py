"""
ASSET/strategy/policies/momentum_v1.py —— "动量战术"的确定性代码翻译。

原战术文字(scripts/ignite.py _DEFAULT_EVO_TACTICS["evo/20260714-momentum"])
核心思想:优先关注最近若干个决策周期内价格单方向变动幅度最大的标的,顺势
而非逆势建仓;必须为每笔决策写清楚"趋势可能反转"的证伪条件。

本实现:
  1. 对 ctx.snapshot 里每个有足够历史(>=REQUIRED_HISTORY_BARS根K线)的symbol,
     计算最近 _MOMENTUM_LOOKBACK 根K线的收益率(动量)。
  2. 选出|动量|最大的那个symbol;如果它的|动量|达到 _MOMENTUM_THRESHOLD 阈值,
     顺势开仓(动量为正开多,为负开空);否则本周期不操作(返回空列表)。
  3. 止损位(falsifier_condition)设在"入场价反向 _ATR_MULT 倍ATR"处——ATR
     用最近 _ATR_WINDOW 根K线的真实波幅均值衡量,这正是"趋势反转"的量化定义:
     价格反向运行超过近期波动幅度的若干倍,原有的"顺势延续"假设就应视为
     已经被证伪。
  4. 纯函数、确定性:所有输入只来自 ctx,没有随机性/墙钟/网络调用。
"""
from __future__ import annotations

import pandas as pd

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

_MOMENTUM_LOOKBACK = 20  # 近20根K线(§原战术:"最近若干个决策周期")
_ATR_WINDOW = 14
_ATR_MULT = 2.0  # 止损=入场价反向2倍ATR
_MOMENTUM_THRESHOLD = 0.03  # 20bar收益率绝对值需>=3%才判定为"明显单边趋势"
_LEVERAGE = 3
_TARGET_NOTIONAL_PCT = 20.0
_HORIZON = "12h"

# 需要同时覆盖"计算20bar收益率"(需要 lookback+1 根)与"计算ATR"
# (需要至少 _ATR_WINDOW+1 根,以便有 prev_close 可用)两个信号的最低历史长度。
REQUIRED_HISTORY_BARS = max(_MOMENTUM_LOOKBACK + 1, _ATR_WINDOW + 1)

DESCRIPTION = (
    "动量战术(momentum):顺势跟随近期单边趋势最明显的标的建仓,而不是像"
    "main分支那样只押BTC低波动锚定逻辑。逐symbol计算最近20根K线收益率,"
    "选|收益率|最大且超过3%阈值的symbol,方向与该收益率符号一致(正->做多,"
    "负->做空)。止损(falsifier_condition)设在入场价反向2倍ATR(14根K线"
    "真实波幅均值)处,量化定义'趋势反转'。杠杆3倍、目标名义仓位20%净值,"
    "无标的达到阈值时本周期不操作(返回空列表)。"
)


def _momentum(df: "pd.DataFrame", n: int) -> float:
    """最近n根K线的收益率:(最新收盘 - n根之前的收盘) / n根之前的收盘。"""
    closes = df["close"]
    latest = float(closes.iloc[-1])
    base = float(closes.iloc[-1 - n])
    if base == 0:
        return 0.0
    return (latest - base) / base


def _atr(df: "pd.DataFrame", n: int) -> float:
    """最近n根K线的真实波幅(True Range)均值,衡量近期波动幅度。"""
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

    # sorted() 保证在dict迭代顺序意外变化时输出仍然确定。
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
        f"绝对值超过{_MOMENTUM_THRESHOLD:.0%}顺势阈值,判断短期{direction_word}"
        f"单边趋势大概率延续,顺势而非逆势建仓,不追逐次强信号标的。"
    )
    falsifier = (
        f"若{best_symbol}价格反向运行至入场价{last_close:.6f}反向"
        f"{_ATR_MULT:g}倍ATR({atr:.6f})对应的{stop_price:.6f}价位,"
        f"视为原有单边趋势已反转,本笔动量假设证伪,应立即平仓离场。"
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
