"""
ASSET/strategy/policies/diversified_v1.py —— "分散战术"的确定性代码翻译。

原战术文字(scripts/ignite.py _DEFAULT_EVO_TACTICS["evo/20260714-diversified"])
核心思想:每个决策周期尽量在universe内多个不相关标的上分别建立小仓位,而
不是像main分支那样长期只集中在BTC一个标的上;单个标的仓位规模明显小于
main/aggressive分支,用"广撒网、小额验证多个假设"的方式积累各标的真实
表现数据。

(注:MEMORY.md 记录过"Trader不应该默认只交易BTC,应该在universe内分散"
这条反馈——本策略与该原则天然一致:下面的实现不对symbol做任何BTC优先的
特殊处理,对ctx.snapshot里每一个有足够历史的symbol一视同仁地计算信号。)

本实现:对 ctx.snapshot 里**每一个**有足够历史(>=REQUIRED_HISTORY_BARS根
K线)的symbol,各自独立计算最近 _MOMENTUM_LOOKBACK 根K线的动量;只要
|动量|超过一个很低的门槛(明显低于momentum_v1的3%,体现"广撒网"而不是
"只挑最强信号"),就在该symbol上按动量方向开一个小仓位。symbol数量不设
上限——只要通过门槛都开,分散度由universe本身的大小决定。
"""
from __future__ import annotations

import pandas as pd

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

_MOMENTUM_LOOKBACK = 20
_ATR_WINDOW = 14
_ATR_MULT = 2.0
_MOMENTUM_THRESHOLD = 0.005  # 很低的门槛:广撒网验证多个假设,而不是只挑最强信号
_LEVERAGE = 2  # 单标的仓位小,杠杆保守
_TARGET_NOTIONAL_PCT = 5.0  # 明显小于momentum(20%)/aggressive(35%)分支
_HORIZON = "1d"

REQUIRED_HISTORY_BARS = max(_MOMENTUM_LOOKBACK + 1, _ATR_WINDOW + 1)

DESCRIPTION = (
    "分散战术(diversified):每个决策周期尽量在universe内多个标的上分别"
    "建立小仓位,而不是像main分支那样长期集中在单一标的。对ctx.snapshot"
    "里每一个有足够历史的symbol独立计算最近20根K线动量,只要|动量|超过"
    "0.5%的低门槛(明显低于momentum_v1的3%,体现'广撒网'而非'只挑最强"
    "信号')就顺势开一个小仓位(杠杆2倍、目标名义仓位5%净值/标的)。"
    "止损(falsifier_condition)设在入场价反向2倍ATR处。symbol数量不设"
    "上限,用小额多标的的方式为后续Reflector反思积累更丰富的样本。"
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
    decisions: list[Decision] = []

    # sorted() 保证多标的开仓的产出顺序确定,不依赖dict插入顺序。
    for symbol in sorted(ctx.snapshot.keys()):
        df = ctx.recent_bars.get(symbol)
        if df is None or len(df) < REQUIRED_HISTORY_BARS:
            continue

        mom = _momentum(df, _MOMENTUM_LOOKBACK)
        if abs(mom) < _MOMENTUM_THRESHOLD:
            continue  # 该标的没有哪怕微弱的方向性信号,不勉强开仓

        last_close = float(df["close"].iloc[-1])
        atr = _atr(df, _ATR_WINDOW)

        if mom > 0:
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
            f"{symbol}最近{_MOMENTUM_LOOKBACK}根K线动量为{mom:.2%},超过"
            f"{_MOMENTUM_THRESHOLD:.1%}的低确认门槛,判断短期{direction_word}"
            f"倾向成立;作为分散战术的一部分,用小仓位在多个不相关标的上"
            f"分别验证这类假设,而不是集中押注单一标的。"
        )
        falsifier = (
            f"若{symbol}价格反向运行至入场价{last_close:.6f}反向"
            f"{_ATR_MULT:g}倍ATR({atr:.6f})对应的{stop_price:.6f}价位,"
            f"视为该标的这次小仓位方向性假设已经证伪,应立即平仓,"
            f"不影响其他标的各自独立的仓位判断。"
        )

        decisions.append(
            Decision(
                ts=ctx.ts,
                symbol=symbol,
                action=action,
                target_notional_pct=_TARGET_NOTIONAL_PCT,
                leverage=_LEVERAGE,
                thesis=thesis,
                falsifier=falsifier,
                horizon=_HORIZON,
                falsifier_condition=falsifier_condition,
            )
        )

    return decisions
