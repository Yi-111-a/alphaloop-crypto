"""
ASSET/strategy/policies/carry_v1.py —— "资金费率carry战术"的确定性代码翻译
(代理实现,见下方"已知限制"说明)。

原战术文字(scripts/ignite.py _DEFAULT_EVO_TACTICS["evo/20260714-carry"])
核心思想:优先寻找资金费率长期偏离零、且价格没有极端单边趋势的标的做反向
持仓吃资金费率;这个分支的核心假设是carry收益本身、不是赌方向,仓位方向
应该跟资金费率符号相反。

已知限制(代理实现,如实说明): 本版 StrategyContext(ASSET/strategy/policies/
__init__.py)还没有 funding 字段——ctx 里只有 ts/positions/snapshot/
recent_bars/memory_context,没有资金费率历史。真正的carry战术依赖"资金费率
符号"来决定方向,这个信息现在拿不到。

因此 v1 用"低波动 + 区间震荡"作为carry的代理信号:资金费率长期偏离零、且
价格没有极端趋势的市场,在价格行为上往往表现为窄幅震荡(低已实现波动率、
价格在近期区间内来回而不是单边突破)——这跟真正的carry交易环境有相关性但
不是同一件事,只是在缺数据时"能做的最接近的事"。等 ctx 扩展 funding 字段
后,应该把下面的"区间震荡代理信号"替换/补充成真实的"资金费率符号 + 幅度"
判断,并把方向逻辑改成显式的"仓位方向与资金费率符号相反"。

代理信号的具体做法:衡量最近 _WINDOW 根K线的(已实现波动率, 收盘价在区间
内的相对位置)。波动率低于阈值(判定为"没有极端单边趋势")时,若收盘价
处于区间上沿则判断为"震荡到顶,均值回归向下"开空;处于区间下沿则开多——
这本质是一个保守的均值回归信号,配合小仓位、低杠杆,与carry策略"追求
稳定小额收益、不赌方向"的精神一致,即便信号来源不是真实资金费率。
"""
from __future__ import annotations

import pandas as pd

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

_WINDOW = 20  # 衡量波动率与区间位置的窗口
_VOLATILITY_THRESHOLD = 0.012  # 逐bar收益率标准差阈值,低于此值视为"低波动/无极端趋势"
_RANGE_EDGE = 0.20  # 收盘价落在区间[0, _RANGE_EDGE]视为下沿,[1-_RANGE_EDGE, 1]视为上沿
_LEVERAGE = 2  # carry追求稳定小额收益,不赌方向,杠杆保守
_TARGET_NOTIONAL_PCT = 10.0
_HORIZON = "1d"

REQUIRED_HISTORY_BARS = _WINDOW + 1

DESCRIPTION = (
    "资金费率carry战术的代理实现(carry proxy):真实carry交易的方向应该由"
    "资金费率符号决定,但当前StrategyContext还没有funding字段,ctx里拿不到"
    "资金费率历史。v1用'低波动+区间震荡'作为代理信号——衡量最近20根K线的"
    "已实现波动率(逐bar收益率标准差)与收盘价在区间内的相对位置:波动率低于"
    "1.2%阈值(无极端单边趋势)且收盘价处于区间上沿->判断震荡到顶、均值回归"
    "开空;处于区间下沿->开多。杠杆2倍、目标名义仓位10%净值,与carry'追求"
    "稳定小额收益、不赌方向'的精神一致。等ctx扩展真实funding字段后,应升级"
    "为'仓位方向与资金费率符号相反'的真实实现,当前只是波动率结构的代理。"
)


def _realized_volatility(df: "pd.DataFrame", n: int) -> float:
    """最近n根K线逐bar收益率的样本标准差,衡量已实现波动率。"""
    closes = df["close"].tail(n + 1)
    returns = closes.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std())


def _range_position(df: "pd.DataFrame", n: int) -> tuple[float, float, float]:
    """返回 (relative_position, range_low, range_high)。

    relative_position: 最新收盘价在[区间最低, 区间最高]里的相对位置,
    0=贴着区间下沿,1=贴着区间上沿。区间宽度为0(极端无波动)时返回0.5,
    视为"既不靠上沿也不靠下沿"。
    """
    window = df.tail(n)
    low = float(window["low"].min())
    high = float(window["high"].max())
    last_close = float(df["close"].iloc[-1])
    span = high - low
    if span <= 0:
        return 0.5, low, high
    rel = (last_close - low) / span
    return rel, low, high


def decide(ctx: StrategyContext) -> list[Decision]:
    best_symbol: str | None = None
    best_extremeness = 0.0  # 离区间中点的距离,用来在多个候选里选最典型的震荡标的
    best_rel = 0.5
    best_low = 0.0
    best_high = 0.0
    best_vol = 0.0

    for symbol in sorted(ctx.snapshot.keys()):
        df = ctx.recent_bars.get(symbol)
        if df is None or len(df) < REQUIRED_HISTORY_BARS:
            continue

        vol = _realized_volatility(df, _WINDOW)
        if vol > _VOLATILITY_THRESHOLD:
            continue  # 波动率不够低,不满足"无极端单边趋势"这个代理前提

        rel, low, high = _range_position(df, _WINDOW)
        if rel >= 1 - _RANGE_EDGE:
            extremeness = rel  # 越靠近区间上沿,均值回归做空的信号越强
        elif rel <= _RANGE_EDGE:
            extremeness = 1 - rel  # 越靠近区间下沿,均值回归做多的信号越强
        else:
            continue  # 处于区间中段,没有明显的均值回归方向

        if extremeness > best_extremeness:
            best_extremeness = extremeness
            best_symbol = symbol
            best_rel = rel
            best_low = low
            best_high = high
            best_vol = vol

    if best_symbol is None:
        return []

    df = ctx.recent_bars[best_symbol]
    last_close = float(df["close"].iloc[-1])

    if best_rel >= 1 - _RANGE_EDGE:
        action = "open_short"
        # 止损设在略高于本轮区间高点,若价格突破区间上沿代表"震荡"这个前提假设失效。
        stop_price = best_high * 1.01
        falsifier_condition = f"price>{stop_price:.6f}"
        edge_word = "上沿"
    else:
        action = "open_long"
        stop_price = best_low * 0.99
        falsifier_condition = f"price<{stop_price:.6f}"
        edge_word = "下沿"

    thesis = (
        f"{best_symbol}最近{_WINDOW}根K线已实现波动率仅{best_vol:.2%}"
        f"(低于{_VOLATILITY_THRESHOLD:.1%}阈值,判断无极端单边趋势),且"
        f"收盘价{last_close:.6f}贴近区间{edge_word}(区间[{best_low:.6f},"
        f"{best_high:.6f}]),作为资金费率数据缺失时的carry代理信号,"
        f"用均值回归小仓位捕捉区间震荡收益,不赌单边方向。"
    )
    falsifier = (
        f"若{best_symbol}价格突破本轮震荡区间,运行至{stop_price:.6f},"
        f"视为'无极端趋势/区间震荡'这一代理前提本身已经失效,carry代理"
        f"假设证伪,应立即平仓,不能因为仓位小、杠杆低就放松止损纪律。"
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
