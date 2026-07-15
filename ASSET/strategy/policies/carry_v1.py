"""
ASSET/strategy/policies/carry_v1.py —— "资金费率carry战术"的确定性代码翻译
(双模式:真实资金费率carry优先,数据不足时回退到低波动区间代理)。

原战术文字(scripts/ignite.py _DEFAULT_EVO_TACTICS["evo/20260714-carry"])
核心思想:优先寻找资金费率长期偏离零、且价格没有极端单边趋势的标的做反向
持仓吃资金费率;这个分支的核心假设是carry收益本身、不是赌方向,仓位方向
应该跟资金费率符号相反。

历史限制与本次升级: v1最初写成时 StrategyContext(ASSET/strategy/policies/
__init__.py)还没有 funding 字段,只能用"低波动+区间震荡"作为代理信号。
现在 ctx 已经扩展出 recent_funding 字段(symbol -> 资金费率历史DataFrame,
只包含 <= ctx.ts 的记录,时间边界铁律与 recent_bars 同源),本文件因此
升级为两段式:

  1. 真carry模式(_decide_real_carry): 如果某个symbol在ctx.recent_funding
     里有足够多(>= _FUNDING_MIN_RECORDS 条)、落在最近 _FUNDING_LOOKBACK_DAYS
     天窗口内的真实资金费率记录,就用这些记录算近10天均值,按均值符号
     决定方向——费率长期为正(多头付给空头)时开空吃正费率;长期为负时
     "谨慎"开多吃负费率(仓位/杠杆比做空方向更保守,因为反向做多本身就是
     在对抗价格趋势的默认假设上叠加杠杆,需要更克制)。同时复用原有的
     已实现波动率过滤,只在"无极端单边趋势"的标的上出手——carry的前提是
     赚费率、不是赌方向,价格如果正在单边狂奔,均值意义上的费率carry很
     容易被方向性亏损吃掉。多个symbol都满足条件时,选均值费率绝对值最大
     的那个(carry信号最强),symbol名字升序排列作为确定性平局判定。
  2. 代理模式回退(_fallback,即原v1的完整实现,原样保留):没有任何
     symbol的真实资金费率数据够格进入真carry模式时,回退到"低波动+区间
     震荡"代理信号——衡量最近 _WINDOW 根K线的已实现波动率与收盘价在区间
     内的相对位置,波动率够低且贴近区间边缘就做均值回归。这一段与升级前
     的实现逐字节保持一致,是"数据不足时"的保底行为,不因为本次升级而
     改变既有回测/测试对代理模式的预期。

decide()的顶层逻辑就是"先试真carry,不行再退回代理",而不是两套逻辑各自
独立出决策再挑——真实资金费率信息一旦足够可信,就应该优先于代理信号。
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

# ---------------------------------------------------------------------------
# 真carry模式的参数
# ---------------------------------------------------------------------------

# 判定"这个symbol的资金费率数据足够可信"的最低条数——资金费率结算周期
# 8h/条,一天3条,30条约等于10天,与下面 _FUNDING_LOOKBACK_DAYS 窗口自洽
# (要求这10天窗口内至少见过30条记录,数据不能是稀疏/断档的)。
_FUNDING_MIN_RECORDS = 30
_FUNDING_LOOKBACK_DAYS = 10
_FUNDING_LOOKBACK_MS = _FUNDING_LOOKBACK_DAYS * 24 * 3600 * 1000

# 均值资金费率的显著性阈值:0.005%/8h(decimal形式 5e-5)。真实费率通常
# 在 -0.03% ~ +0.03%/8h 之间波动,0.005%这个量级足以过滤掉"接近零、噪声
# 主导"的情况,同时不会高到把大多数真实偏离都当作噪声滤掉——这是本次升级
# 自行拍板的阈值,写在这里方便后续回测/进化时按经验调整。
_FUNDING_RATE_THRESHOLD = 0.00005

# "谨慎做多吃负费率"相对于"做空吃正费率"的仓位折扣:反向做多不仅要赚
# 费率,还额外背上了"价格可能持续下跌"的方向性风险(即便过了波动率过滤,
# 波动率低不代表没有缓慢阴跌的单边趋势),所以用半仓、半杠杆保持保守,
# 这是本次升级自行拍板的"谨慎"具体量化方式。
_CAUTIOUS_LONG_NOTIONAL_PCT = _TARGET_NOTIONAL_PCT / 2
_CAUTIOUS_LONG_LEVERAGE = max(1, _LEVERAGE // 2)

DESCRIPTION = (
    "资金费率carry战术,双模式:(1) 真carry——某symbol近10天有>=30条真实"
    "资金费率记录时,用均值费率符号决定方向:均值>0.005%/8h(多头付空头)"
    "且价格无极端单边趋势(复用低波动过滤)->开空吃正费率;均值<-0.005%/8h"
    "->谨慎开多吃负费率(半仓半杠杆,比做空方向更保守,因为反向做多要额外"
    "承担方向性风险)。(2) 代理回退——没有symbol的真实费率数据够格时,退回"
    "'低波动+区间震荡'代理信号:已实现波动率低于1.2%阈值且收盘价贴近区间"
    "边缘,判断均值回归方向。两种模式下杠杆均保守(2倍或以下)、目标仓位"
    "不超过净值10%,与carry'追求稳定小额收益、不赌方向'的精神一致。"
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


# ---------------------------------------------------------------------------
# 真carry模式
# ---------------------------------------------------------------------------


def _eligible_funding_window(fdf: "pd.DataFrame", ts: int) -> "pd.DataFrame":
    """从某symbol的完整资金费率历史里截取"最近_FUNDING_LOOKBACK_DAYS天"
    这一子窗口(ctx.recent_funding本身已经满足"<=ts"的时间边界,这里只是
    在其基础上再裁掉太老的部分,不涉及任何未来数据)。"""
    if fdf is None or fdf.empty:
        return fdf if fdf is not None else pd.DataFrame(columns=["timestamp", "funding_rate"])
    window_start = ts - _FUNDING_LOOKBACK_MS
    return fdf[fdf["timestamp"] >= window_start]


def _decide_real_carry(ctx: StrategyContext) -> Decision | None:
    """尝试真carry逻辑,找不到够格的symbol时返回None(调用方据此回退到代理)。"""
    # getattr兜底:ctx有可能是鸭子类型的旧版本/没有这个字段的调用方构造出来的
    # 对象(见 ASSET/strategy/policies/__init__.py::StrategyContext.recent_funding
    # 文档字符串里对这个写法的说明)。
    funding_map = getattr(ctx, "recent_funding", {}) or {}

    best_symbol: str | None = None
    best_abs_rate = 0.0
    best_mean_rate = 0.0
    best_low = 0.0
    best_high = 0.0
    best_vol = 0.0

    for symbol in sorted(ctx.snapshot.keys()):
        fdf = funding_map.get(symbol)
        eligible = _eligible_funding_window(fdf, ctx.ts)
        if eligible is None or len(eligible) < _FUNDING_MIN_RECORDS:
            continue  # 这个symbol的真实费率数据不够格,留给代理模式处理

        df = ctx.recent_bars.get(symbol)
        if df is None or len(df) < REQUIRED_HISTORY_BARS:
            continue  # 没有足够K线做波动率过滤,不能确认"无极端单边趋势"

        vol = _realized_volatility(df, _WINDOW)
        if vol > _VOLATILITY_THRESHOLD:
            continue  # 价格正在单边跑,carry的前提(赚费率不赌方向)不成立

        mean_rate = float(eligible["funding_rate"].mean())
        if abs(mean_rate) <= _FUNDING_RATE_THRESHOLD:
            continue  # 费率接近零,不动

        abs_rate = abs(mean_rate)
        if abs_rate > best_abs_rate:
            best_abs_rate = abs_rate
            best_mean_rate = mean_rate
            best_symbol = symbol
            _, low, high = _range_position(df, _WINDOW)
            best_low, best_high = low, high
            best_vol = vol

    if best_symbol is None:
        return None

    df = ctx.recent_bars[best_symbol]
    last_close = float(df["close"].iloc[-1])

    if best_mean_rate > 0:
        # 长期正费率:多头持续付给空头,开空吃这份费率。
        action = "open_short"
        notional_pct = _TARGET_NOTIONAL_PCT
        leverage = _LEVERAGE
        stop_price = best_high * 1.01
        falsifier_condition = f"price>{stop_price:.6f}"
        direction_word = "开空"
    else:
        # 长期负费率:空头持续付给多头,"谨慎"开多——半仓半杠杆,因为这是
        # 在对抗默认的"不赌方向"精神之外,额外承担了一份方向性风险。
        action = "open_long"
        notional_pct = _CAUTIOUS_LONG_NOTIONAL_PCT
        leverage = _CAUTIOUS_LONG_LEVERAGE
        stop_price = best_low * 0.99
        falsifier_condition = f"price<{stop_price:.6f}"
        direction_word = "谨慎开多"

    thesis = (
        f"{best_symbol}最近{_FUNDING_LOOKBACK_DAYS}天真实资金费率均值为"
        f"{best_mean_rate:.6%}/8h(超过{_FUNDING_RATE_THRESHOLD:.4%}显著性"
        f"阈值),且最近{_WINDOW}根K线已实现波动率仅{best_vol:.2%}"
        f"(低于{_VOLATILITY_THRESHOLD:.1%}阈值,无极端单边趋势),"
        f"{direction_word}吃资金费率:仓位方向与费率符号相反,赚的是费率"
        f"本身,不是赌价格方向。"
    )
    falsifier = (
        f"若{best_symbol}价格运行至{stop_price:.6f},视为'无极端单边趋势'"
        f"这一carry前提本身已经失效,费率carry假设证伪,应立即平仓,不能"
        f"因为吃的是费率收益就放松止损纪律。"
    )

    return Decision(
        ts=ctx.ts,
        symbol=best_symbol,
        action=action,
        target_notional_pct=notional_pct,
        leverage=leverage,
        thesis=thesis,
        falsifier=falsifier,
        horizon=_HORIZON,
        falsifier_condition=falsifier_condition,
    )


# ---------------------------------------------------------------------------
# 代理模式回退(原v1实现,逐字节保留)
# ---------------------------------------------------------------------------


def _fallback(ctx: StrategyContext) -> list[Decision]:
    """低波动+区间震荡代理信号,升级前的原始carry_v1完整实现。真实资金费率
    数据不足以支撑_decide_real_carry时,统一退回这里——保持与升级前完全
    一致的行为,不因为本次升级而改变代理模式下的既有回测/测试预期。"""
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


def decide(ctx: StrategyContext) -> list[Decision]:
    real_decision = _decide_real_carry(ctx)
    if real_decision is not None:
        return [real_decision]
    return _fallback(ctx)
