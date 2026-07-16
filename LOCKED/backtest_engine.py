"""
backtest_engine.py —— 快速历史回测引擎(M6)。

职责:给定一个策略函数(strategy_fn)、一组标的、一组时间窗口,逐bar驱动
LOCKED/simulator.py 完整走一遍撮合/资金费率结算/爆仓校验,产出每个窗口的
BacktestResult,并按防过拟合规则打一个单一分数(score())。

铁律相关(§0,与本项目其它 LOCKED 模块一致):
- 本模块不下任何真实订单,只是把已经独立测试过的 LOCKED 模块
  (DataPipeline / Simulator / CircuitBreaker / BTCHoldAgent / FakeClock)
  接起来离线回放历史数据,不新增任何交易所写入接口。
- 所有日志/数据库写入必须落在调用方传入的 scratch_root 下面
  (scratch_root/{experiment_id}/{window.label}/...),绝不碰生产 LOG/ 和
  state/ 目录——回测是研究工具,不能污染真实运行时的账本和审计日志。

M6 授权范围:本文件是新增文件(不在原 LOCKED 区人类锁定范围内),另外
唯一被授权修改的 LOCKED 区文件是 data_pipeline.py 的资金费率分页缺陷
(见该文件的 M6 注释)。本文件不修改 simulator.py/schemas.py 等任何既有
LOCKED 模块一行代码,只是按它们已经公开的构造签名/方法去调用。

StrategyContext 协议(鸭子类型,M7 会正式定义完整协议,这里只按当前已知
的最小字段集构造一个轻量 dataclass 喂给 strategy_fn):
    ts: int                                  # 决策时刻(UTC ms)
    positions: dict[str, PerpPosition]       # 当前持仓快照(只读语义上的拷贝)
    snapshot: dict[str, dict]                # {symbol: {"last": 最新收盘价}}
    recent_bars: dict[str, pd.DataFrame]     # {symbol: 截至当前bar为止的最近K线}
    recent_funding: dict[str, pd.DataFrame]  # {symbol: 截至当前bar为止的资金费率历史}
                                              # (M7资金费率感官注入,见下方"资金
                                              # 费率注入"说明;带default_factory,
                                              # 不传不炸,与 ASSET/strategy/
                                              # policies/__init__.py::
                                              # StrategyContext.recent_funding
                                              # 同源约定)。
    recent_spot: dict[str, pd.DataFrame]     # {symbol: 截至当前bar为止的现货K线}
                                              # (M9现货溢价感官注入,列同
                                              # recent_bars——basis=perp_close/
                                              # spot_close-1由策略自己算,ctx只
                                              # 给原料。带default_factory,与
                                              # recent_funding同源约定)。
    recent_oi: dict[str, pd.DataFrame]       # {symbol: 截至当前bar为止的持仓量
                                              # (OI)历史,列["timestamp",
                                              # "open_interest"]}(M9 OI感官
                                              # 注入,带default_factory，同源
                                              # 约定)。

资金费率/现货/OI注入(时间边界铁律): 本引擎原本就为资金费率结算(Simulator.
settle_funding)加载了每个symbol的资金费率历史(见 _load_funding_frames /
_build_funding_lookup)。M7升级carry_v1等策略需要在ctx里也能看到这份数据,
这里复用同一份已加载的DataFrame(不为了喂ctx而重新拉取/重新读parquet),
用 _FundingCursor 按主时间线cur_ts做单调游标切片,只把"时间戳 <= 当前bar
ts"的那部分喂给ctx.recent_funding——整份DataFrame不加切片直接塞进ctx是
未来函数(策略会在bar N看到bar N之后才发生的真实资金费率,回测就失去了
存在的意义),必须是切片视图。窗口内某symbol完全没有资金费率数据时,
ctx.recent_funding[symbol] 是一个空DataFrame(列仍是
["timestamp","funding_rate"]),不是该symbol缺key。

M9 新增的 recent_spot(现货K线)/recent_oi(持仓量历史)是完全同一套手法
的复制品:窗口开始时通过 DataPipeline.fetch_spot_ohlcv /
fetch_open_interest_history 一次性加载好各symbol的现货K线/OI历史,分别
用 _SpotCursor / _OICursor(与 _FundingCursor 逐行照抄同一个"只前进不
回退的单调游标"实现,仅docstring按各自数据语义改写)按cur_ts做切片,
时间边界铁律与recent_funding完全相同——ctx里绝不会出现
timestamp > cur_ts 的行。窗口内某symbol完全没有现货/OI数据时,
ctx.recent_spot[symbol]/ctx.recent_oi[symbol] 同样是列齐全的空DataFrame，
不是该symbol缺key。

已知的简化/工程判断(如实记录,供 M7 review 时对照):
- 多symbol对齐:以 symbols[0] 的时间戳为"主时间线"驱动整个回放循环,其余
  symbol 用 _SymbolCursor 按自己的时间戳序列独立前进、取"截至主时间线当前
  时刻为止最新一条已知K线"。这在本项目当前的实际数据(universe 内各币种
  在同一交易所、同一时间框架下抓取)下工作良好;若某 symbol 在主时间线的
  某个时刻还没有任何历史数据(比如上市晚于窗口起点),该 symbol 在
  snapshot/recent_bars 里就不会出现,对应的决策会因为找不到"下一根bar"
  被记为 rejection,而不是伪造一根不存在的K线去撮合。
- benchmark 固定用 "BTC/USDT:USDT"(与 LOCKED/baseline_agents.py
  BTCHoldAgent 的默认 symbol 一致),不依赖调用方传入的 symbols 列表里
  是否包含 BTC——基准尺子本来就该是一个固定的、不受策略标的选择影响的参照物。
- recent_bars 做了一个有界回看窗口(_RECENT_BARS_MAX_LOOKBACK),而不是把
  从窗口开始到当前的全部历史都切片给策略——这是为了让回测在长窗口/多bar场景
  下仍然保持近似线性的时间复杂度,不是"能力还没被要求就先优化"式的过度设计,
  只是顺手把一个明显的 O(n^2) 切片成本苗头掐掉。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from LOCKED.baseline_agents import BTCHoldAgent
from LOCKED.circuit_breaker import CircuitBreaker
from LOCKED.clock import FakeClock
from LOCKED.schemas import Decision, PerpPosition, Rejection, Trade
from LOCKED.simulator import Simulator

logger = logging.getLogger(__name__)

_BENCHMARK_SYMBOL = "BTC/USDT:USDT"
# 回测不拉取真实24h成交额,给一个明显高于 simulator._slippage_bps_for 滑点
# 加倍门槛(SLIPPAGE_DOUBLE_VOLUME_THRESHOLD_USDT=5亿USDT)的哨兵值,避免
# "缺数据导致滑点被意外加倍"污染回测结果——与 scripts/m1_live_shakedown.py
# 的既有处理方式一致。
_VOLUME_SENTINEL_USDT = 1e9
# recent_bars 回看上限(见模块 docstring 的工程判断说明)。
_RECENT_BARS_MAX_LOOKBACK = 200
# M9:recent_spot 与 recent_bars 列同构(现货K线),recent_oi 列见
# LOCKED.data_pipeline.OI_COLUMNS。本文件一贯不跨模块 import data_pipeline
# 的列常量(与 _timeframe_to_ms 就地复刻同一原则,避免对其内部实现细节
# 产生耦合),这里就地声明同构的列名列表。
_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_OI_COLUMNS = ["timestamp", "open_interest"]


def _timeframe_to_ms(timeframe: str) -> int:
    """"4h" -> 14400000,与 data_pipeline.py 的同名私有工具函数逻辑一致
    (本文件不导入 data_pipeline 的私有函数,避免跨模块依赖内部实现细节,
    这里就地复刻这个几行的小工具)。"""
    unit_ms = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    unit = timeframe[-1]
    if unit not in unit_ms:
        raise ValueError(f"unsupported timeframe unit in {timeframe!r}: expected one of m/h/d")
    return int(timeframe[:-1]) * unit_ms[unit]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class BacktestWindow:
    label: str  # "train" / "val_1" / "val_2" / "holdout"
    start_ts: int  # UTC ms,含
    end_ts: int  # UTC ms,不含
    is_holdout: bool = False


@dataclass
class BacktestResult:
    window: BacktestWindow
    nav_series: list[tuple[int, float]]
    benchmark_nav_series: list[tuple[int, float]]
    return_pct: float
    max_drawdown_pct: float
    edge_vs_benchmark_pct: float
    branch_dead: bool
    trade_count: int
    rejection_count: int


@dataclass
class StrategyContext:
    """喂给 strategy_fn 的鸭子类型上下文对象,见模块 docstring 的协议说明。"""

    ts: int
    positions: dict[str, PerpPosition]
    snapshot: dict[str, dict]
    recent_bars: dict[str, pd.DataFrame]
    recent_funding: dict[str, pd.DataFrame] = field(default_factory=dict)
    recent_spot: dict[str, pd.DataFrame] = field(default_factory=dict)
    recent_oi: dict[str, pd.DataFrame] = field(default_factory=dict)


StrategyFn = Callable[[StrategyContext], list[Decision]]


# ---------------------------------------------------------------------------
# 内部工具:按主时间线前进的单symbol游标
# ---------------------------------------------------------------------------


class _SymbolCursor:
    """给一个 symbol 的原始K线序列一个"随主时间线单调前进"的指针查询:
    advance_to(ts) 返回"时间戳 <= ts 的最后一条K线"(没有则返回 None,代表
    该symbol此刻还没有任何历史数据)。指针只会前进不会回退,配合主时间线
    本身单调递增,整个回放过程里每个symbol的总前进步数不超过自己的K线总数,
    避免对每个bar都重新做一次全量查找/切片。"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._ts = self.df["timestamp"].to_numpy()
        self._idx = -1  # -1 = 尚未有任何已知K线

    def advance_to(self, ts: int) -> Optional[pd.Series]:
        n = len(self._ts)
        while self._idx + 1 < n and self._ts[self._idx + 1] <= ts:
            self._idx += 1
        if self._idx < 0:
            return None
        return self.df.iloc[self._idx]

    def recent_window(self) -> pd.DataFrame:
        """截至当前指针位置为止的最近一段K线(见 _RECENT_BARS_MAX_LOOKBACK)。"""
        if self._idx < 0:
            return self.df.iloc[0:0]
        start = max(0, self._idx - _RECENT_BARS_MAX_LOOKBACK + 1)
        return self.df.iloc[start : self._idx + 1].reset_index(drop=True)


class _FundingCursor:
    """同 _SymbolCursor 的单调游标手法,但用于资金费率历史序列:
    advance_to(ts) 返回"时间戳 <= ts 的资金费率历史"(按timestamp升序的
    DataFrame切片,可能为空)。指针只前进不回退,配合主时间线本身单调递增,
    对每个symbol的推进步数总共不超过该symbol的费率记录条数——不会每个bar
    都对整份历史重新做一次全量比较/切片,也不重新读盘(费率DataFrame在
    _run_window开始时已经从 data_pipeline 一次性加载好,这里只是复用同一份
    内存对象做视图切片)。这是M7资金费率注入的时间边界铁律的具体落点:
    advance_to(ts) 的返回值里绝不会出现 timestamp > ts 的行。"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("timestamp").reset_index(drop=True)
        self._ts = self.df["timestamp"].to_numpy()
        self._idx = -1  # -1 = 尚未有任何 timestamp <= ts 的费率记录

    def advance_to(self, ts: int) -> pd.DataFrame:
        n = len(self._ts)
        while self._idx + 1 < n and self._ts[self._idx + 1] <= ts:
            self._idx += 1
        if self._idx < 0:
            return self.df.iloc[0:0]
        return self.df.iloc[: self._idx + 1]


class _SpotCursor:
    """M9现货溢价感官注入:与 _FundingCursor 逐行照抄同一个"只前进不回退的
    单调游标"实现(本类不引用任何funding专属的列名,_FundingCursor本身就是
    列无关的纯timestamp游标,这里独立复制一份只是为了让类名/调用点见名知义,
    与backtest_engine.py其余"每种感官一个游标类"的既有风格一致)。
    advance_to(ts) 返回"时间戳 <= ts 的现货K线历史"(按timestamp升序的
    DataFrame切片,可能为空)。同款时间边界铁律:advance_to(ts) 的返回值里
    绝不会出现 timestamp > ts 的行。"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("timestamp").reset_index(drop=True)
        self._ts = self.df["timestamp"].to_numpy()
        self._idx = -1  # -1 = 尚未有任何 timestamp <= ts 的现货K线

    def advance_to(self, ts: int) -> pd.DataFrame:
        n = len(self._ts)
        while self._idx + 1 < n and self._ts[self._idx + 1] <= ts:
            self._idx += 1
        if self._idx < 0:
            return self.df.iloc[0:0]
        return self.df.iloc[: self._idx + 1]


class _OICursor:
    """M9持仓量(OI)感官注入:同 _SpotCursor/_FundingCursor 逐行照抄同一个
    "只前进不回退的单调游标"实现。advance_to(ts) 返回"时间戳 <= ts 的OI
    历史"(按timestamp升序的DataFrame切片,可能为空)。同款时间边界铁律:
    advance_to(ts) 的返回值里绝不会出现 timestamp > ts 的行。"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("timestamp").reset_index(drop=True)
        self._ts = self.df["timestamp"].to_numpy()
        self._idx = -1  # -1 = 尚未有任何 timestamp <= ts 的OI记录

    def advance_to(self, ts: int) -> pd.DataFrame:
        n = len(self._ts)
        while self._idx + 1 < n and self._ts[self._idx + 1] <= ts:
            self._idx += 1
        if self._idx < 0:
            return self.df.iloc[0:0]
        return self.df.iloc[: self._idx + 1]


def _max_drawdown_pct(nav_series: list[tuple[int, float]]) -> float:
    """标准定义:回撤 = (运行中峰值 - 当前值) / 运行中峰值 * 100,取全程最大值。"""
    peak: Optional[float] = None
    max_dd = 0.0
    for _, nav in nav_series:
        if peak is None or nav > peak:
            peak = nav
        if peak and peak > 0:
            dd = (peak - nav) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _return_pct(nav_series: list[tuple[int, float]]) -> float:
    if len(nav_series) < 2:
        return 0.0
    start_nav = nav_series[0][1]
    end_nav = nav_series[-1][1]
    if start_nav == 0:
        return 0.0
    return (end_nav / start_nav - 1.0) * 100.0


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """驱动 Simulator 逐bar回放历史数据的快速回测引擎。"""

    def __init__(self, config: dict, data_pipeline: Any, scratch_root: Path) -> None:
        self.config = config
        self.data_pipeline = data_pipeline
        self.scratch_root = Path(scratch_root)
        self.timeframe: str = (config.get("data", {}) or {}).get("timeframe", "4h")
        self.settle_hours_utc: set[int] = set(
            (config.get("funding", {}) or {}).get("settle_hours_utc", [0, 8, 16])
        )

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_fn: StrategyFn,
        symbols: list[str],
        windows: list[BacktestWindow],
        experiment_id: str,
    ) -> dict[str, BacktestResult]:
        results: dict[str, BacktestResult] = {}
        for window in windows:
            results[window.label] = self._run_window(strategy_fn, symbols, window, experiment_id)
        return results

    def _run_window(
        self,
        strategy_fn: StrategyFn,
        symbols: list[str],
        window: BacktestWindow,
        experiment_id: str,
    ) -> BacktestResult:
        window_root = self.scratch_root / experiment_id / window.label
        window_root.mkdir(parents=True, exist_ok=True)

        clock = FakeClock(start_ms=window.start_ts)

        # 每个窗口独立的熔断器实例——窗口之间的NAV历史/冻结状态互不污染,
        # 否则前一个窗口触发的 FROZEN_FULL 会粘性地拖累下一个窗口的评估。
        circuit_breaker = CircuitBreaker(config=self.config, log_root=window_root)

        sim = Simulator(
            config=self.config,
            circuit_breaker=circuit_breaker,
            cold_start_gate=lambda: False,  # 回测的历史数据本来就完整,COLD_START状态机的适用前提不成立
            universe_symbols=symbols,
            db_path=window_root / "portfolio.db",
            branch=experiment_id,
            log_root=window_root,
            resume=False,
        )

        bars_by_symbol = self._load_window_bars(symbols, window)
        master_symbol = symbols[0]
        master_df = bars_by_symbol[master_symbol]
        master_ts: list[int] = [int(t) for t in master_df["timestamp"].tolist()]

        cursors = {sym: _SymbolCursor(df) for sym, df in bars_by_symbol.items()}

        benchmark_df = bars_by_symbol.get(_BENCHMARK_SYMBOL)
        if benchmark_df is None:
            benchmark_df = self._fetch_symbol_bars(_BENCHMARK_SYMBOL, window)
        benchmark_cursor = _SymbolCursor(benchmark_df)

        funding_frames = self._load_funding_frames(symbols, window)
        funding_lookup, funding_miss = self._build_funding_lookup(funding_frames)
        # ctx.recent_funding 的逐bar切片游标,复用 funding_frames 里已经加载好
        # 的DataFrame(见 _FundingCursor 类文档字符串的时间边界说明)。
        funding_cursors = {sym: _FundingCursor(df) for sym, df in funding_frames.items()}

        # M9:ctx.recent_spot / ctx.recent_oi 的逐bar切片游标,窗口开始时一次性
        # 加载好各symbol的现货K线/OI历史(与funding同一手法:不为了喂ctx而在
        # 循环内重复拉取)。
        spot_frames = self._load_spot_frames(symbols, window)
        spot_cursors = {sym: _SpotCursor(df) for sym, df in spot_frames.items()}
        oi_frames = self._load_oi_frames(symbols, window)
        oi_cursors = {sym: _OICursor(df) for sym, df in oi_frames.items()}

        nav_series: list[tuple[int, float]] = []
        benchmark_nav_series: list[tuple[int, float]] = []
        trade_count = 0
        rejection_count = 0
        # 内存判重:(ts, symbol, action) 三元组。不落盘不重读文件——生产环境
        # Simulator._decision_is_logged() 是每次 execute() 都重新读一遍
        # decisions.jsonl 全文的 O(n) I/O,回测几千个bar级联下来会退化成
        # O(n^2);既然 simulator.py 本身不在本次授权可修改范围内,backtest_engine
        # 至少不应该在自己这一层再叠加一次同样代价的"是否已处理过"文件查重,
        # 用一个纯内存 set 就够了。
        seen_decisions: set[tuple[int, str, str]] = set()

        if not master_ts:
            # 窗口内完全没有可用K线数据——退化为"什么都没发生"的空结果。
            initial_nav = sim.get_portfolio()["nav"]
            self._close_sim(sim)
            return BacktestResult(
                window=window,
                nav_series=[(window.start_ts, initial_nav)],
                benchmark_nav_series=[(window.start_ts, float(self.config.get("capital_usdt", 0)))],
                return_pct=0.0,
                max_drawdown_pct=0.0,
                edge_vs_benchmark_pct=0.0,
                branch_dead=False,
                trade_count=0,
                rejection_count=0,
            )

        # ---- benchmark 起点:窗口第一根 BTC K线开盘价,只 enter 一次 ----
        benchmark_agent = BTCHoldAgent.from_config(self.config, symbol=_BENCHMARK_SYMBOL, branch=experiment_id)
        entry_price = float(benchmark_df.iloc[0]["open"])
        benchmark_agent.enter(entry_price=entry_price)

        clock.set_ms(master_ts[0])
        nav_series.append((master_ts[0], sim.get_portfolio()["nav"]))
        benchmark_nav_series.append((master_ts[0], benchmark_agent.nav(entry_price)))

        for i in range(len(master_ts) - 1):
            cur_ts = master_ts[i]
            nxt_ts = master_ts[i + 1]
            clock.set_ms(cur_ts)

            # ---- 构造 ctx:snapshot/recent_bars/recent_funding/recent_spot/
            # recent_oi 截至当前bar为止 ----
            snapshot: dict[str, dict] = {}
            recent_bars: dict[str, pd.DataFrame] = {}
            recent_funding: dict[str, pd.DataFrame] = {}
            recent_spot: dict[str, pd.DataFrame] = {}
            recent_oi: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                row = cursors[sym].advance_to(cur_ts)
                if row is None:
                    continue  # 该symbol此刻还没有任何历史数据(比如尚未上市)
                snapshot[sym] = {"last": float(row["close"])}
                recent_bars[sym] = cursors[sym].recent_window()
                fcursor = funding_cursors.get(sym)
                # 时间边界铁律:只切"<= cur_ts"这部分资金费率给策略,绝不把
                # 完整的窗口内历史整份塞进ctx——否则策略在bar N就能看到
                # 未来才会发生的真实费率,是未来函数。
                recent_funding[sym] = (
                    fcursor.advance_to(cur_ts)
                    if fcursor is not None
                    else pd.DataFrame(columns=["timestamp", "funding_rate"])
                )
                # M9:recent_spot/recent_oi 同款时间边界铁律,只切"<= cur_ts"。
                scursor = spot_cursors.get(sym)
                recent_spot[sym] = (
                    scursor.advance_to(cur_ts) if scursor is not None else pd.DataFrame(columns=_OHLCV_COLUMNS)
                )
                oicursor = oi_cursors.get(sym)
                recent_oi[sym] = (
                    oicursor.advance_to(cur_ts) if oicursor is not None else pd.DataFrame(columns=_OI_COLUMNS)
                )

            ctx = StrategyContext(
                ts=cur_ts + 1,  # 决策必须严格早于将要撮合的下一根bar的open_time
                positions=dict(sim.positions),
                snapshot=snapshot,
                recent_bars=recent_bars,
                recent_funding=recent_funding,
                recent_spot=recent_spot,
                recent_oi=recent_oi,
            )
            decisions = strategy_fn(ctx) or []

            for decision in decisions:
                key = (decision.ts, decision.symbol, decision.action)
                if key in seen_decisions:
                    continue  # 同一决策已经处理过,内存判重,不重复落盘/执行
                seen_decisions.add(key)

                sim.log_decision(decision)

                nxt_row = cursors.get(decision.symbol, None)
                nxt_row = nxt_row.advance_to(nxt_ts) if nxt_row is not None else None
                if nxt_row is None or int(nxt_row["timestamp"]) != nxt_ts:
                    # 决策指向的symbol在"下一根bar"这个时间点没有真实成交数据
                    # 可用(数据缺口/尚未上市),不可能撮合——记一次拒绝,而不是
                    # 伪造一根不存在的K线去成交。
                    rejection_count += 1
                    continue

                next_bar = {
                    "open_time": int(nxt_row["timestamp"]),
                    "open": float(nxt_row["open"]),
                    "high": float(nxt_row["high"]),
                    "low": float(nxt_row["low"]),
                    "close": float(nxt_row["close"]),
                    "volume_24h_usdt": _VOLUME_SENTINEL_USDT,
                }
                result = sim.execute(decision, next_bar)
                if isinstance(result, Trade):
                    if result.action != "hold":
                        # hold 决策也会产生一条 notional=0/fee=0 的 Trade 记录
                        # (simulator._compute_fill 的既有行为),不算一笔真实成交。
                        trade_count += 1
                elif isinstance(result, Rejection):
                    rejection_count += 1

            clock.set_ms(nxt_ts)

            # ---- 资金费率结算(落在 UTC 0/8/16点的bar) ----
            settle_hour = (nxt_ts // 3_600_000) % 24
            if settle_hour in self.settle_hours_utc:
                sim.settle_funding(nxt_ts, funding_lookup)

            # ---- 逐bar插针检查(用刚形成的这根bar的high/low/close) ----
            mark_prices: dict[str, dict] = {}
            for sym in list(sim.positions.keys()):
                row = cursors[sym].advance_to(nxt_ts) if sym in cursors else None
                if row is not None and int(row["timestamp"]) == nxt_ts:
                    mark_prices[sym] = {"high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])}
            if mark_prices:
                sim.check_liquidation(mark_prices, ts_utc=nxt_ts)

            # ---- nav_series 每bar记一个点 ----
            price_hint = {
                sym: float(cursors[sym].advance_to(nxt_ts)["close"])
                for sym in sim.positions
                if sym in cursors and cursors[sym].advance_to(nxt_ts) is not None
            }
            nav = sim.mark_to_market(price_hint)
            nav_series.append((nxt_ts, nav))

            btc_row = benchmark_cursor.advance_to(nxt_ts)
            if btc_row is not None:
                benchmark_nav_series.append((nxt_ts, benchmark_agent.nav(float(btc_row["close"]))))

            if sim.branch_dead:
                break

        if funding_miss["count"] > 0:
            logger.warning(
                "backtest_engine window=%s: %d funding rate lookups had no cached data, defaulted to 0.0",
                window.label, funding_miss["count"],
            )

        return_pct = _return_pct(nav_series)
        benchmark_return_pct = _return_pct(benchmark_nav_series)
        max_dd_pct = _max_drawdown_pct(nav_series)

        result = BacktestResult(
            window=window,
            nav_series=nav_series,
            benchmark_nav_series=benchmark_nav_series,
            return_pct=return_pct,
            max_drawdown_pct=max_dd_pct,
            edge_vs_benchmark_pct=return_pct - benchmark_return_pct,
            branch_dead=sim.branch_dead,
            trade_count=trade_count,
            rejection_count=rejection_count,
        )
        self._close_sim(sim)
        return result

    @staticmethod
    def _close_sim(sim: Simulator) -> None:
        """窗口结束后显式关掉Simulator的sqlite连接——Windows下打开的连接会
        锁住portfolio.db文件,导致上层无法清理scratch实验目录(真实踩到的
        PermissionError,不是防御性设计)。"""
        conn = getattr(sim, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 -- 关连接失败不应让回测结果作废
                pass

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _bar_limit_for_window(self, window: BacktestWindow) -> int:
        interval_ms = _timeframe_to_ms(self.timeframe)
        span_ms = window.end_ts - window.start_ts
        return int(span_ms // interval_ms) + 10  # 缓冲几根,防止边界取整少一根

    def _fetch_symbol_bars(self, symbol: str, window: BacktestWindow) -> pd.DataFrame:
        limit = self._bar_limit_for_window(window)
        df = self.data_pipeline.fetch_ohlcv(
            symbol, timeframe=self.timeframe, since=window.start_ts, limit=limit
        )
        return df[df["timestamp"] < window.end_ts].reset_index(drop=True)

    def _load_window_bars(self, symbols: list[str], window: BacktestWindow) -> dict[str, pd.DataFrame]:
        return {sym: self._fetch_symbol_bars(sym, window) for sym in symbols}

    def _load_funding_frames(
        self, symbols: list[str], window: BacktestWindow
    ) -> dict[str, pd.DataFrame]:
        """按symbol一次性加载窗口内的资金费率历史(升序DataFrame),供
        资金费率结算的lookup表和ctx.recent_funding的逐bar切片共用同一份
        已加载数据——不为了喂ctx而对同一个symbol多拉一次。
        资金费率结算周期(8h)比默认4h K线更稀疏,复用同一个上限估算函数
        得到的条数上限天然绰绰有余,不需要单独再算一遍。"""
        limit = self._bar_limit_for_window(window)
        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                fdf = self.data_pipeline.fetch_funding_rate_history(sym, since=window.start_ts, limit=limit)
            except Exception as exc:  # noqa: BLE001 - 回测里某个symbol没有资金费率缓存不应打断整个回放
                logger.warning("backtest_engine: fetch_funding_rate_history(%s) failed: %r", sym, exc)
                fdf = pd.DataFrame(columns=["timestamp", "funding_rate"])
            fdf = fdf[fdf["timestamp"] < window.end_ts]
            frames[sym] = fdf.sort_values("timestamp").reset_index(drop=True)
        return frames

    def _load_spot_frames(
        self, symbols: list[str], window: BacktestWindow
    ) -> dict[str, pd.DataFrame]:
        """M9:按symbol一次性加载窗口内的现货K线(升序DataFrame),供
        ctx.recent_spot的逐bar切片使用——与 _load_funding_frames 同一手法,
        窗口开始时一次性拉好,不在逐bar循环里重复拉取。某symbol没有现货
        对应(比如只在这个交易所上了永续、没上现货)/拉取失败时,
        DataPipeline.fetch_spot_ohlcv 本身已经承诺返回空DataFrame不抛异常
        (见该方法docstring),这里仍然包一层try/except——因为传入的
        data_pipeline在某些调用路径下是测试用的鸭子类型替身,不能假设它
        真的遵守这条"不抛异常"的契约。"""
        limit = self._bar_limit_for_window(window)
        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                sdf = self.data_pipeline.fetch_spot_ohlcv(
                    sym, timeframe=self.timeframe, since=window.start_ts, limit=limit
                )
            except Exception as exc:  # noqa: BLE001 - 回测里某个symbol没有现货缓存不应打断整个回放
                logger.warning("backtest_engine: fetch_spot_ohlcv(%s) failed: %r", sym, exc)
                sdf = pd.DataFrame(columns=_OHLCV_COLUMNS)
            sdf = sdf[sdf["timestamp"] < window.end_ts]
            frames[sym] = sdf.sort_values("timestamp").reset_index(drop=True)
        return frames

    def _load_oi_frames(
        self, symbols: list[str], window: BacktestWindow
    ) -> dict[str, pd.DataFrame]:
        """M9:按symbol一次性加载窗口内的持仓量(OI)历史(升序DataFrame),
        供ctx.recent_oi的逐bar切片使用,手法与 _load_spot_frames/
        _load_funding_frames 完全一致。"""
        limit = self._bar_limit_for_window(window)
        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                odf = self.data_pipeline.fetch_open_interest_history(sym, since=window.start_ts, limit=limit)
            except Exception as exc:  # noqa: BLE001 - 回测里某个symbol没有OI缓存不应打断整个回放
                logger.warning("backtest_engine: fetch_open_interest_history(%s) failed: %r", sym, exc)
                odf = pd.DataFrame(columns=_OI_COLUMNS)
            odf = odf[odf["timestamp"] < window.end_ts]
            frames[sym] = odf.sort_values("timestamp").reset_index(drop=True)
        return frames

    @staticmethod
    def _build_funding_lookup(
        funding_frames: dict[str, pd.DataFrame],
    ) -> tuple[Callable[[str, int], float], dict[str, int]]:
        funding_by_symbol: dict[str, dict[int, float]] = {
            sym: dict(zip(fdf["timestamp"].astype("int64"), fdf["funding_rate"].astype(float)))
            for sym, fdf in funding_frames.items()
        }

        miss_counter = {"count": 0}

        def funding_rate_lookup(symbol: str, ts: int) -> float:
            table = funding_by_symbol.get(symbol)
            if not table:
                miss_counter["count"] += 1
                return 0.0
            if ts in table:
                return table[ts]
            # 缓存里有这个symbol的费率数据,但没有恰好对齐的时间戳(数据边界
            # 效应)——取最近的一条,与 scripts/m1_live_shakedown.py 的既有
            # 处理方式一致。
            nearest = min(table.keys(), key=lambda t: abs(t - ts))
            return table[nearest]

        return funding_rate_lookup, miss_counter

    # ------------------------------------------------------------------
    # score()
    # ------------------------------------------------------------------

    def score(self, results: dict[str, BacktestResult]) -> float:
        """防过拟合评分——铁律,不可更改这条逻辑本身:

        1. holdout 窗口的结果绝不参与评分。这是"内环迭代期间 holdout 不可见"
           这条防过拟合硬要求的直接体现:如果评分函数偷看了 holdout,策略/
           超参搜索循环就会不知不觉地把 holdout 当成又一个可以拟合的验证集,
           holdout 也就丧失了作为"最终、独立、只看一次"的检验意义。
        2. train 窗口的结果同样不参与评分——train 本来就是用来拟合参数的
           窗口,拿它自己的表现给自己打分是循环论证。
        3. 取参与评分的验证窗口里"最差"的 edge_vs_benchmark_pct(min),而不是
           平均值。平均会掩盖"策略只在某一种行情结构下有效、换一个窗口就失灵"
           这种脆弱性——一个在 val_1 里大赚、在 val_2 里巨亏的策略,平均分可能
           看起来还不错,但这恰恰是最危险的过拟合信号;取最差窗口能把这种
           脆弱性直接体现在最终分数里,逼着候选策略必须在"所有"验证窗口下都
           站得住脚,而不是只挑一个行情配合的窗口。
        4. 任意一个验证窗口触发爆仓(branch_dead)时直接返回 -inf——爆仓是硬性
           一票否决,不参与"最差窗口"这种连续量纲的比较(一个爆仓分支不该
           因为其它窗口edge很高而被平均/min掉这个事实)。
        5. 回撤惩罚项(0.5 倍验证窗口最大回撤)沿用同一个"取最差"精神:用
           参与评分窗口里出现过的最大回撤,而不是平均回撤。
        """
        val_results = [r for label, r in results.items() if not r.window.is_holdout and label != "train"]
        if not val_results:
            raise ValueError(
                "score(): no validation window results to score "
                "(need at least one window that is neither 'train' nor is_holdout=True)"
            )
        if any(r.branch_dead for r in val_results):
            return float("-inf")
        edges = [r.edge_vs_benchmark_pct for r in val_results]
        return min(edges) - 0.5 * max(r.max_drawdown_pct for r in val_results)
