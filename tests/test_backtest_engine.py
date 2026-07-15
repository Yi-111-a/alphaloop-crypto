"""
test_backtest_engine.py —— M6 LOCKED/backtest_engine.py 离线单测。

全部使用合成K线数据 + 鸭子类型的 stub DataPipeline(不发起任何真实网络
请求,也不依赖 data_cache/ 里的真实文件是否存在),覆盖规格书验收标准里
可以离线验证的部分:hold策略跑通、高杠杆死扛触发爆仓传播、score()取最差
窗口而非平均、holdout绝不参与评分、不污染生产LOG、确定性、性能冒烟。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from LOCKED.backtest_engine import BacktestEngine, BacktestResult, BacktestWindow, StrategyContext
from LOCKED.schemas import Decision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

STEP_MS = 4 * 60 * 60 * 1000  # 4h
# 对齐到4h网格的一个任意起点,保证 UTC 0/8/16点结算判断能命中一部分bar,
# 与生产环境的真实网格保持一致(不是随手选一个不对齐的数字)。
BASE_TS = 1_700_000_000_000 - (1_700_000_000_000 % STEP_MS)

_LONG_THESIS = "回测单测固定占位thesis文本,长度刻意超过20字符满足Simulator第3步校验"
_LONG_FALSIFIER = "回测单测固定占位falsifier文本,长度刻意超过20字符满足Simulator第3步校验"

BTC = "BTC/USDT:USDT"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_flat_ohlcv(n: int, start_ts: int = BASE_TS, price: float = 50_000.0) -> pd.DataFrame:
    """n根完全平盘的4h K线(open=high=low=close=price,volume恒定)。"""
    rows = []
    for i in range(n):
        ts = start_ts + i * STEP_MS
        rows.append([ts, price, price, price, price, 1_000.0])
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def make_wavy_ohlcv(n: int, start_ts: int = BASE_TS, price: float = 50_000.0) -> pd.DataFrame:
    """n根小幅波动(但不崩盘)的4h K线,用于性能/确定性冒烟测试——需要一点
    波动而不是完全平盘,免得测试意外掩盖了取整/浮点比较方面的bug。"""
    rows = []
    for i in range(n):
        ts = start_ts + i * STEP_MS
        price = price * (1 + ((i % 7) - 3) * 0.001)
        rows.append([ts, price, price * 1.001, price * 0.999, price, 1_000.0])
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def make_crash_ohlcv(n: int, start_ts: int = BASE_TS, price: float = 50_000.0, crash_at: int = 5) -> pd.DataFrame:
    """前 crash_at 根平盘,之后连续暴跌(每根close比上一根close低15%,且当根
    K线low低至close的一半,足以插针击穿任何常规杠杆多头仓位)。"""
    rows = []
    cur = price
    for i in range(n):
        ts = start_ts + i * STEP_MS
        if i < crash_at:
            o, h, l, c = cur, cur * 1.001, cur * 0.999, cur
        else:
            cur = cur * 0.85
            o, h, l, c = cur / 0.85, cur * 1.01, cur * 0.5, cur
        rows.append([ts, o, h, l, c, 1_000.0])
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


class StubDataPipeline:
    """鸭子类型的 DataPipeline 替身:只实现 backtest_engine 实际调用到的两个
    方法,签名与 LOCKED.data_pipeline.DataPipeline 完全一致,但直接从内存里
    的合成 DataFrame 切片返回,不碰磁盘/网络——保证测试完全离线确定性。"""

    def __init__(self, ohlcv_by_symbol: dict[str, pd.DataFrame], funding_by_symbol: dict[str, pd.DataFrame] | None = None):
        self.ohlcv_by_symbol = ohlcv_by_symbol
        self.funding_by_symbol = funding_by_symbol or {}
        self.ohlcv_calls = 0
        self.funding_calls = 0

    def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000):
        self.ohlcv_calls += 1
        df = self.ohlcv_by_symbol.get(symbol)
        if df is None:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if since is not None:
            df = df[df["timestamp"] >= since]
        return df.head(limit).reset_index(drop=True)

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        self.funding_calls += 1
        df = self.funding_by_symbol.get(symbol)
        if df is None:
            return pd.DataFrame(columns=["timestamp", "funding_rate"])
        if since is not None:
            df = df[df["timestamp"] >= since]
        return df.head(limit).reset_index(drop=True)


def hold_forever_strategy(ctx: StrategyContext) -> list[Decision]:
    """"永远hold"策略:每个bar都只产出一条 action="hold" 的决策,从不开仓。"""
    return [
        Decision(
            ts=ctx.ts,
            symbol=BTC,
            action="hold",
            target_notional_pct=0.0,
            leverage=1,
            thesis=_LONG_THESIS,
            falsifier=_LONG_FALSIFIER,
            horizon="4h",
        )
    ]


def make_allin_leverage_strategy():
    """"10倍杠杆满仓做多然后死扛"策略:第一个bar开满仓多单(单币名义上限=
    100% of NAV,是 config.constraints.max_position_notional_pct 允许的
    最大值,配合 leverage=10 就是"用足杠杆上限的满仓单向下注"),此后永远
    hold(死扛,不止损不加仓)。用一个闭包变量确保只开仓一次。"""
    state = {"opened": False}

    def _strategy(ctx: StrategyContext) -> list[Decision]:
        if state["opened"]:
            return [
                Decision(
                    ts=ctx.ts, symbol=BTC, action="hold", target_notional_pct=0.0, leverage=1,
                    thesis=_LONG_THESIS, falsifier=_LONG_FALSIFIER, horizon="4h",
                )
            ]
        state["opened"] = True
        return [
            Decision(
                ts=ctx.ts,
                symbol=BTC,
                action="open_long",
                target_notional_pct=100.0,  # 单币名义仓位上限(config.constraints.max_position_notional_pct)
                leverage=10,
                thesis=_LONG_THESIS,
                falsifier=_LONG_FALSIFIER,
                horizon="4h",
            )
        ]

    return _strategy


def make_engine(tmp_path: Path, ohlcv_by_symbol: dict[str, pd.DataFrame], funding_by_symbol: dict[str, pd.DataFrame] | None = None) -> tuple[BacktestEngine, StubDataPipeline]:
    dp = StubDataPipeline(ohlcv_by_symbol, funding_by_symbol)
    engine = BacktestEngine(config=load_config(), data_pipeline=dp, scratch_root=tmp_path / "scratch")
    return engine, dp


# ---------------------------------------------------------------------------
# 1. "永远hold"策略:跑通、trade_count==0、nav基本持平
# ---------------------------------------------------------------------------


def test_hold_forever_strategy_runs_with_zero_trades_and_flat_nav(tmp_path):
    n = 40
    ohlcv = make_flat_ohlcv(n)
    engine, dp = make_engine(tmp_path, {BTC: ohlcv})
    window = BacktestWindow(label="val_1", start_ts=BASE_TS, end_ts=BASE_TS + n * STEP_MS)

    results = engine.run(hold_forever_strategy, [BTC], [window], experiment_id="exp_hold")
    result = results["val_1"]

    assert isinstance(result, BacktestResult)
    assert result.trade_count == 0
    assert result.rejection_count == 0
    assert result.branch_dead is False
    # hold策略从不开仓 -> settle_funding 对空仓位是no-op,nav应该恰好持平
    # (不是"基本持平"要打折扣,这里从头到尾就没有任何持仓可以产生费用)。
    capital = float(load_config()["capital_usdt"])
    for _, nav in result.nav_series:
        assert nav == pytest.approx(capital)
    assert result.return_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. 10倍杠杆满仓死扛 + 暴跌 -> branch_dead 传播,score() 返回 -inf
# ---------------------------------------------------------------------------


def test_leveraged_allin_strategy_dies_on_crash_and_score_is_neg_inf(tmp_path):
    n = 20
    ohlcv = make_crash_ohlcv(n, crash_at=5)
    engine, dp = make_engine(tmp_path, {BTC: ohlcv})
    window = BacktestWindow(label="val_1", start_ts=BASE_TS, end_ts=BASE_TS + n * STEP_MS)

    strategy = make_allin_leverage_strategy()
    results = engine.run(strategy, [BTC], [window], experiment_id="exp_crash")
    result = results["val_1"]

    assert result.branch_dead is True
    # 爆仓应该在跑完全部bar之前就中止(暴跌只从第5根开始,总共20根)。
    assert len(result.nav_series) < n

    score = engine.score(results)
    assert score == float("-inf")


# ---------------------------------------------------------------------------
# 3. score() 取两个val窗口里最差的edge,而不是平均
# ---------------------------------------------------------------------------


def _make_window(label: str, is_holdout: bool = False) -> BacktestWindow:
    return BacktestWindow(label=label, start_ts=BASE_TS, end_ts=BASE_TS + 10 * STEP_MS, is_holdout=is_holdout)


def _make_result(label: str, edge_pct: float, max_dd_pct: float = 0.0, branch_dead: bool = False, is_holdout: bool = False) -> BacktestResult:
    window = _make_window(label, is_holdout=is_holdout)
    return BacktestResult(
        window=window,
        nav_series=[(BASE_TS, 100.0), (BASE_TS + STEP_MS, 100.0 + edge_pct)],
        benchmark_nav_series=[(BASE_TS, 100.0), (BASE_TS + STEP_MS, 100.0)],
        return_pct=edge_pct,
        max_drawdown_pct=max_dd_pct,
        edge_vs_benchmark_pct=edge_pct,
        branch_dead=branch_dead,
        trade_count=1,
        rejection_count=0,
    )


def test_score_takes_worst_val_window_not_average(tmp_path):
    engine, _ = make_engine(tmp_path, {BTC: make_flat_ohlcv(5)})

    results = {
        "val_1": _make_result("val_1", edge_pct=5.0),
        "val_2": _make_result("val_2", edge_pct=-2.0),
    }
    score = engine.score(results)

    average_edge = (5.0 - 2.0) / 2.0
    # min(edges) - 0.5*max(dd) = -2.0 - 0 = -2.0,必须严格等于"最差窗口",
    # 而不是碰巧落在平均值附近。
    assert score == pytest.approx(-2.0)
    assert score != pytest.approx(average_edge)


def test_score_penalizes_by_worst_drawdown(tmp_path):
    engine, _ = make_engine(tmp_path, {BTC: make_flat_ohlcv(5)})
    results = {
        "val_1": _make_result("val_1", edge_pct=5.0, max_dd_pct=4.0),
        "val_2": _make_result("val_2", edge_pct=3.0, max_dd_pct=10.0),
    }
    score = engine.score(results)
    # min(edges)=3.0, 惩罚项用两个窗口里最大的回撤(10.0),不是各自窗口自己的回撤。
    assert score == pytest.approx(3.0 - 0.5 * 10.0)


# ---------------------------------------------------------------------------
# 4. holdout 结果绝不参与评分
# ---------------------------------------------------------------------------


def test_score_excludes_holdout_and_train_results(tmp_path):
    engine, _ = make_engine(tmp_path, {BTC: make_flat_ohlcv(5)})

    results_without_holdout = {
        "train": _make_result("train", edge_pct=999.0),  # 极端值,若被误纳入会立刻暴露
        "val_1": _make_result("val_1", edge_pct=5.0),
        "val_2": _make_result("val_2", edge_pct=-2.0),
    }
    score_without_holdout = engine.score(results_without_holdout)

    results_with_holdout = dict(results_without_holdout)
    # holdout 用一个"如果被纳入评分会让结果天差地别"的极端负值,以及
    # branch_dead=True(如果holdout意外参与,any(branch_dead)会让分数变成
    # -inf,从而被本测试直接抓到)。
    results_with_holdout["holdout"] = _make_result(
        "holdout", edge_pct=-9999.0, branch_dead=True, is_holdout=True
    )
    score_with_holdout = engine.score(results_with_holdout)

    assert score_with_holdout == pytest.approx(score_without_holdout)
    assert score_with_holdout != float("-inf")


def test_score_raises_without_any_validation_window(tmp_path):
    engine, _ = make_engine(tmp_path, {BTC: make_flat_ohlcv(5)})
    results = {
        "train": _make_result("train", edge_pct=5.0),
        "holdout": _make_result("holdout", edge_pct=5.0, is_holdout=True),
    }
    with pytest.raises(ValueError):
        engine.score(results)


# ---------------------------------------------------------------------------
# 5. 回测绝不污染生产 LOG/decisions.jsonl
# ---------------------------------------------------------------------------


def test_backtest_run_does_not_modify_production_log(tmp_path):
    prod_decisions_path = PROJECT_ROOT / "LOG" / "decisions.jsonl"
    before_exists = prod_decisions_path.exists()
    before_lines = prod_decisions_path.read_text(encoding="utf-8").count("\n") if before_exists else None

    n = 20
    engine, _ = make_engine(tmp_path, {BTC: make_flat_ohlcv(n)})
    window = BacktestWindow(label="val_1", start_ts=BASE_TS, end_ts=BASE_TS + n * STEP_MS)
    strategy = make_allin_leverage_strategy()  # 用会真正产生交易/爆仓的策略,而不是空转的hold
    engine.run(strategy, [BTC], [window], experiment_id="exp_no_pollution")

    after_exists = prod_decisions_path.exists()
    after_lines = prod_decisions_path.read_text(encoding="utf-8").count("\n") if after_exists else None

    assert after_exists == before_exists
    assert after_lines == before_lines


# ---------------------------------------------------------------------------
# 6. 确定性:同一策略+同一窗口跑两次,结果完全相同
# ---------------------------------------------------------------------------


def test_same_strategy_same_window_reproduces_identical_nav_series(tmp_path):
    n = 60
    ohlcv = make_wavy_ohlcv(n)
    strategy = make_allin_leverage_strategy()
    window = BacktestWindow(label="val_1", start_ts=BASE_TS, end_ts=BASE_TS + n * STEP_MS)

    engine1, _ = make_engine(tmp_path, {BTC: ohlcv})
    results1 = engine1.run(strategy, [BTC], [window], experiment_id="exp_det_1")

    # 全新策略实例(闭包状态重置)+ 全新engine实例,指向同一份合成数据。
    strategy2 = make_allin_leverage_strategy()
    engine2, _ = make_engine(tmp_path, {BTC: ohlcv})
    results2 = engine2.run(strategy2, [BTC], [window], experiment_id="exp_det_2")

    assert results1["val_1"].nav_series == results2["val_1"].nav_series
    assert results1["val_1"].benchmark_nav_series == results2["val_1"].benchmark_nav_series
    assert results1["val_1"].branch_dead == results2["val_1"].branch_dead
    assert results1["val_1"].trade_count == results2["val_1"].trade_count


# ---------------------------------------------------------------------------
# 7. 性能冒烟:500根bar的窗口回放耗时 < 30秒(宽松阈值,防O(n^2)回归)
# ---------------------------------------------------------------------------


def test_performance_smoke_500_bars_under_30_seconds(tmp_path):
    n = 500
    ohlcv = make_wavy_ohlcv(n)
    engine, _ = make_engine(tmp_path, {BTC: ohlcv})
    window = BacktestWindow(label="val_1", start_ts=BASE_TS, end_ts=BASE_TS + n * STEP_MS)

    start = time.time()
    results = engine.run(hold_forever_strategy, [BTC], [window], experiment_id="exp_perf")
    elapsed = time.time() - start

    assert elapsed < 30.0, f"500-bar replay took {elapsed:.2f}s, exceeds the 30s regression budget"
    assert len(results["val_1"].nav_series) == n


def test_ctx_recent_funding_never_leaks_future_data(tmp_path):
    """M7资金费率感官注入最重要的一条测试(未来函数检测):回放过程中任何
    一个bar喂给策略的ctx.recent_funding,其内部时间戳的最大值绝不能超过
    该bar的ctx.ts——否则策略在bar N就能看到bar N之后才真实发生的资金费率
    结算,是未来函数,直接让回测结果失去参考意义。

    断言方式:用一个"回声策略"(echo strategy),每次被调用时只记录
    (ctx.ts, ctx.recent_funding里的最大timestamp),不产生任何真实交易决策
    (返回[]),跑完整个窗口后对记录下来的每一对做未来函数检测,而不是只
    抽查某一个bar。"""
    n = 30
    ohlcv = make_wavy_ohlcv(n)

    # 资金费率恰好每根K线结算一次(简化的确定性构造,不代表真实8h周期,但
    # 足以精确验证游标切片的边界:第i根bar对应的funding记录时间戳就是
    # BASE_TS + i*STEP_MS,与cur_ts一一对齐)。
    funding_df = pd.DataFrame(
        {
            "timestamp": [BASE_TS + i * STEP_MS for i in range(n)],
            "funding_rate": [0.0001 * (i + 1) for i in range(n)],
        }
    )

    engine, _ = make_engine(tmp_path, {BTC: ohlcv}, {BTC: funding_df})
    window = BacktestWindow(label="val_1", start_ts=BASE_TS, end_ts=BASE_TS + n * STEP_MS)

    observed = []

    def echo_strategy(ctx):
        fdf = ctx.recent_funding.get(BTC)
        max_ts = int(fdf["timestamp"].max()) if fdf is not None and not fdf.empty else None
        observed.append((ctx.ts, max_ts))
        return []

    engine.run(echo_strategy, [BTC], [window], experiment_id="exp_funding_boundary")

    assert observed, "expected the echo strategy to be invoked at least once"
    assert any(max_ts is not None for _, max_ts in observed), (
        "expected at least some bars to see non-empty funding data (otherwise this test is vacuous)"
    )

    for ctx_ts, max_funding_ts in observed:
        if max_funding_ts is None:
            continue
        # 核心断言(未来函数检测):任意bar看到的funding最大timestamp必须
        # <= 该bar的ctx.ts。
        assert max_funding_ts <= ctx_ts, (
            "future function detected: ctx.recent_funding max timestamp %r > ctx.ts %r"
            % (max_funding_ts, ctx_ts)
        )
        # 更精确的边界断言:本测试构造的funding记录与每根bar的cur_ts一一
        # 对齐(ctx.ts = cur_ts + 1),游标切到"<=cur_ts"应该恰好等于
        # ctx.ts - 1,而不是提前把ctx.ts当成这一bar自己的费率也算进去、
        # 更不是把窗口内全部未来记录都给了。
        assert max_funding_ts == ctx_ts - 1, (
            "expected max funding timestamp to be exactly ctx.ts - 1 (cursor cutoff == cur_ts), "
            "got %r for ctx.ts=%r" % (max_funding_ts, ctx_ts)
        )


def test_ctx_recent_funding_is_empty_dataframe_when_symbol_has_no_funding_data(tmp_path):
    """窗口内某symbol完全没有资金费率数据时,ctx.recent_funding[symbol]必须
    是一个列齐全的空DataFrame,而不是该symbol缺key/None。"""
    n = 10
    ohlcv = make_flat_ohlcv(n)
    engine, _ = make_engine(tmp_path, {BTC: ohlcv})  # 不提供任何funding_by_symbol
    window = BacktestWindow(label="val_1", start_ts=BASE_TS, end_ts=BASE_TS + n * STEP_MS)

    probe = {}

    def echo_strategy(ctx):
        if "checked" not in probe:
            fdf = ctx.recent_funding.get(BTC)
            probe["checked"] = True
            probe["is_dataframe"] = isinstance(fdf, pd.DataFrame)
            probe["is_empty"] = fdf.empty if fdf is not None else None
            probe["columns"] = list(fdf.columns) if fdf is not None else None
        return []

    engine.run(echo_strategy, [BTC], [window], experiment_id="exp_funding_empty")

    assert probe.get("checked") is True
    assert probe["is_dataframe"] is True
    assert probe["is_empty"] is True
    assert probe["columns"] == ["timestamp", "funding_rate"]


def test_ctx_recent_funding_injected_alongside_recent_bars(tmp_path):
    """基本注入验收:有真实资金费率数据时,ctx.recent_funding确实被填充。"""
    n = 15
    ohlcv = make_wavy_ohlcv(n)
    funding_df = pd.DataFrame(
        {
            "timestamp": [BASE_TS + i * STEP_MS for i in range(n)],
            "funding_rate": [0.0002] * n,
        }
    )
    engine, _ = make_engine(tmp_path, {BTC: ohlcv}, {BTC: funding_df})
    window = BacktestWindow(label="val_1", start_ts=BASE_TS, end_ts=BASE_TS + n * STEP_MS)

    seen_non_empty = {"flag": False}

    def echo_strategy(ctx):
        fdf = ctx.recent_funding.get(BTC)
        if fdf is not None and not fdf.empty:
            seen_non_empty["flag"] = True
            assert set(fdf.columns) >= {"timestamp", "funding_rate"}
        return []

    engine.run(echo_strategy, [BTC], [window], experiment_id="exp_funding_injected")

    assert seen_non_empty["flag"] is True
