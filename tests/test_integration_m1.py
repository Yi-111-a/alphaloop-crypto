"""
M1 跨模块集成校验:不 mock 任何 LOCKED 模块之间的接口,
只在最外层(ccxt 交易所对象)打桩,验证 universe_filter -> simulator ->
circuit_breaker -> scorer 的真实数据能端到端串起来,并逐条对照 spec §6 M1 验收清单。
"""
from __future__ import annotations

import json
import time

from LOCKED.baseline_agents import RandomAgent
from LOCKED.circuit_breaker import CircuitBreaker
from LOCKED.log_writer import append_jsonl, read_jsonl
from LOCKED.scorer import Scorer
from LOCKED.simulator import Simulator
from LOCKED.universe_filter import UniverseFilter


class _FakeMarket(dict):
    pass


class _FakeExchange:
    """最小可用的假交易所,只实现 universe_filter/simulator 需要的公开端点。"""

    def __init__(self):
        # universe_filter.fetch_candidates() computes listing_days against real
        # wall-clock time.time(), so "created" timestamps here must be anchored
        # to real now, not a fixed fake epoch (unlike circuit_breaker, this
        # module has no time-injection contract).
        now_ms = int(time.time() * 1000)
        self.markets = {
            "BTC/USDT:USDT": _FakeMarket(
                symbol="BTC/USDT:USDT", type="swap", swap=True, linear=True, quote="USDT",
                base="BTC", created=now_ms - 400 * 86_400_000,
                info={"onboardDate": now_ms - 400 * 86_400_000},
            ),
            "DOGEUP/USDT:USDT": _FakeMarket(
                symbol="DOGEUP/USDT:USDT", type="swap", swap=True, linear=True, quote="USDT",
                base="DOGEUP", created=now_ms - 400 * 86_400_000,
                info={"onboardDate": now_ms - 400 * 86_400_000},
            ),
            "NEWCOIN/USDT:USDT": _FakeMarket(
                symbol="NEWCOIN/USDT:USDT", type="swap", swap=True, linear=True, quote="USDT",
                base="NEWCOIN", created=now_ms - 10 * 86_400_000,
                info={"onboardDate": now_ms - 10 * 86_400_000},
            ),
        }

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return {
            "BTC/USDT:USDT": {"quoteVolume": 500_000_000},
            "DOGEUP/USDT:USDT": {"quoteVolume": 500_000_000},
            "NEWCOIN/USDT:USDT": {"quoteVolume": 500_000_000},
        }


def test_universe_filter_feeds_simulator_directly(tmp_path, make_decision):
    config = {
        "universe_rule": {
            "min_24h_volume_usdt": 100e6,
            "min_listing_days": 90,
            "blacklist": [],
        },
        "leverage": {"max": 10, "default": 3},
        "fees": {"taker_pct": 0.0005, "slippage_bps": 15},
        "constraints": {
            "max_position_notional_pct": 100,
            "max_total_notional_pct": 300,
            "min_free_margin_pct": 15,
            "max_drawdown_pct": 20,
            "daily_loss_freeze_pct": 8,
        },
        "capital_usdt": 100_000,
    }

    uf = UniverseFilter(config, exchange=_FakeExchange())
    out_path = tmp_path / "universe_active.json"
    result = uf.refresh(output_path=out_path)

    assert "BTC/USDT:USDT" in result["symbols"]
    assert "DOGEUP/USDT:USDT" not in result["symbols"], "leveraged token must be excluded"
    assert "NEWCOIN/USDT:USDT" not in result["symbols"], "coin listed <90 days must be excluded"

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["symbols"] == result["symbols"]

    cb = CircuitBreaker(config, log_root=tmp_path / "LOG")
    sim = Simulator(
        config,
        circuit_breaker=cb,
        universe_symbols=result["symbols"],
        db_path=tmp_path / "state" / "portfolio_main.db",
        log_root=tmp_path / "LOG",
        branch="main",
    )

    # NOTE: BTCHoldAgent's literal spec'd 100%-notional / 1x-leverage decision
    # cannot fill through the fully-constrained Simulator: at leverage=1,
    # margin == notional, so a 100%-of-NAV position leaves 0% free margin,
    # violating constraints.min_free_margin_pct (15%) by construction (max
    # feasible 1x allocation is 100 - min_free_margin_pct = 85%). This is a
    # real design tension between the BTC_HOLD benchmark spec ("满仓买入") and
    # the uniform risk constraints §2.2 applies to every decision through
    # execute() - it is NOT an M1 module bug, and is left for M2's main.py to
    # resolve (e.g. size the benchmark at 100-min_free_margin_pct, or give
    # baseline agents a constraint-exempt execution path). Flagged in the
    # M1 report; using a leveraged decision here instead so this test can
    # focus on universe_filter -> simulator wiring.
    decision = make_decision(
        symbol="BTC/USDT:USDT", action="open_long", target_notional_pct=50.0, leverage=3, branch="main"
    )
    append_jsonl("decisions.jsonl", decision, root=tmp_path / "LOG")

    next_bar = {"open_time": 1_700_000_100_000, "open": 60_000.0, "volume_24h_usdt": 500_000_000}
    trade = sim.execute(decision, next_bar)

    from LOCKED.schemas import Trade
    assert isinstance(trade, Trade), f"expected a fill, got a Rejection: {trade}"
    assert trade.symbol == "BTC/USDT:USDT"

    portfolio = sim.get_portfolio(snapshot={"BTC/USDT:USDT": 60_000.0})
    assert portfolio["nav"] > 0
    assert len(portfolio["positions"]) == 1


def test_random_agent_rejected_outside_universe(tmp_path):
    config = {
        "leverage": {"max": 10, "default": 3},
        "fees": {"taker_pct": 0.0005, "slippage_bps": 15},
        "constraints": {
            "max_position_notional_pct": 100,
            "max_total_notional_pct": 300,
            "min_free_margin_pct": 15,
            "max_drawdown_pct": 20,
            "daily_loss_freeze_pct": 8,
        },
        "capital_usdt": 100_000,
    }
    cb = CircuitBreaker(config, log_root=tmp_path / "LOG")
    sim = Simulator(
        config,
        circuit_breaker=cb,
        universe_symbols=["BTC/USDT:USDT"],  # NEWCOIN is not here
        db_path=tmp_path / "state" / "portfolio_random.db",
        log_root=tmp_path / "LOG",
        branch="random",
    )
    agent = RandomAgent(universe_symbols=["BTC/USDT:USDT", "NEWCOIN/USDT:USDT"], seed=1, branch="random")

    # drive until we get a trade decision against a symbol NOT in the simulator's universe
    for ts in range(1_700_000_000_000, 1_700_000_000_000 + 200 * 3600, 3600):
        d = agent.decide(ts=ts)
        if d.action != "hold" and d.symbol == "NEWCOIN/USDT:USDT":
            append_jsonl("decisions.jsonl", d, root=tmp_path / "LOG")
            next_bar = {"open_time": ts + 100, "open": 1.0, "volume_24h_usdt": 10_000_000}
            result = sim.execute(d, next_bar)
            from LOCKED.schemas import Rejection
            assert isinstance(result, Rejection)
            assert "universe" in result.reason.lower() or "symbol" in result.reason.lower()
            return
    raise AssertionError("random agent never produced a NEWCOIN trade decision in 200 draws - flaky seed, widen loop")


def test_scorer_and_circuit_breaker_share_nav_semantics(tmp_path):
    """scorer.daily_mark 写出的 nav.tsv 数值链能直接喂给 circuit_breaker.check。"""
    config = {"constraints": {"max_drawdown_pct": 20, "daily_loss_freeze_pct": 8}}
    scorer = Scorer(config, log_root=tmp_path / "LOG")
    navs = [100_000, 98_000, 95_000, 70_000]  # 30% drawdown from peak
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    for d, nav in zip(dates, navs):
        scorer.daily_mark(nav_agent=nav, nav_benchmark=100_000, nav_random=100_000, date=d)

    nav_tsv = (tmp_path / "LOG" / "nav.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert nav_tsv[0].split("\t") == ["date", "nav_agent", "nav_benchmark", "nav_random"]

    ts_ms = {d: 1_700_000_000_000 + i * 86_400_000 for i, d in enumerate(dates)}
    nav_series = [(ts_ms[d], nav) for d, nav in zip(dates, navs)]

    cb = CircuitBreaker(config, log_root=tmp_path / "LOG")
    state = cb.check(nav_series, now_ts=ts_ms[dates[-1]])
    assert state == "FROZEN_FULL", "30% drawdown must trip the full circuit breaker"
    assert cb.is_frozen() is True
