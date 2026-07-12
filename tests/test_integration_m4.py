"""
M4 全链路集成校验:三个真实 Simulator 账本(main / 注定跑赢的分支 / 注定跑输的
分支)+ 一个真实清算分支,真实执行成交推动净值变化(不是手造NAV序列),把结果
喂给真实 Scorer + 真实 EvolutionOrchestrator,验证 spec §6 M4 验收清单的四条:
- 一个注定优于主线的分支在窗口后被 PROMOTE
- 一个劣于主线的分支被 ARCHIVE
- 熔断(爆仓)触发的分支无论收益直接 FAIL
- monthly_report 能读取晋升记录生成报告
"""
from __future__ import annotations

from LOCKED import log_writer
from LOCKED.evolution_orchestrator import EvolutionOrchestrator
from LOCKED.schemas import Trade
from LOCKED.scorer import Scorer
from LOCKED.simulator import Simulator

UNIVERSE = ["BTC/USDT:USDT"]

CONFIG = {
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
    "evolution": {"max_concurrent_branches": 3, "min_promote_edge_pct": 0.5},
}

DAY_MS = 86_400_000
BASE_TS = 1_700_000_000_000


def _make_sim(tmp_path, log_root, branch):
    return Simulator(
        config=CONFIG,
        universe_symbols=UNIVERSE,
        db_path=tmp_path / "state" / f"portfolio_{branch.replace('/', '_')}.db",
        log_root=log_root,
        branch=branch,
    )


def _open_long(sim, log_root, branch, ts, price, pct, leverage):
    from LOCKED.schemas import Decision

    decision = Decision(
        ts=ts, symbol="BTC/USDT:USDT", action="open_long", target_notional_pct=pct,
        leverage=leverage, thesis="集成测试用决策,内容占位凑够二十个字符长度",
        falsifier="集成测试用证伪条件,内容占位凑够二十个字符长度",
        falsifier_condition=f"price<{price * 0.5}", horizon="3d", branch=branch,
    )
    log_writer.append_jsonl("decisions.jsonl", decision, root=log_root)
    trade = sim.execute(decision, {"open_time": ts + 1000, "open": price})
    assert isinstance(trade, Trade), f"expected fill for {branch}, got {trade}"
    return trade


def test_full_m4_loop_promote_archive_fail_and_monthly_report(tmp_path):
    log_root = tmp_path / "LOG"

    sim_main = _make_sim(tmp_path, log_root, "main")
    sim_winner = _make_sim(tmp_path, log_root, "evo/winner")
    sim_loser = _make_sim(tmp_path, log_root, "evo/loser")
    sim_liq = _make_sim(tmp_path, log_root, "evo/liquidated")

    day1_ts = BASE_TS
    entry_price = 50_000.0

    # main: modest long, small size, low leverage -> modest gains as price rises.
    _open_long(sim_main, log_root, "main", day1_ts, entry_price, pct=20.0, leverage=2)
    # winner: aggressive long, rides the same rally hard -> big gains.
    _open_long(sim_winner, log_root, "evo/winner", day1_ts, entry_price, pct=80.0, leverage=5)
    # loser: opens a SHORT right before a rally -> loses money as price rises against it.
    from LOCKED.schemas import Decision as _D
    loser_decision = _D(
        ts=day1_ts, symbol="BTC/USDT:USDT", action="open_short", target_notional_pct=50.0,
        leverage=3, thesis="集成测试用决策,内容占位凑够二十个字符长度",
        falsifier="集成测试用证伪条件,内容占位凑够二十个字符长度",
        falsifier_condition="price>200000", horizon="3d", branch="evo/loser",
    )
    log_writer.append_jsonl("decisions.jsonl", loser_decision, root=log_root)
    loser_trade = sim_loser.execute(loser_decision, {"open_time": day1_ts + 1000, "open": entry_price})
    assert isinstance(loser_trade, Trade)
    # liquidated: 10x leverage long, about to get wiped out by a crash.
    _open_long(sim_liq, log_root, "evo/liquidated", day1_ts, entry_price, pct=50.0, leverage=10)

    # Price rallies hard over 3 days: 50000 -> 55000 -> 65000. Great for main/winner,
    # bad for the short, and the liquidated branch never gets to benefit because...
    day2_ts = day1_ts + DAY_MS
    day3_ts = day1_ts + 2 * DAY_MS
    price_day2 = 55_000.0
    price_day3 = 65_000.0

    # ...on day2 there's a sharp wick DOWN that liquidates the 10x long before the
    # rally resumes (insurance the liquidation is real, not just "bad returns").
    pos = sim_liq.positions["BTC/USDT:USDT"]
    mmr = 0.005
    margin_over_notional = pos.margin / pos.notional
    p_liq = pos.entry_price * (1 + mmr - margin_over_notional)
    liq_events = sim_liq.check_liquidation(
        {"BTC/USDT:USDT": {"high": entry_price + 100, "low": p_liq - 50.0, "close": price_day2}},
        ts_utc=day2_ts,
    )
    assert len(liq_events) == 1
    assert sim_liq.branch_dead is True

    def nav_series_for(sim, prices_by_day):
        series = []
        for date, ts, price in prices_by_day:
            portfolio = sim.get_portfolio({"BTC/USDT:USDT": price})
            series.append((date, portfolio["nav"]))
        return series

    days = [
        ("2026-01-01", day1_ts, entry_price),
        ("2026-01-02", day2_ts, price_day2),
        ("2026-01-03", day3_ts, price_day3),
    ]
    main_navs = nav_series_for(sim_main, days)
    winner_navs = nav_series_for(sim_winner, days)
    loser_navs = nav_series_for(sim_loser, days)
    liq_navs = nav_series_for(sim_liq, days)
    benchmark_navs = [(d, 100_000.0 * (p / entry_price)) for d, _, p in days]  # BTC_HOLD-equivalent

    scorer = Scorer(CONFIG, log_root=log_root)
    orchestrator = EvolutionOrchestrator(CONFIG, scorer=scorer, log_root=log_root)

    for branch in ["evo/winner", "evo/loser", "evo/liquidated"]:
        assert orchestrator.register_branch(branch, created_date="2026-01-01") is True

    branch_navs = {
        "main": main_navs,
        "evo/winner": winner_navs,
        "evo/loser": loser_navs,
        "evo/liquidated": liq_navs,
    }
    verdicts = orchestrator.judge(
        now_date="2026-01-03",
        branch_navs=branch_navs,
        benchmark_navs=benchmark_navs,
        branch_dead_flags={"evo/liquidated": sim_liq.branch_dead},
    )

    assert verdicts["evo/winner"].decision == "PROMOTE"
    assert orchestrator.current_main_branch == "evo/winner"
    assert verdicts["evo/loser"].decision == "ARCHIVE"
    assert verdicts["evo/liquidated"].decision == "FAIL"
    assert "liquidat" in verdicts["evo/liquidated"].reason.lower() or "清算" in verdicts["evo/liquidated"].reason

    # verdicts persisted append-only
    logged = log_writer.read_jsonl("ratchet_verdicts.jsonl", root=log_root)
    assert len(logged) == 3
    assert {r["branch"] for r in logged} == {"evo/winner", "evo/loser", "evo/liquidated"}

    # monthly_report needs nav.tsv (the three-line agent/benchmark/random series)
    # populated first via daily_mark -- use main as the "agent" line here.
    for date, nav in main_navs:
        bench_nav = dict(benchmark_navs)[date]
        scorer.daily_mark(nav_agent=nav, nav_benchmark=bench_nav, nav_random=bench_nav, date=date)

    report = scorer.monthly_report(promotions=orchestrator.promotion_records(), branch_navs=branch_navs)
    assert "Promoted Branches" in report
    assert "evo/winner" in report
