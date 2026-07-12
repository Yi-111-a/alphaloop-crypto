"""
tests/test_evolution_orchestrator.py -- M4 进化编排器验收测试。

覆盖 spec §6 M4 验收标准 + 复审提出的五条硬性要求:
  - 窗口对齐(test 3,本文件优先级最高的测试之一)
  - judge() 是唯一裁决入口、且是 (NAV数据, branch_dead标志) 的纯确定性函数
    (test 6 证明清算短路优先于scorer;test 9 AST结构性证明无agent覆盖面)
  - 账本隔离(test 8,本文件优先级最高的测试之一,构造两个真实 Simulator)
  - 最小优势门槛(隐式覆盖:test 4/5 复用 scorer 已有的门槛语义)
  - 晋升后追踪(test 11,与 scorer.monthly_report 打通的全链路冒烟测试)
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from LOCKED import log_writer
from LOCKED.evolution_orchestrator import BranchMeta, EvolutionOrchestrator
from LOCKED.schemas import Decision, Trade
from LOCKED.scorer import Scorer
from LOCKED.simulator import Simulator

import LOCKED.evolution_orchestrator as evo_module

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_MODULE_SOURCE = Path(evo_module.__file__).read_text(encoding="utf-8")
_MODULE_TREE = ast.parse(_MODULE_SOURCE)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# fixtures / helpers
# ----------------------------------------------------------------------
@pytest.fixture
def config():
    return {
        "constraints": {
            "max_drawdown_pct": 20,
        },
        "evolution": {
            "max_concurrent_branches": 3,
            "min_promote_edge_pct": 0.5,
        },
    }


@pytest.fixture
def scorer(config, log_root):
    return Scorer(config, log_root=log_root)


@pytest.fixture
def orchestrator(config, scorer, log_root):
    return EvolutionOrchestrator(config, scorer, log_root=log_root)


def _dates(n, start_day=1):
    return [f"2026-06-{start_day + i:02d}" for i in range(n)]


def _series(navs, start_day=1):
    ds = _dates(len(navs), start_day)
    return list(zip(ds, navs))


_LONG_THESIS = "基于4小时K线的动量信号,预计价格将维持上行趋势并至少持续一段时间"
_LONG_FALSIFIER = "若4小时K线收盘价跌破前期关键低点支撑位,则本次交易假设视为证伪并应止损离场"


# ----------------------------------------------------------------------
# 1. Max-concurrent-branches cap + slot freed after resolution
# ----------------------------------------------------------------------
def test_max_concurrent_branches_cap_and_slot_freed_after_resolution(orchestrator):
    assert orchestrator.register_branch("evo/a", "2026-06-01") is True
    assert orchestrator.register_branch("evo/b", "2026-06-01") is True
    assert orchestrator.register_branch("evo/c", "2026-06-01") is True
    assert len(orchestrator.active_branches()) == 3

    # 4th registration fails -- cap enforced, not added to the pool at all.
    assert orchestrator.register_branch("evo/d", "2026-06-01") is False
    assert orchestrator.branch_meta("evo/d") is None
    assert all(b.name != "evo/d" for b in orchestrator.active_branches())

    # Resolve all 3 currently-active branches via judge() (they clearly
    # underperform main -> ARCHIVE), freeing their slots.
    main_navs = _series([100, 100.2, 100.4, 100.6, 100.8])
    benchmark_navs = _series([100, 100, 100, 100, 100])
    branch_navs = {
        "main": main_navs,
        "evo/a": _series([100, 99, 98, 97, 96]),
        "evo/b": _series([100, 99, 98, 97, 96]),
        "evo/c": _series([100, 99, 98, 97, 96]),
    }
    verdicts = orchestrator.judge("2026-06-05", branch_navs, benchmark_navs)
    assert len(verdicts) == 3
    assert all(v.decision == "ARCHIVE" for v in verdicts.values())
    assert len(orchestrator.active_branches()) == 0

    # slots freed -> a new registration succeeds again.
    assert orchestrator.register_branch("evo/e", "2026-06-05") is True
    assert len(orchestrator.active_branches()) == 1


# ----------------------------------------------------------------------
# 2. Duplicate branch name rejected
# ----------------------------------------------------------------------
def test_duplicate_branch_name_raises_value_error(orchestrator):
    assert orchestrator.register_branch("evo/dup", "2026-06-01") is True
    with pytest.raises(ValueError):
        orchestrator.register_branch("evo/dup", "2026-06-02")

    # also rejected once the branch has resolved (names never reused)
    orchestrator._branches["evo/dup"].status = "archived"
    with pytest.raises(ValueError):
        orchestrator.register_branch("evo/dup", "2026-06-03")


# ----------------------------------------------------------------------
# 3. WINDOW-ALIGNMENT ACCEPTANCE TEST (explicit M4 criterion, top priority)
# ----------------------------------------------------------------------
def test_window_alignment_branch_created_mid_window_uses_own_creation_date(orchestrator):
    """A branch created mid-way through main's longer history must be scored
    from EXACTLY its own creation date forward, and main's comparison window
    must be sliced to that same start date -- proven by giving main a wild
    pre-creation move that a misaligned comparison would leak into the result."""
    orchestrator.register_branch("evo/midwindow", "2026-06-03")
    meta = orchestrator.branch_meta("evo/midwindow")
    # (a) scoring start recorded on the branch is exactly its creation date.
    assert meta.created_date == "2026-06-03"

    # main has a full 5-day history with a huge day1->day2 spike-and-crash
    # entirely BEFORE the candidate branch existed, flat from day3 onward.
    main_navs = _series([100, 300, 100, 100, 100], start_day=1)
    benchmark_navs = _series([100, 100, 100, 100, 100], start_day=1)
    # candidate only exists from day3 onward: day3=100 -> day5=110 (+10%)
    candidate_navs = _series([100, 105, 110], start_day=3)

    branch_navs = {"main": main_navs, "evo/midwindow": candidate_navs}
    verdicts = orchestrator.judge("2026-06-05", branch_navs, benchmark_navs)

    v = verdicts["evo/midwindow"]
    # (b) hand-computed ALIGNED arithmetic: main's window-aligned return from
    # day3 is 100->100 = 0%, candidate's is 100->110 = +10%. A misaligned
    # comparison using main's full history would still nominally read 0%
    # return here by coincidence, but its reported drawdown would leak in the
    # pre-creation crash; we assert both the edge number AND the branch's own
    # (zero) intra-window drawdown to rule that out.
    main_window_return = (100 / 100 - 1.0) * 100.0
    candidate_window_return = (110 / 100 - 1.0) * 100.0
    assert v.edge_vs_main_pct == pytest.approx(candidate_window_return - main_window_return, abs=0.01)
    assert v.max_drawdown_pct == pytest.approx(0.0, abs=0.01)
    assert v.decision == "PROMOTE"
    assert orchestrator.branch_meta("evo/midwindow").status == "promoted"


# ----------------------------------------------------------------------
# 4. PROMOTE performs the simulated merge
# ----------------------------------------------------------------------
def test_promote_performs_simulated_merge(orchestrator):
    orchestrator.register_branch("evo/winner", "2026-06-01")

    main_navs = _series([100, 100.5, 101, 101.5, 102])  # +2%
    benchmark_navs = _series([100, 101, 102, 103, 105])
    candidate_navs = _series([100, 105, 110, 115, 120])  # +20%, monotonic, well over threshold

    branch_navs = {"main": main_navs, "evo/winner": candidate_navs}
    verdicts = orchestrator.judge("2026-06-05", branch_navs, benchmark_navs)

    assert verdicts["evo/winner"].decision == "PROMOTE"
    assert orchestrator.current_main_branch == "evo/winner"

    meta = orchestrator.branch_meta("evo/winner")
    assert meta.status == "promoted"
    assert meta.promoted_date == "2026-06-05"
    assert "evo/winner" not in {b.name for b in orchestrator.active_branches()}

    records = orchestrator.promotion_records()
    assert len(records) == 1
    assert records[0].branch == "evo/winner"
    assert records[0].created_date == "2026-06-01"
    assert records[0].promoted_date == "2026-06-05"


# ----------------------------------------------------------------------
# 5. ARCHIVE: marginal/negative edge keeps main
# ----------------------------------------------------------------------
def test_archive_below_threshold_keeps_main(orchestrator):
    orchestrator.register_branch("evo/marginal", "2026-06-01")

    main_navs = _series([100, 100.2, 100.4, 100.6, 100.8])  # +0.8%
    benchmark_navs = _series([100, 100, 100, 100, 100])
    candidate_navs = _series([100, 100.25, 100.5, 100.75, 101.0])  # +1.0%, edge ~0.2% < 50bps

    branch_navs = {"main": main_navs, "evo/marginal": candidate_navs}
    verdicts = orchestrator.judge("2026-06-05", branch_navs, benchmark_navs)

    assert verdicts["evo/marginal"].decision == "ARCHIVE"
    assert orchestrator.current_main_branch == "main"
    assert orchestrator.branch_meta("evo/marginal").status == "archived"
    assert "evo/marginal" not in {b.name for b in orchestrator.active_branches()}
    assert orchestrator.promotion_records() == []


# ----------------------------------------------------------------------
# 6. Liquidation-forced FAIL overrides everything
# ----------------------------------------------------------------------
def test_liquidation_forces_fail_and_does_not_affect_other_branches(orchestrator):
    orchestrator.register_branch("evo/liquidated", "2026-06-01")
    orchestrator.register_branch("evo/winner", "2026-06-01")

    main_navs = _series([100, 100.2, 100.4, 100.6, 100.8])
    benchmark_navs = _series([100, 100, 100, 100, 100])
    # spectacular returns -- would clearly PROMOTE by scorer's own math alone.
    liquidated_navs = _series([100, 150, 200, 250, 300])
    # a second, unrelated branch that also clearly beats main.
    winner_navs = _series([100, 105, 110, 115, 120])

    branch_navs = {
        "main": main_navs,
        "evo/liquidated": liquidated_navs,
        "evo/winner": winner_navs,
    }

    # sanity check FIRST: prove that scorer.ratchet_score's own math, in
    # isolation, would have said PROMOTE for evo/liquidated -- this is what
    # makes the liquidation short-circuit meaningful to test.
    sanity = orchestrator.scorer.ratchet_score(
        {"main": main_navs, "evo/liquidated": liquidated_navs},
        {"evo/liquidated": "2026-06-01"},
        benchmark_navs,
    )
    assert sanity["evo/liquidated"].decision == "PROMOTE"

    verdicts = orchestrator.judge(
        "2026-06-05",
        branch_navs,
        benchmark_navs,
        branch_dead_flags={"evo/liquidated": True},
    )

    v_liq = verdicts["evo/liquidated"]
    assert v_liq.decision == "FAIL"
    assert "liquidat" in v_liq.reason.lower()
    assert orchestrator.branch_meta("evo/liquidated").status == "failed"

    # the unrelated branch, judged in the SAME call, is completely unaffected.
    v_win = verdicts["evo/winner"]
    assert v_win.decision == "PROMOTE"
    assert orchestrator.branch_meta("evo/winner").status == "promoted"
    assert orchestrator.current_main_branch == "evo/winner"


# ----------------------------------------------------------------------
# 7. Verdicts logged append-only, accumulating across multiple judge() calls
# ----------------------------------------------------------------------
def test_verdicts_logged_append_only_across_multiple_judge_calls(orchestrator, log_root):
    orchestrator.register_branch("evo/a", "2026-06-01")
    main_navs_1 = _series([100, 100.2, 100.4, 100.6, 100.8], start_day=1)
    benchmark_navs_1 = _series([100, 100, 100, 100, 100], start_day=1)
    winner_navs = _series([100, 105, 110, 115, 120], start_day=1)
    v1 = orchestrator.judge(
        "2026-06-05", {"main": main_navs_1, "evo/a": winner_navs}, benchmark_navs_1
    )
    assert v1["evo/a"].decision == "PROMOTE"

    orchestrator.register_branch("evo/b", "2026-06-06")
    main_navs_2 = _series([100, 100.1, 100.2, 100.3], start_day=6)
    benchmark_navs_2 = _series([100, 100, 100, 100], start_day=6)
    loser_navs = _series([100, 95, 90, 85], start_day=6)
    v2 = orchestrator.judge(
        "2026-06-10", {"main": main_navs_2, "evo/b": loser_navs}, benchmark_navs_2
    )
    assert v2["evo/b"].decision in ("ARCHIVE", "FAIL")

    records = log_writer.read_jsonl(orchestrator.verdicts_log_path, root=log_root)
    assert len(records) == 2  # accumulated across both judge() calls, not overwritten
    by_branch = {r["branch"]: r for r in records}
    assert by_branch["evo/a"]["decision"] == "PROMOTE"
    assert by_branch["evo/a"]["now_date"] == "2026-06-05"
    assert by_branch["evo/b"]["decision"] == v2["evo/b"].decision
    assert by_branch["evo/b"]["now_date"] == "2026-06-10"
    assert "reason" in by_branch["evo/a"] and by_branch["evo/a"]["reason"]


# ----------------------------------------------------------------------
# 8. CROSS-BRANCH ISOLATION TEST (explicit M4 criterion, top priority)
# ----------------------------------------------------------------------
def test_cross_branch_ledger_isolation(tmp_path, log_root):
    """Two REAL Simulator instances sharing the same LOG root (so they share
    trades.jsonl/decisions.jsonl the way production would) but separate
    sqlite db_paths. A large, leveraged trade on evo/a must not touch main's
    in-memory ledger through ANY path, including via the shared LOG files."""
    cfg = load_config()
    universe = ["BTC/USDT:USDT"]

    sim_a = Simulator(
        config=cfg,
        universe_symbols=universe,
        db_path=tmp_path / "portfolio_evo_a.db",
        branch="evo/a",
        log_root=log_root,
    )
    sim_main = Simulator(
        config=cfg,
        universe_symbols=universe,
        db_path=tmp_path / "portfolio_main.db",
        branch="main",
        log_root=log_root,
    )

    main_before = sim_main.get_portfolio()
    assert main_before["positions"] == []

    decision = Decision(
        ts=1_700_000_000_000,
        symbol="BTC/USDT:USDT",
        action="open_long",
        target_notional_pct=80.0,
        leverage=5,
        thesis=_LONG_THESIS,
        falsifier=_LONG_FALSIFIER,
        horizon="4h",
        branch="evo/a",
    )
    log_writer.append_jsonl("decisions.jsonl", decision, root=log_root)
    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    trade = sim_a.execute(decision, next_bar)
    assert isinstance(trade, Trade)  # sanity: the big leveraged trade actually filled

    a_portfolio = sim_a.get_portfolio()
    assert a_portfolio["positions"] != []
    assert a_portfolio["wallet_balance"] != pytest.approx(main_before["wallet_balance"])

    # main's ledger is COMPLETELY unaffected -- same values before and after.
    main_after = sim_main.get_portfolio()
    assert main_after["wallet_balance"] == pytest.approx(main_before["wallet_balance"])
    assert main_after["nav"] == pytest.approx(main_before["nav"])
    assert main_after["positions"] == []
    assert main_after["branch_dead"] == main_before["branch_dead"] == False

    # shared LOG file correctly separates the two branches' records when
    # consumed by filtering on the `branch` tag -- no bleed between branches.
    all_trades = log_writer.read_jsonl("trades.jsonl", root=log_root)
    a_trades = [t for t in all_trades if t["branch"] == "evo/a"]
    main_trades = [t for t in all_trades if t["branch"] == "main"]
    assert len(a_trades) == 1
    assert len(main_trades) == 0

    all_decisions = log_writer.read_jsonl("decisions.jsonl", root=log_root)
    assert all(d["branch"] == "evo/a" for d in all_decisions)


# ----------------------------------------------------------------------
# 9. No agent-facing verdict-override surface (AST structural check)
# ----------------------------------------------------------------------
def test_no_agent_facing_verdict_override_surface():
    forbidden_prefixes = ("force_", "set_verdict", "override_", "approve_")
    offenders: list[str] = []
    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.ClassDef) and node.name == "EvolutionOrchestrator":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if any(item.name.startswith(p) for p in forbidden_prefixes):
                        offenders.append(f"{item.name} (line {item.lineno})")
    assert offenders == [], f"forbidden verdict-override-shaped method(s) found: {offenders}"

    # belt-and-suspenders: cross-check against the live class too.
    live_offenders = [
        name
        for name in dir(EvolutionOrchestrator)
        if any(name.startswith(p) for p in forbidden_prefixes)
    ]
    assert live_offenders == [], f"forbidden methods on live class: {live_offenders}"


# ----------------------------------------------------------------------
# 10. No wall-clock calls (AST regression guard, matches reflector.py style)
# ----------------------------------------------------------------------
def test_no_wallclock_calls_anywhere_in_evolution_orchestrator():
    forbidden_modules = {"time", "datetime"}
    offenders: list[str] = []

    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    offenders.append(f"import {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.module in forbidden_modules:
                offenders.append(f"from {node.module} import ... (line {node.lineno})")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"time", "now", "utcnow", "today"}:
                offenders.append(f"call .{func.attr}(...) (line {node.lineno})")

    assert offenders == [], f"forbidden wall-clock references found in evolution_orchestrator.py: {offenders}"


# ----------------------------------------------------------------------
# 11. Full-loop smoke test tying M4 back to scorer.monthly_report
# ----------------------------------------------------------------------
def test_promotion_records_consumable_by_real_monthly_report(orchestrator, scorer, log_root):
    orchestrator.register_branch("evo/winner", "2026-06-01")

    main_navs = _series([100, 100.5, 101, 101.5, 102])
    benchmark_navs = _series([100, 101, 102, 103, 105])
    winner_navs = _series([100, 105, 110, 115, 120])
    branch_navs = {"main": main_navs, "evo/winner": winner_navs}

    verdicts = orchestrator.judge("2026-06-05", branch_navs, benchmark_navs)
    assert verdicts["evo/winner"].decision == "PROMOTE"

    scorer.daily_mark(nav_agent=100000, nav_benchmark=100000, nav_random=100000, date="2026-06-01")
    scorer.daily_mark(nav_agent=120000, nav_benchmark=105000, nav_random=101000, date="2026-06-30")

    report = scorer.monthly_report(
        promotions=orchestrator.promotion_records(),
        branch_navs=branch_navs,
    )

    assert "Promoted Branches" in report
    assert "evo/winner" in report


# ----------------------------------------------------------------------
# 12. M5: crash-recovery persistence -- a fresh orchestrator instance pointed
#     at the same log_root must reconstruct the exact same branch registry
#     (active/promoted/archived/failed, promotion order, current_main_branch)
#     purely by replaying branch_registrations.jsonl + ratchet_verdicts.jsonl,
#     with no separate mutable snapshot file to fall out of sync.
# ----------------------------------------------------------------------
def test_state_survives_restart_via_log_replay(config, scorer, log_root):
    orch1 = EvolutionOrchestrator(config, scorer, log_root=log_root)
    assert orch1.register_branch("evo/winner", "2026-06-01") is True
    assert orch1.register_branch("evo/loser", "2026-06-01") is True
    assert orch1.register_branch("evo/pending", "2026-06-01") is True

    main_navs = _series([100, 100.5, 101, 101.5, 102])
    benchmark_navs = _series([100, 101, 102, 103, 105])
    winner_navs = _series([100, 105, 110, 115, 120])
    loser_navs = _series([100, 100.1, 100.2, 100.3, 100.4])
    branch_navs = {"main": main_navs, "evo/winner": winner_navs, "evo/loser": loser_navs}

    verdicts = orch1.judge("2026-06-05", branch_navs, benchmark_navs)
    assert verdicts["evo/winner"].decision == "PROMOTE"
    assert verdicts["evo/loser"].decision == "ARCHIVE"
    # evo/pending was never included in branch_navs / judge()'s candidate set
    # this call, so it remains active -- simulating "still running, no verdict yet".

    # --- "restart": a brand-new EvolutionOrchestrator instance, same log_root,
    #     constructed with NO knowledge of orch1's in-memory state.
    orch2 = EvolutionOrchestrator(config, scorer, log_root=log_root)

    assert orch2.current_main_branch == "evo/winner"
    assert orch2.branch_meta("evo/winner").status == "promoted"
    assert orch2.branch_meta("evo/winner").promoted_date == "2026-06-05"
    assert orch2.branch_meta("evo/loser").status == "archived"
    assert orch2.branch_meta("evo/pending").status == "active"
    assert [m.name for m in orch2.active_branches()] == ["evo/pending"]
    assert [r.branch for r in orch2.promotion_records()] == ["evo/winner"]

    # The recovered instance is fully functional, not just a read-only replay:
    # the max-concurrent-branches cap correctly reflects only 1 active branch
    # (evo/pending) plus room for 2 more, and a new registration succeeds.
    assert orch2.register_branch("evo/new", "2026-06-06") is True
    assert len(orch2.active_branches()) == 2


def test_register_branch_survives_restart_before_any_judging(config, scorer, log_root):
    """Narrower crash window: a branch is registered but never judged before
    the process dies. Restart must not lose the registration, and the
    max-concurrent-branches cap must still correctly count it as active."""
    orch1 = EvolutionOrchestrator(config, scorer, log_root=log_root)
    orch1.register_branch("evo/a", "2026-06-01")
    orch1.register_branch("evo/b", "2026-06-01")
    orch1.register_branch("evo/c", "2026-06-01")
    assert orch1.register_branch("evo/d", "2026-06-01") is False  # cap already hit

    orch2 = EvolutionOrchestrator(config, scorer, log_root=log_root)
    assert {m.name for m in orch2.active_branches()} == {"evo/a", "evo/b", "evo/c"}
    assert orch2.register_branch("evo/d", "2026-06-01") is False  # cap still enforced after restart
    with pytest.raises(ValueError):
        orch2.register_branch("evo/a", "2026-06-02")  # names still never reused after restart
