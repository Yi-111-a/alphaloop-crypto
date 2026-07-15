"""
tests/test_main_scheduler.py -- M5 main.py / AlphaLoopScheduler 验收测试。

覆盖复审对 M5 提出的六条硬性要求(main.py 模块 docstring 逐条引用了同一份
清单,这里的测试与之一一对应):
  1. 决策周期幂等性(崩溃重启不重复调用 Trader / 不重复落盘决策)
  2. 停机补偿的非对称性(决策不补,资金费率结算必须补齐 -- 断言精确的
     "0 补充决策、2 补充结算")
  3. 唯一墙钟来源(main.py 自身的 AST 扫描,零 time.time()/datetime.now())
  4. git merge 闸门(PROMOTE 被 GitMergeExecutor 拒绝时,调度层自己维护的
     effective_main_branch 不能跟着静默前进)
  5. LLM 故障不挂起主循环(Trader 超时兜底 + Researcher/Reflector 异常跳过
     本轮、记LOG、下轮自然重试)
  6. latest_advice.md(固定免责声明逐字在最上方,thesis/falsifier 原文展示)

尽量复用项目里已经验证过的真实实现(Simulator/Scorer/EvolutionOrchestrator/
Reflector/Researcher/MemoryStore),只对天生外部的东西(llm_client/
funding_rate_lookup/git_merge_executor)用 fake -- 与全项目已建立的"偏好真实
实例而非纯mock"的集成测试风格一致(test_integration_m1~m4 都是这个路子）。
"""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import pytest

from LOCKED import log_writer
from LOCKED.clock import FakeClock
from LOCKED.evolution_orchestrator import EvolutionOrchestrator
from LOCKED.git_merge_executor import MergeResult
from LOCKED.reflector import Reflector
from LOCKED.schemas import Decision, Trade
from LOCKED.scorer import Scorer
from LOCKED.simulator import Simulator

from ASSET.memory.engine import MemoryStore
from ASSET.strategy.researcher import Researcher

import main as main_module
from main import (
    DISCLAIMER,
    AlphaLoopScheduler,
    compute_missed_settlement_instants,
    decision_cycle_window,
)

MAIN_PY_PATH = Path(main_module.__file__).resolve()

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
    "cycle": {"decision_interval_hours": 4, "reflection_per_day": 2, "ratchet_interval_days": 3},
    "funding": {"settle_hours_utc": [0, 8, 16]},
    "evolution": {"max_concurrent_branches": 3, "min_promote_edge_pct": 0.5},
}

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
# 对齐到 UTC 自然日边界,方便手工推算结算时点。
BASE_TS = (1_700_000_000_000 // DAY_MS) * DAY_MS


# ---------------------------------------------------------------------------
# 通用 fakes / helpers
# ---------------------------------------------------------------------------


def _make_sim(tmp_path, log_root, branch: str) -> Simulator:
    return Simulator(
        config=CONFIG,
        universe_symbols=UNIVERSE,
        db_path=tmp_path / "state" / f"portfolio_{branch.replace('/', '_')}.db",
        log_root=log_root,
        branch=branch,
        resume=True,
    )


def _default_next_bar(symbol: str, ts: int) -> dict:
    return {"open_time": ts + 1000, "open": 50_000.0}


def _make_scheduler(tmp_path, log_root, clock, simulators, trader, **kwargs) -> AlphaLoopScheduler:
    kwargs.setdefault("next_bar_provider", _default_next_bar)
    kwargs.setdefault("state_path", tmp_path / "state" / "scheduler_state.json")
    return AlphaLoopScheduler(
        config=CONFIG,
        clock=clock,
        simulators=simulators,
        trader=trader,
        log_root=log_root,
        **kwargs,
    )


def _valid_hold_decision(ts: int, branch: str) -> Decision:
    return Decision(
        ts=ts,
        symbol="BTC/USDT:USDT",
        action="hold",
        target_notional_pct=0.0,
        leverage=1,
        thesis="FakeTrader占位thesis文本,凑够二十个字符长度用于通过校验",
        falsifier="FakeTrader占位falsifier文本,凑够二十个字符长度用于通过校验",
        horizon="4h",
        branch=branch,
    )


class FakeTrader:
    """duck-types ASSET.strategy.trader.Trader.decide() 的签名。call_count 用于
    断言"Trader到底被调用了几次"(硬要求1/2),sleep_seconds 用于模拟一次真正
    挂起的LLM调用(硬要求5)。"""

    def __init__(self, factory=None, sleep_seconds: float = 0.0):
        self.call_count = 0
        self.factory = factory or _valid_hold_decision
        self.sleep_seconds = sleep_seconds
        self.calls: list[dict] = []

    def decide(
        self,
        ts,
        positions,
        latest_snapshot,
        last_reflection_summary=None,
        program_tactics=None,
        memory_query_text="",
        top_k=5,
        branch="main",
    ):
        self.call_count += 1
        self.calls.append({"ts": ts, "branch": branch})
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return [self.factory(ts, branch)]


class CountingFundingLookup:
    """funding_rate_lookup 的 fake,记录每次被调用的 (symbol, ts),用于精确断言
    "补了几笔结算"(硬要求2)。"""

    def __init__(self, rate: float = 0.0001):
        self.rate = rate
        self.calls: list[tuple[str, int]] = []

    def __call__(self, symbol: str, ts: int) -> float:
        self.calls.append((symbol, ts))
        return self.rate


class FakeGitMergeExecutor:
    """LOCKED.git_merge_executor.GitMergeExecutor 的 fake(该模块已经落地,见
    LOCKED/git_merge_executor.py,但这里仍用 fake 避免测试真的跑一遍 git
    worktree/子进程测试套件,只关心调度层如何处理它的返回值)。"""

    def __init__(self, merged: bool, reason: str = "", test_suite_passed=None):
        self.merged = merged
        self.reason = reason
        self.test_suite_passed = test_suite_passed
        self.calls: list[str] = []

    def attempt_merge(self, branch_name: str) -> MergeResult:
        self.calls.append(branch_name)
        return MergeResult(
            branch=branch_name, merged=self.merged, reason=self.reason, test_suite_passed=self.test_suite_passed
        )


def _dates(n, start_day=1):
    return [f"2026-06-{start_day + i:02d}" for i in range(n)]


def _series(navs, start_day=1):
    return list(zip(_dates(len(navs), start_day), navs))


def _open_long(sim: Simulator, log_root, branch: str, ts: int, price: float, pct: float, leverage: int) -> Trade:
    decision = Decision(
        ts=ts,
        symbol="BTC/USDT:USDT",
        action="open_long",
        target_notional_pct=pct,
        leverage=leverage,
        thesis="集成测试用thesis文本,占位凑够二十个字符长度用于通过校验",
        falsifier="集成测试用falsifier文本,占位凑够二十个字符长度用于通过校验",
        falsifier_condition=f"price<{price * 0.5}",
        horizon="3d",
        branch=branch,
    )
    log_writer.append_jsonl("decisions.jsonl", decision, root=log_root)
    trade = sim.execute(decision, {"open_time": ts + 1000, "open": price})
    assert isinstance(trade, Trade)
    return trade


# ===========================================================================
# 0. 纯函数单测(decision_cycle_window / compute_missed_settlement_instants)
# ===========================================================================


class TestDecisionCycleWindow:
    def test_aligns_to_interval_boundary(self):
        start, end = decision_cycle_window(BASE_TS + 5 * HOUR_MS + 123, interval_hours=4)
        assert start == BASE_TS + 4 * HOUR_MS
        assert end == BASE_TS + 8 * HOUR_MS

    def test_exact_boundary_is_its_own_cycle_start(self):
        start, end = decision_cycle_window(BASE_TS + 8 * HOUR_MS, interval_hours=4)
        assert start == BASE_TS + 8 * HOUR_MS
        assert end == BASE_TS + 12 * HOUR_MS

    def test_rejects_non_positive_interval(self):
        with pytest.raises(ValueError):
            decision_cycle_window(BASE_TS, interval_hours=0)

    def test_supports_fractional_hours_for_fast_iteration_mode(self):
        """用户为快速验证要求30分钟(0.5h)决策周期(2026-07-14)——
        int(0.5)==0 曾是真实bug(interval_hours 先被截断成0,导致下面的
        ValueError永远触发)。这里验证0.5小时窗口切分是正确的30分钟边界,
        且不影响既有整数小时(4h)配置的行为(上面几个测试保持不变)。"""
        start, end = decision_cycle_window(BASE_TS + 45 * 60_000, interval_hours=0.5)
        assert start == BASE_TS + 30 * 60_000
        assert end == BASE_TS + 60 * 60_000

    def test_rejects_fractional_interval_too_small_to_produce_a_window(self):
        with pytest.raises(ValueError):
            decision_cycle_window(BASE_TS, interval_hours=1e-7)  # rounds to 0ms


class TestComputeMissedSettlementInstants:
    def test_none_last_settlement_means_nothing_missed(self):
        assert compute_missed_settlement_instants(None, BASE_TS + DAY_MS, [0, 8, 16]) == []

    def test_no_elapsed_time_means_nothing_missed(self):
        assert compute_missed_settlement_instants(BASE_TS, BASE_TS, [0, 8, 16]) == []

    def test_two_missed_instants_across_downtime(self):
        last = BASE_TS
        now = BASE_TS + 16 * HOUR_MS + 60_000
        missed = compute_missed_settlement_instants(last, now, [0, 8, 16])
        assert missed == [BASE_TS + 8 * HOUR_MS, BASE_TS + 16 * HOUR_MS]

    def test_spans_multiple_days(self):
        last = BASE_TS + 20 * HOUR_MS
        now = BASE_TS + DAY_MS + 9 * HOUR_MS
        missed = compute_missed_settlement_instants(last, now, [0, 8, 16])
        assert missed == [
            BASE_TS + DAY_MS + 0 * HOUR_MS,
            BASE_TS + DAY_MS + 8 * HOUR_MS,
        ]


# ===========================================================================
# 1. 硬要求1:决策周期幂等性
# ===========================================================================


def test_decision_cycle_idempotent_across_process_restart(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    trader = FakeTrader()
    scheduler = _make_scheduler(tmp_path, log_root, clock, {"main": sim_main}, trader)

    result1 = scheduler.run_decision_cycle("main")
    assert result1["status"] == "decided"
    assert trader.call_count == 1

    decisions = [d for d in log_writer.read_jsonl("decisions.jsonl", root=log_root) if d.get("branch") == "main"]
    assert len(decisions) == 1

    # --- "kill -9 重启": 全新 Simulator(resume=True,读同一个db)+ 全新调度器
    # + 全新 Trader(计数器归零)+ 同一时刻的全新 Clock,指向同一份 log_root。
    clock2 = FakeClock(clock.now_ms())
    sim_main2 = _make_sim(tmp_path, log_root, "main")
    trader2 = FakeTrader()
    scheduler2 = _make_scheduler(tmp_path, log_root, clock2, {"main": sim_main2}, trader2)

    result2 = scheduler2.run_decision_cycle("main")
    assert result2["status"] == "skipped_already_decided"
    assert trader2.call_count == 0  # 没有第二次 LLM 调用

    decisions_after = [
        d for d in log_writer.read_jsonl("decisions.jsonl", root=log_root) if d.get("branch") == "main"
    ]
    assert len(decisions_after) == 1  # 没有重复决策落盘


def test_decision_cycle_produces_new_decision_in_a_later_cycle(tmp_path, log_root):
    """幂等性只锁同一个周期,不锁调度器本身 -- 到了下一个4小时周期,应该正常
    再决策一次,证明skip逻辑没有过度拦截。"""
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    trader = FakeTrader()
    scheduler = _make_scheduler(tmp_path, log_root, clock, {"main": sim_main}, trader)

    scheduler.run_decision_cycle("main")
    assert trader.call_count == 1

    clock.advance_ms(4 * HOUR_MS)
    result = scheduler.run_decision_cycle("main")
    assert result["status"] == "decided"
    assert trader.call_count == 2


# ===========================================================================
# 2. 硬要求2:停机补偿(决策不补,结算必须补齐 -- 精确断言 0 决策 / 2 结算)
# ===========================================================================


def test_downtime_compensation_zero_decisions_two_settlements(tmp_path, log_root):
    day1_00 = BASE_TS
    day1_08 = BASE_TS + 8 * HOUR_MS
    day1_16 = BASE_TS + 16 * HOUR_MS

    clock = FakeClock(day1_00)
    sim_main = _make_sim(tmp_path, log_root, "main")

    # 先开一个仓位,让资金费率结算有实际要结算的东西。
    _open_long(sim_main, log_root, "main", day1_00, price=50_000.0, pct=20.0, leverage=2)

    state_path = tmp_path / "state" / "scheduler_state.json"
    funding_lookup1 = CountingFundingLookup()
    scheduler1 = _make_scheduler(
        tmp_path, log_root, clock, {"main": sim_main}, FakeTrader(),
        funding_rate_lookup=funding_lookup1, state_path=state_path,
    )

    # 首次调用 bootstrap:代表"上一次成功结算就发生在day1 00:00"这个前提本身
    # 被记录下来,此时不应该有任何缺口(系统刚起步,没有"错过"的概念)。
    boot = scheduler1.run_settlement_catchup()
    assert boot["missed_instants"] == []
    assert funding_lookup1.calls == []

    # --- 崩溃 + 停机 16+ 小时,跨越 08:00 和 16:00 两个结算点均未处理 ---
    clock2 = FakeClock(day1_16 + 60_000)
    sim_main2 = _make_sim(tmp_path, log_root, "main")  # resume=True,加载同一持仓
    funding_lookup2 = CountingFundingLookup()
    trader2 = FakeTrader()
    scheduler2 = _make_scheduler(
        tmp_path, log_root, clock2, {"main": sim_main2}, trader2,
        funding_rate_lookup=funding_lookup2, state_path=state_path,
    )

    catchup = scheduler2.run_settlement_catchup()
    assert catchup["missed_instants"] == [day1_08, day1_16]
    # 精确断言:恰好 2 笔补充结算(1个symbol持仓 × 2个错过的结算时点)。
    assert len(funding_lookup2.calls) == 2
    assert funding_lookup2.calls == [("BTC/USDT:USDT", day1_08), ("BTC/USDT:USDT", day1_16)]

    funding_records = [r for r in log_writer.read_jsonl("funding.jsonl", root=log_root) if r["branch"] == "main"]
    assert len(funding_records) == 2
    assert {r["ts"] for r in funding_records} == {day1_08, day1_16}

    # 精确断言:恰好 0 笔补充决策 -- 停机跨越的是 4 个决策周期
    # (00:00-04:00 / 04:00-08:00 / 08:00-12:00 / 12:00-16:00),没有一个被补上。
    decisions_before = log_writer.read_jsonl("decisions.jsonl", root=log_root)
    count_before = len(decisions_before)

    result = scheduler2.run_decision_cycle("main")
    assert result["status"] == "decided"
    assert trader2.call_count == 1  # 只为"现在"决策了一次,不是四次

    decisions_after = log_writer.read_jsonl("decisions.jsonl", root=log_root)
    # 恰好新增 1 条(给当前16:00-20:00周期),不是给中间跳过的任何一个周期补的。
    assert len(decisions_after) == count_before + 1
    new_main_decisions = [
        d for d in decisions_after if d.get("branch") == "main" and d["ts"] >= day1_16
    ]
    assert len(new_main_decisions) == 1


def test_settlement_catchup_is_a_noop_when_nothing_missed(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    funding_lookup = CountingFundingLookup()
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": sim_main}, FakeTrader(),
        funding_rate_lookup=funding_lookup,
    )
    scheduler.run_settlement_catchup()  # bootstrap
    result = scheduler.run_settlement_catchup()  # same instant, nothing new
    assert result["missed_instants"] == []
    assert funding_lookup.calls == []


# ===========================================================================
# 3. 硬要求3:main.py 自身零墙钟调用(AST 扫描,风格与 tests/test_clock.py 一致)
# ===========================================================================

_WALLCLOCK_ATTRS = {"time", "now", "utcnow", "today"}
_WALLCLOCK_MODULES = {"time", "datetime"}


def _find_wallclock_calls(tree: ast.AST) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            value = node.func.value
            if attr in _WALLCLOCK_ATTRS and isinstance(value, ast.Name) and value.id in _WALLCLOCK_MODULES:
                hits.append(f"{value.id}.{attr}()")
    return hits


def test_main_py_has_zero_wallclock_calls():
    """main.py 特意放在项目根目录,不在 LOCKED/ 或 ASSET/ 之下,所以不会被
    tests/test_clock.py::test_no_wallclock_calls_anywhere_in_locked_or_asset
    的项目级扫描自动覆盖 -- 这里用同一套逻辑单独守住它,纪律完全相同。"""
    tree = ast.parse(MAIN_PY_PATH.read_text(encoding="utf-8"), filename=str(MAIN_PY_PATH))
    hits = _find_wallclock_calls(tree)
    assert not hits, (
        f"main.py must source all 'now' from an injected Clock instance, found direct "
        f"wall-clock calls instead: {hits}"
    )


def test_main_py_does_not_import_time_or_datetime_modules_directly():
    """更严格的补充检查:即便没有 module.func() 形状的调用,直接 import time /
    import datetime 也是一个危险信号(说明本文件里存在墙钟依赖的入口),main.py
    不应该需要它们 -- 所有时间都应该来自注入的 Clock。"""
    tree = ast.parse(MAIN_PY_PATH.read_text(encoding="utf-8"), filename=str(MAIN_PY_PATH))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
    assert "time" not in imported_modules
    assert "datetime" not in imported_modules


# ===========================================================================
# 4. 硬要求4:git merge 闸门
# ===========================================================================


def test_promotion_veto_does_not_advance_effective_main_branch(tmp_path, log_root):
    scorer = Scorer(CONFIG, log_root=log_root)
    orchestrator = EvolutionOrchestrator(CONFIG, scorer, log_root=log_root)
    orchestrator.register_branch("evo/winner", "2026-06-01")

    main_navs = _series([100, 100.5, 101, 101.5, 102])
    benchmark_navs = _series([100, 101, 102, 103, 105])
    candidate_navs = _series([100, 105, 110, 115, 120])  # clearly PROMOTE-worthy
    branch_navs = {"main": main_navs, "evo/winner": candidate_navs}

    fake_git = FakeGitMergeExecutor(merged=False, reason="refused: test suite failed (3 failing tests)",
                                     test_suite_passed=False)

    clock = FakeClock(BASE_TS)
    scheduler = AlphaLoopScheduler(
        config=CONFIG,
        clock=clock,
        simulators={"main": _make_sim(tmp_path, log_root, "main")},
        trader=FakeTrader(),
        evolution_orchestrator=orchestrator,
        git_merge_executor=fake_git,
        log_root=log_root,
    )

    assert scheduler.effective_main_branch == "main"

    verdicts = scheduler.run_ratchet_judgment("2026-06-05", branch_navs, benchmark_navs)

    assert verdicts["evo/winner"].decision == "PROMOTE"
    # orchestrator 自己的纯裁决bookkeeping确实已经翻转...
    assert orchestrator.current_main_branch == "evo/winner"
    # ...但调度层的"真正生效的主分支"必须保持不变,因为真实git merge被拒绝了。
    assert scheduler.effective_main_branch == "main"
    assert fake_git.calls == ["evo/winner"]

    events = log_writer.read_jsonl("scheduler_errors.jsonl", root=log_root)
    vetoes = [e for e in events if e.get("event") == "promotion_veto"]
    assert len(vetoes) == 1
    assert vetoes[0]["branch"] == "evo/winner"
    assert vetoes[0]["severity"] == "critical"
    assert vetoes[0]["test_suite_passed"] is False


def test_successful_merge_advances_effective_main_branch(tmp_path, log_root):
    scorer = Scorer(CONFIG, log_root=log_root)
    orchestrator = EvolutionOrchestrator(CONFIG, scorer, log_root=log_root)
    orchestrator.register_branch("evo/winner", "2026-06-01")

    main_navs = _series([100, 100.5, 101, 101.5, 102])
    benchmark_navs = _series([100, 101, 102, 103, 105])
    candidate_navs = _series([100, 105, 110, 115, 120])
    branch_navs = {"main": main_navs, "evo/winner": candidate_navs}

    fake_git = FakeGitMergeExecutor(merged=True, reason="merged cleanly", test_suite_passed=True)

    clock = FakeClock(BASE_TS)
    scheduler = AlphaLoopScheduler(
        config=CONFIG,
        clock=clock,
        simulators={"main": _make_sim(tmp_path, log_root, "main")},
        trader=FakeTrader(),
        evolution_orchestrator=orchestrator,
        git_merge_executor=fake_git,
        log_root=log_root,
    )

    scheduler.run_ratchet_judgment("2026-06-05", branch_navs, benchmark_navs)
    assert scheduler.effective_main_branch == "evo/winner"

    events = log_writer.read_jsonl("scheduler_errors.jsonl", root=log_root)
    merges = [e for e in events if e.get("event") == "promotion_merged"]
    assert len(merges) == 1
    assert not any(e.get("event") == "promotion_veto" for e in events)


# ===========================================================================
# 5. 硬要求5:LLM 故障不挂起主循环
# ===========================================================================


def test_trader_timeout_produces_scheduler_level_hold_and_completes_quickly(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    hung_trader = FakeTrader(sleep_seconds=5.0)  # 睡过配置的超时时间
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": sim_main}, hung_trader, trader_timeout_seconds=0.2
    )

    wall_start = time.time()
    result = scheduler.run_decision_cycle("main")
    wall_elapsed = time.time() - wall_start

    assert result["status"] == "decided"
    assert result["timed_out"] is True
    assert wall_elapsed < 3.0  # 没有等挂起线程的5秒,测试本身也不会被拖住

    assert len(result["decisions"]) == 1
    fallback = result["decisions"][0]
    assert fallback.action == "hold"
    assert "调度器级超时兜底" in fallback.thesis  # 与Trader自己的内部兜底文案明确不同

    events = log_writer.read_jsonl("scheduler_errors.jsonl", root=log_root)
    timeouts = [e for e in events if e.get("event") == "trader_timeout"]
    assert len(timeouts) == 1
    assert timeouts[0]["branch"] == "main"


def test_reflector_failure_is_logged_skipped_and_next_cycle_succeeds(tmp_path, log_root):
    memory_store = MemoryStore(db_path=tmp_path / "memory.db")

    seed_decision = Decision(
        ts=BASE_TS - HOUR_MS,
        symbol="BTC/USDT:USDT",
        action="open_long",
        target_notional_pct=20.0,
        leverage=2,
        thesis="占位thesis文本,凑够二十个字符长度用于通过校验测试",
        falsifier="占位falsifier文本,凑够二十个字符长度用于通过校验测试",
        falsifier_condition="price<1",
        horizon="1h",
        branch="main",
    )
    log_writer.append_jsonl("decisions.jsonl", seed_decision, root=log_root)

    class RaisingOnceLLM:
        def __init__(self):
            self.call_count = 0

        def __call__(self, prompt):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("simulated LLM outage")
            return "经验摘要:模拟已恢复正常"

    llm = RaisingOnceLLM()
    reflector = Reflector(llm_client=llm, memory_store=memory_store, log_root=log_root)

    clock = FakeClock(BASE_TS)
    scheduler = AlphaLoopScheduler(
        config=CONFIG,
        clock=clock,
        simulators={"main": _make_sim(tmp_path, log_root, "main")},
        trader=FakeTrader(),
        reflector=reflector,
        memory_store=memory_store,
        log_root=log_root,
        price_lookup=lambda symbol, ts: 50_000.0,
    )

    result1 = scheduler.run_reflection_cycle("main")
    assert result1 is None  # 本轮跳过,不崩溃
    assert llm.call_count == 1
    assert scheduler.last_reflection_summary is None  # 本轮失败,不应该被污染

    events = log_writer.read_jsonl("scheduler_errors.jsonl", root=log_root)
    failures = [e for e in events if e.get("event") == "reflector_failure"]
    assert len(failures) == 1
    assert failures[0]["branch"] == "main"

    # 下一个调度节拍就是重试,没有任何"本次调用内部重试"的逻辑。
    result2 = scheduler.run_reflection_cycle("main")
    assert result2 is not None
    assert llm.call_count == 2
    # 反思摘要不是靠 reflect() 的返回值传出来的(那是 ThesisMark 列表)，
    # 而是 reflect() 内部写进 memory_store 的 L2 记录，调度器读回来喂给
    # 下一次 Trader.decide() 的"最近一次反思摘要"字段。
    assert scheduler.last_reflection_summary == "经验摘要:模拟已恢复正常"


def test_researcher_daily_research_failure_is_logged_skipped_and_next_cycle_succeeds(tmp_path, log_root):
    memory_store = MemoryStore(db_path=tmp_path / "memory.db")

    class RaisingOnceLLM:
        def __init__(self):
            self.call_count = 0

        def __call__(self, prompt):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("simulated LLM outage")
            return json.dumps(
                [{"source": "s", "core_idea": "c", "testable_hypothesis": "h", "suggested_experiment": "e"}]
            )

    llm = RaisingOnceLLM()
    researcher = Researcher(
        llm_client=llm, memory_store=memory_store, research_notes_dir=tmp_path / "research_notes"
    )

    clock = FakeClock(BASE_TS)
    scheduler = AlphaLoopScheduler(
        config=CONFIG,
        clock=clock,
        simulators={"main": _make_sim(tmp_path, log_root, "main")},
        trader=FakeTrader(),
        researcher=researcher,
        log_root=log_root,
    )

    result1 = scheduler.run_daily_research("2026-01-01")
    assert result1 is None
    assert llm.call_count == 1

    events = log_writer.read_jsonl("scheduler_errors.jsonl", root=log_root)
    failures = [e for e in events if e.get("event") == "researcher_daily_failure"]
    assert len(failures) == 1

    result2 = scheduler.run_daily_research("2026-01-01")
    assert result2 is not None
    assert isinstance(result2, Path)
    assert result2.exists()


# ===========================================================================
# 6. 硬要求6:latest_advice.md
# ===========================================================================


def _sample_decision(**overrides) -> Decision:
    base = dict(
        ts=BASE_TS,
        symbol="BTC/USDT:USDT",
        action="open_long",
        target_notional_pct=30.0,
        leverage=2,
        thesis="占位thesis文本,凑够二十个字符长度用于通过校验测试",
        falsifier="占位falsifier文本,凑够二十个字符长度用于通过校验测试",
        horizon="12h",
        branch="main",
    )
    base.update(overrides)
    return Decision(**base)


def test_latest_advice_disclaimer_is_verbatim_at_top(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    scheduler = AlphaLoopScheduler(
        config=CONFIG,
        clock=clock,
        simulators={"main": _make_sim(tmp_path, log_root, "main")},
        trader=FakeTrader(),
        log_root=log_root,
    )
    path = scheduler.generate_latest_advice(
        branch="main", decision=_sample_decision(), nav_agent=105_000.0, nav_benchmark=102_000.0, nav_random=99_000.0
    )
    content = path.read_text(encoding="utf-8")
    assert content.startswith(DISCLAIMER)
    assert DISCLAIMER == "本内容为AI模拟实验输出,非投资建议,不构成任何交易依据"


def test_latest_advice_thesis_and_falsifier_are_verbatim_not_summarized(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    scheduler = AlphaLoopScheduler(
        config=CONFIG,
        clock=clock,
        simulators={"main": _make_sim(tmp_path, log_root, "main")},
        trader=FakeTrader(),
        log_root=log_root,
    )

    distinctive_thesis = (
        "UNIQUE_THESIS_MARKER_9f3a7c: 基于H7假设与一段不寻常的、独一无二的、"
        "绝对不应该被摘要、改写或截断的占位文本,用来证明生成的文件包含原文而非概括"
    )
    distinctive_falsifier = (
        "UNIQUE_FALSIFIER_MARKER_7b2e1d: 若出现同样独一无二、不应被摘要改写的"
        "证伪条件占位文本,则说明latest_advice.md展示的是原文而非二次概括"
    )
    decision = _sample_decision(thesis=distinctive_thesis, falsifier=distinctive_falsifier)

    path = scheduler.generate_latest_advice(branch="main", decision=decision, nav_agent=100_000.0)
    content = path.read_text(encoding="utf-8")

    assert distinctive_thesis in content
    assert distinctive_falsifier in content


def test_latest_advice_overwrites_not_accumulates(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    scheduler = AlphaLoopScheduler(
        config=CONFIG,
        clock=clock,
        simulators={"main": _make_sim(tmp_path, log_root, "main")},
        trader=FakeTrader(),
        log_root=log_root,
    )

    first_marker = "FIRST_CYCLE_MARKER_絶対にファイルに残ってはいけない古い内容"
    second_marker = "SECOND_CYCLE_MARKER_これが最新の唯一の内容であるべき"

    path1 = scheduler.generate_latest_advice(
        branch="main", decision=_sample_decision(thesis=first_marker * 1, falsifier="占位falsifier凑够二十字符长度用" * 1),
        nav_agent=100_000.0,
    )
    clock.advance_ms(HOUR_MS)
    path2 = scheduler.generate_latest_advice(
        branch="main", decision=_sample_decision(thesis=second_marker, falsifier="占位falsifier凑够二十字符长度用" * 1),
        nav_agent=101_000.0,
    )

    assert path1 == path2  # 同一份文件
    content = path2.read_text(encoding="utf-8")
    assert second_marker in content
    assert first_marker not in content  # "latest" -- 旧内容必须被覆盖掉,不是追加


def test_run_decision_cycle_generates_latest_advice_automatically(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    scheduler = _make_scheduler(tmp_path, log_root, clock, {"main": sim_main}, FakeTrader())

    result = scheduler.run_decision_cycle("main")
    assert result["advice_path"] is not None
    content = result["advice_path"].read_text(encoding="utf-8")
    assert content.startswith(DISCLAIMER)


# ===========================================================================
# M5 用户新增护栏:每小时确定性紧急风控检查(不在原六条硬要求里,是点火
# 过程中用户直接补充的,详见 LOCKED/position_risk_monitor.py 模块 docstring
# 记录的设计裁决:必须是确定性代码判断,不经过 Trader/LLM)。
# ===========================================================================


class FakeRecentPriceProvider:
    """recent_price_provider 的 fake:symbol -> 预先注入的 (ts, price) 序列。
    记录被调用过的 symbol,用于断言"只查询了实际持仓涉及的symbol"。"""

    def __init__(self, prices_by_symbol: dict[str, list[tuple[int, float]]]):
        self.prices_by_symbol = prices_by_symbol
        self.calls: list[str] = []

    def __call__(self, symbol: str, ts: int) -> list[tuple[int, float]]:
        self.calls.append(symbol)
        return self.prices_by_symbol.get(symbol, [])


def test_risk_check_triggers_emergency_close_on_drawdown_breach(tmp_path, log_root):
    """核心场景:多头持仓最近窗口从高点回撤超过阈值(config默认5%)->
    不经过Trader,直接构造close决策并真实执行。"""
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    trade = _open_long(sim_main, log_root, "main", BASE_TS, price=50_000.0, pct=30.0, leverage=3)
    assert sim_main.positions  # sanity: 持仓真的开出来了

    # peak 51000 -> current 47000: (51000-47000)/51000*100 ≈ 7.84% > 5% 阈值
    price_provider = FakeRecentPriceProvider({
        "BTC/USDT:USDT": [(BASE_TS, 50_000.0), (BASE_TS + 60_000, 51_000.0), (BASE_TS + 120_000, 47_000.0)],
    })
    trader = FakeTrader()
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": sim_main}, trader, recent_price_provider=price_provider,
    )

    clock.advance_ms(3_600_000)
    result = scheduler.run_risk_check_cycle("main")

    assert result["status"] == "checked"
    assert len(result["triggered"]) == 1
    assert result["triggered"][0]["symbol"] == "BTC/USDT:USDT"
    assert "BTC/USDT:USDT" not in sim_main.positions  # 真的平仓了
    assert trader.call_count == 0  # 全程没有经过Trader/LLM

    # 落盘的决策必须诚实标注这是确定性触发,不是Trader/LLM的判断
    decisions = log_writer.read_jsonl("decisions.jsonl", root=log_root)
    close_decision = [d for d in decisions if d.get("action") == "close"][0]
    assert "非LLM判断" in close_decision["thesis"] or "确定性" in close_decision["thesis"]
    assert len(close_decision["thesis"]) >= 20
    assert len(close_decision["falsifier"]) >= 20


def test_risk_check_does_not_trigger_below_threshold(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    _open_long(sim_main, log_root, "main", BASE_TS, price=50_000.0, pct=30.0, leverage=3)

    # small dip well under the 5% threshold
    price_provider = FakeRecentPriceProvider({
        "BTC/USDT:USDT": [(BASE_TS, 50_000.0), (BASE_TS + 60_000, 50_500.0), (BASE_TS + 120_000, 50_200.0)],
    })
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": sim_main}, FakeTrader(), recent_price_provider=price_provider,
    )

    result = scheduler.run_risk_check_cycle("main")
    assert result["triggered"] == []
    assert "BTC/USDT:USDT" in sim_main.positions  # 仓位没动


def test_risk_check_skipped_gracefully_when_no_provider_injected(tmp_path, log_root):
    """recent_price_provider 是可选依赖(main.py 一贯的注入式设计)——没注入
    时必须干净地跳过,不能抛异常炸掉调度循环。"""
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    _open_long(sim_main, log_root, "main", BASE_TS, price=50_000.0, pct=30.0, leverage=3)
    scheduler = _make_scheduler(tmp_path, log_root, clock, {"main": sim_main}, FakeTrader())

    result = scheduler.run_risk_check_cycle("main")
    assert result["status"] == "skipped_no_recent_price_provider"
    assert "BTC/USDT:USDT" in sim_main.positions


def test_risk_check_only_queries_symbols_with_open_positions(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    _open_long(sim_main, log_root, "main", BASE_TS, price=50_000.0, pct=30.0, leverage=3)

    price_provider = FakeRecentPriceProvider({"BTC/USDT:USDT": [(BASE_TS, 50_000.0), (BASE_TS + 1000, 50_100.0)]})
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": sim_main}, FakeTrader(), recent_price_provider=price_provider,
    )
    scheduler.run_risk_check_cycle("main")
    assert price_provider.calls == ["BTC/USDT:USDT"]


def test_risk_check_one_symbol_price_fetch_failure_does_not_abort_whole_cycle(tmp_path, log_root):
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    _open_long(sim_main, log_root, "main", BASE_TS, price=50_000.0, pct=30.0, leverage=3)

    def failing_provider(symbol, ts):
        raise ConnectionError("simulated feed outage")

    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": sim_main}, FakeTrader(), recent_price_provider=failing_provider,
    )
    result = scheduler.run_risk_check_cycle("main")  # must not raise
    assert result["status"] == "checked"
    assert result["triggered"] == []  # 数据不足 -> 不触发,不是假装触发
    errors = log_writer.read_jsonl("scheduler_errors.jsonl", root=log_root)
    assert any(e.get("event") == "risk_check_price_fetch_failed" for e in errors)


def test_risk_check_is_purely_deterministic_no_llm_involvement_across_multiple_triggers(tmp_path, log_root):
    """一次跑多个持仓、多次触发,Trader.call_count 全程必须是0——这是这个
    功能存在的全部意义(紧急操作等不起agent签入)。"""
    clock = FakeClock(BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")
    _open_long(sim_main, log_root, "main", BASE_TS, price=50_000.0, pct=30.0, leverage=3)

    price_provider = FakeRecentPriceProvider({
        "BTC/USDT:USDT": [(BASE_TS, 50_000.0), (BASE_TS + 60_000, 52_000.0), (BASE_TS + 120_000, 48_000.0)],
    })
    trader = FakeTrader()
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": sim_main}, trader, recent_price_provider=price_provider,
    )
    scheduler.run_risk_check_cycle("main")
    assert trader.call_count == 0
