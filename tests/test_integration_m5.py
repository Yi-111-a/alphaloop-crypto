"""
M5 全链路集成校验:两条此前只在各自模块单测里被 fake 掉的路径,在这里用真实
对象串起来:
  1. scheduler.run_ratchet_judgment() 背后接的是一个指向真实一次性 git 仓库的
     真实 GitMergeExecutor(不是 test_main_scheduler.py 里的 FakeGitMergeExecutor)
     -- 验证 PROMOTE 判定真的能让 git 仓库发生一次真实的、可在 git log 里看到
     的合并,而不只是内存里的一个字符串指针翻转。
  2. 一次跨越"决策 + 资金费率结算"两种周期的真实进程重启场景,用两个先后构造
     的 AlphaLoopScheduler 实例(共用同一个 log_root/db)驱动同一个真实
     Simulator 账本,证明幂等性在调度器层面是真实生效的,不只是各自子模块
     分别测试过。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from LOCKED.clock import FakeClock
from LOCKED.evolution_orchestrator import EvolutionOrchestrator
from LOCKED.git_merge_executor import GitMergeExecutor
from LOCKED.scorer import Scorer
from LOCKED.simulator import Simulator

from main import AlphaLoopScheduler

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
BASE_TS = (1_700_000_000_000 // DAY_MS) * DAY_MS


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, shell=False)
    if result.returncode != 0:
        raise RuntimeError(f"git -C {cwd} {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


def _write_pkg(repo: Path, marker: str) -> None:
    (repo / "pkg.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
    (repo / "test_pkg.py").write_text(
        f"from pkg import MARKER\n\n\ndef test_marker():\n    assert MARKER == {marker!r}\n",
        encoding="utf-8",
    )


@pytest.fixture
def real_repo(tmp_path):
    repo = tmp_path / "candidate_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _write_pkg(repo, "main-version")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial commit on main")

    _git(repo, "checkout", "-b", "evo/winner")
    _write_pkg(repo, "winner-version")  # its own test still passes
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "evo/winner: change marker, tests still pass")
    _git(repo, "checkout", "main")
    return repo


def _make_sim(tmp_path, log_root, branch: str) -> Simulator:
    return Simulator(
        config=CONFIG, universe_symbols=UNIVERSE,
        db_path=tmp_path / "state" / f"portfolio_{branch.replace('/', '_')}.db",
        log_root=log_root, branch=branch, resume=True,
    )


class _CallCountingTrader:
    def __init__(self):
        self.call_count = 0

    def decide(self, ts, positions, latest_snapshot, last_reflection_summary=None,
               program_tactics=None, memory_query_text="", top_k=5, branch="main"):
        self.call_count += 1
        from LOCKED.schemas import Decision
        return [Decision(
            ts=ts, symbol="BTC/USDT:USDT", action="hold", target_notional_pct=0.0, leverage=1,
            thesis="占位thesis文本,凑够二十个字符长度用于通过simulator校验",
            falsifier="占位falsifier文本,凑够二十个字符长度用于通过simulator校验",
            horizon="4h", branch=branch,
        )]


def test_real_git_merge_through_scheduler_ratchet_judgment(tmp_path, real_repo):
    """PROMOTE at the ratchet level must translate into a REAL git merge, not
    just an in-memory pointer flip -- verified by inspecting the repo's actual
    git log/file contents afterward."""
    log_root = tmp_path / "LOG"
    clock = FakeClock(start_ms=BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")

    scorer = Scorer(CONFIG, log_root=log_root)
    orchestrator = EvolutionOrchestrator(CONFIG, scorer=scorer, log_root=log_root)
    assert orchestrator.register_branch("evo/winner", created_date="2026-01-01") is True

    git_executor = GitMergeExecutor(repo_path=real_repo, log_root=log_root)

    scheduler = AlphaLoopScheduler(
        config=CONFIG, clock=clock, simulators={"main": sim_main}, trader=_CallCountingTrader(),
        scorer=scorer, evolution_orchestrator=orchestrator, git_merge_executor=git_executor,
        log_root=log_root, state_path=tmp_path / "state" / "scheduler_state.json",
    )
    assert scheduler.effective_main_branch == "main"

    main_navs = [("2026-01-01", 100.0), ("2026-01-03", 100.5)]  # +0.5%
    winner_navs = [("2026-01-01", 100.0), ("2026-01-03", 130.0)]  # +30%, comfortably over 50bps
    benchmark_navs = [("2026-01-01", 100.0), ("2026-01-03", 100.0)]  # flat

    verdicts = scheduler.run_ratchet_judgment(
        now_date="2026-01-03",
        branch_navs={"main": main_navs, "evo/winner": winner_navs},
        benchmark_navs=benchmark_navs,
    )
    assert verdicts["evo/winner"].decision == "PROMOTE"

    # The scheduler's own bookkeeping only advances if the REAL merge succeeded.
    assert scheduler.effective_main_branch == "evo/winner"

    # And the git repo itself actually changed -- not just an in-memory fact.
    log = _git(real_repo, "log", "--oneline", "-n", "5").stdout
    assert "evo/winner" in log or "change marker" in log
    content = (real_repo / "pkg.py").read_text(encoding="utf-8")
    assert "winner-version" in content, "the real repo's main branch must now contain evo/winner's change"


def test_real_git_merge_rejection_leaves_scheduler_main_untouched(tmp_path, real_repo):
    """The mirror image: a branch that wins the ratchet but whose code fails
    its own test suite must be vetoed at the code level -- scheduler's
    effective_main_branch must NOT advance even though the ratchet said PROMOTE."""
    log_root = tmp_path / "LOG"
    clock = FakeClock(start_ms=BASE_TS)
    sim_main = _make_sim(tmp_path, log_root, "main")

    # Add a branch whose own tests fail.
    _git(real_repo, "checkout", "-b", "evo/broken")
    (real_repo / "pkg.py").write_text("MARKER = 'broken'\n", encoding="utf-8")
    (real_repo / "test_pkg.py").write_text(
        "from pkg import MARKER\n\n\ndef test_marker():\n    assert MARKER == 'this-will-never-match'\n",
        encoding="utf-8",
    )
    _git(real_repo, "add", "-A")
    _git(real_repo, "commit", "-m", "evo/broken: deliberately broken")
    _git(real_repo, "checkout", "main")

    scorer = Scorer(CONFIG, log_root=log_root)
    orchestrator = EvolutionOrchestrator(CONFIG, scorer=scorer, log_root=log_root)
    orchestrator.register_branch("evo/broken", created_date="2026-01-01")
    git_executor = GitMergeExecutor(repo_path=real_repo, log_root=log_root)

    scheduler = AlphaLoopScheduler(
        config=CONFIG, clock=clock, simulators={"main": sim_main}, trader=_CallCountingTrader(),
        scorer=scorer, evolution_orchestrator=orchestrator, git_merge_executor=git_executor,
        log_root=log_root, state_path=tmp_path / "state" / "scheduler_state.json",
    )

    main_navs = [("2026-01-01", 100.0), ("2026-01-03", 100.5)]
    broken_navs = [("2026-01-01", 100.0), ("2026-01-03", 130.0)]  # wins on paper
    benchmark_navs = [("2026-01-01", 100.0), ("2026-01-03", 100.0)]

    verdicts = scheduler.run_ratchet_judgment(
        now_date="2026-01-03",
        branch_navs={"main": main_navs, "evo/broken": broken_navs},
        benchmark_navs=benchmark_navs,
    )
    assert verdicts["evo/broken"].decision == "PROMOTE"  # wins on score...
    assert scheduler.effective_main_branch == "main"  # ...but code-level veto keeps main in place
    content = (real_repo / "pkg.py").read_text(encoding="utf-8")
    assert "broken" not in content


def test_crash_restart_across_decision_and_funding_cycles_end_to_end(tmp_path):
    """A single realistic crash-restart scenario spanning BOTH decision-cycle
    idempotency and funding-settlement catch-up, driven through two separately
    constructed AlphaLoopScheduler instances sharing the same log_root/db --
    proving the idempotency guarantees hold at the scheduler level, not just
    inside each underlying module's own isolated tests."""
    log_root = tmp_path / "LOG"
    db_path = tmp_path / "state" / "portfolio_main.db"

    def make_sim():
        return Simulator(config=CONFIG, universe_symbols=UNIVERSE, db_path=db_path,
                          log_root=log_root, branch="main", resume=True)

    trader1 = _CallCountingTrader()
    clock1 = FakeClock(start_ms=BASE_TS)
    scheduler1 = AlphaLoopScheduler(
        config=CONFIG, clock=clock1, simulators={"main": make_sim()}, trader=trader1,
        next_bar_provider=lambda symbol, ts: {"open_time": ts + 1000, "open": 50_000.0},
        funding_rate_lookup=lambda symbol, ts: 0.0001,
        snapshot_provider=lambda ts: {"BTC/USDT:USDT": {"last": 50_000.0}},
        log_root=log_root, state_path=tmp_path / "state" / "scheduler_state.json",
    )
    scheduler1.run_decision_cycle("main")
    assert trader1.call_count == 1

    # Bootstrap the funding-settlement cursor at BASE_TS (00:00 UTC) -- this is
    # the "last known good settlement point" a real deployment would already
    # have established well before any downtime occurs.
    bootstrap = scheduler1.run_settlement_catchup()
    assert bootstrap["missed_instants"] == []  # nothing missed yet, cursor just initialized

    # "Downtime": clock jumps 16h forward -- past two settlement instants
    # (08:00 and 16:00 UTC) relative to BASE_TS (aligned to 00:00 UTC).
    clock1.set_ms(BASE_TS + 16 * HOUR_MS)

    # --- "crash": scheduler1 is simply discarded, never runs its catch-up. ---

    trader2 = _CallCountingTrader()
    clock2 = FakeClock(start_ms=BASE_TS + 16 * HOUR_MS)
    scheduler2 = AlphaLoopScheduler(
        config=CONFIG, clock=clock2, simulators={"main": make_sim()}, trader=trader2,
        next_bar_provider=lambda symbol, ts: {"open_time": ts + 1000, "open": 50_000.0},
        funding_rate_lookup=lambda symbol, ts: 0.0001,
        snapshot_provider=lambda ts: {"BTC/USDT:USDT": {"last": 50_000.0}},
        log_root=log_root, state_path=tmp_path / "state" / "scheduler_state.json",
    )

    # Decision side: the ORIGINAL cycle (from BASE_TS) must not be re-decided;
    # calling run_decision_cycle again for "now" (16h later) makes exactly one
    # NEW decision for the current moment, not one for each of the ~4 missed
    # 4h cycles in between.
    result = scheduler2.run_decision_cycle("main")
    assert trader2.call_count == 1
    assert result["status"] == "decided"  # a genuinely new cycle for "now", not a backfill of the old one

    from LOCKED import log_writer
    all_decisions = log_writer.read_jsonl("decisions.jsonl", root=log_root)
    main_decisions = [d for d in all_decisions if d.get("branch") == "main"]
    assert len(main_decisions) == 2, "exactly 2 decisions total: the original + the one post-downtime, no backfill"

    # Funding side: must catch up BOTH missed settlement instants (08:00 and
    # 16:00 UTC), exactly the numbers from the reviewer's own acceptance test.
    settlement_result = scheduler2.run_settlement_catchup()
    assert len(settlement_result["missed_instants"]) == 2, (
        f"expected exactly 2 backfilled settlement instants, got: {settlement_result['missed_instants']}"
    )
    assert settlement_result["missed_instants"] == [BASE_TS + 8 * HOUR_MS, BASE_TS + 16 * HOUR_MS]

    all_funding = log_writer.read_jsonl("funding.jsonl", root=log_root)
    assert len(all_funding) == 0, (
        "no positions were ever opened in this test (only hold decisions), so both missed "
        "instants were correctly processed with zero actual settlements -- confirms "
        "missed_instants counts INSTANTS PROCESSED, not positions settled"
    )
