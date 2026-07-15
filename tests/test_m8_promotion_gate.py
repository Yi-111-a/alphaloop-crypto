"""
tests/test_m8_promotion_gate.py —— M8 晋升闸门重构验收测试(改造规格书M8§4.4)。

覆盖:
  1. 零LLM断言(§4.4第一条):带policy_id的名册分支经 DispatchingTrader 走
     scheduler.run_decision_cycle,决策产出自M7策略代码,llm调用计数==0;
     无policy_id分支回退到LLM路径(计数==1),既有提示词分支行为不受影响。
  2. 窗口拉长防误判(§4.4第二条):合成"12小时内偶然+1%但7天累计-2%"的
     分支净值 vs "7天+1%"的main,断言日级 evaluate_tactic_tournament 不判
     它PROMOTE——这正是M8把edge分母从小时级换回日级+拉长窗口要防住的
     那类噪声晋升。
  3. 端到端(§4.4第三条):tmp_path 一次性真实git仓库 + FakeClock:
     admit_policy_to_forward_pool → 合成达标日级净值 → 锦标赛PROMOTE →
     promote_policy_branch → 断言 EvolutionOrchestrator.judge() 真实返回
     PROMOTE、GitMergeExecutor.attempt_merge() 真实执行(git log 可见
     merge commit)、ratchet_verdicts.jsonl / branch_registrations.jsonl 有
     真实记录——修复规格书§0.1诊断2"这条生产链路从未被真实调用过"。
  4. monthly_report 新增"回测vs前向一致性"栏:有配对记录时正确输出差值,
     无记录时优雅跳过不报错。
  5. DispatchingTrader 的安全行为:数据不足时策略返回空列表,调度器照常
     完成周期(不合成假hold、不炸循环);positions list/dict 两种形状兼容。

测试风格:与 tests/test_main_scheduler.py 一致(FakeClock + 真实
Simulator/Scorer/EvolutionOrchestrator,只对天生外部的东西用fake);git
部分与 tests/test_git_merge_executor.py 一致(tmp_path 下构造真实一次性
仓库,绝不触碰 alphaloop 自身的仓库/生产 state/LOG)。ignite.py 的模块级
路径常量(STATE_ROOT/LOG_ROOT/PROJECT_ROOT/各json路径)在测试里统一
monkeypatch 到 tmp_path,保证测试不写生产目录。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from LOCKED import log_writer
from LOCKED.clock import FakeClock
from LOCKED.evolution_orchestrator import EvolutionOrchestrator
from LOCKED.git_merge_executor import GitMergeExecutor
from LOCKED.scorer import Scorer
from LOCKED.simulator import Simulator

from ASSET.strategy.policy_trader import DispatchingTrader
from ASSET.strategy.trader import Trader

import scripts.ignite as ignite
from main import AlphaLoopScheduler

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
# 对齐到UTC自然日边界,方便手工推算日级降采样结果。
BASE_TS = (1_700_000_000_000 // DAY_MS) * DAY_MS

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
    "cycle": {"decision_interval_hours": 4},
    "funding": {"settle_hours_utc": [0, 8, 16]},
    # M8§4.2 收敛后的数值(与config.yaml对齐,测试用同一口径)
    "evolution": {"max_concurrent_branches": 4, "min_promote_edge_pct": 0.5},
    "tactic_tournament": {
        "min_hours_before_judgment": 168,
        "promote_edge_pct": 0.5,
        "fail_drawdown_pct": 15,
        "cull_interval_hours": 72,
        "cull_min_age_hours": 72,
    },
}


# ---------------------------------------------------------------------------
# fakes / helpers
# ---------------------------------------------------------------------------


class CountingLLM:
    """llm_client fake:计数器 + 固定返回一条合法的hold决策JSON。零LLM断言
    的核心观测点——policy路径全程不许碰它。"""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        return json.dumps([
            {
                "ts": 1,
                "symbol": "BTC/USDT:USDT",
                "action": "hold",
                "target_notional_pct": 0.0,
                "leverage": 1,
                "thesis": "LLM路径占位thesis文本,凑够二十个字符长度用于通过校验",
                "falsifier": "LLM路径占位falsifier文本,凑够二十个字符长度用于通过校验",
                "horizon": "4h",
            }
        ])


class FakeMemoryStore:
    def retrieve(self, query, query_ts, top_k=5, branch=None):
        return []


class FakeDataPipeline:
    """fetch_ohlcv 的离线fake:按symbol返回预先注入的合成K线,记录调用参数
    (limit是否带上了REQUIRED_HISTORY_BARS+缓冲)。"""

    def __init__(self, bars_by_symbol: dict[str, pd.DataFrame]):
        self.bars_by_symbol = bars_by_symbol
        self.calls: list[tuple[str, str, int]] = []

    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", since=None, limit: int = 1000) -> pd.DataFrame:
        self.calls.append((symbol, timeframe, limit))
        df = self.bars_by_symbol.get(symbol, pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        ))
        return df.tail(limit).reset_index(drop=True)


def _make_bars(n: int, base_price: float, drift_pct: float) -> pd.DataFrame:
    """确定性合成K线(同 tests/test_policies.py 的手法,无随机性)。"""
    rows = []
    prev_close = base_price
    interval_ms = 4 * HOUR_MS
    for i in range(n):
        close = base_price * (1 + drift_pct) ** i
        open_ = prev_close
        high = max(open_, close) * 1.002
        low = min(open_, close) * 0.998
        rows.append({
            "timestamp": BASE_TS - (n - i) * interval_ms,
            "open": open_, "high": high, "low": low, "close": close, "volume": 1000.0,
        })
        prev_close = close
    return pd.DataFrame(rows)


def _make_sim(tmp_path, log_root, branch: str) -> Simulator:
    return Simulator(
        config=CONFIG,
        universe_symbols=UNIVERSE,
        db_path=tmp_path / "state" / f"portfolio_{branch.replace('/', '_')}.db",
        log_root=log_root,
        branch=branch,
        resume=True,
    )


def _make_scheduler(tmp_path, log_root, clock, simulators, trader, **kwargs) -> AlphaLoopScheduler:
    kwargs.setdefault("next_bar_provider", lambda symbol, ts: {"open_time": ts + 1000, "open": 50_000.0})
    kwargs.setdefault("state_path", tmp_path / "state" / "scheduler_state.json")
    return AlphaLoopScheduler(
        config=CONFIG, clock=clock, simulators=simulators, trader=trader,
        log_root=log_root, **kwargs,
    )


@pytest.fixture
def ignite_paths(tmp_path, monkeypatch):
    """把 ignite.py 的模块级路径常量全部指到 tmp_path,测试绝不写生产目录。"""
    state_root = tmp_path / "state"
    log_root = tmp_path / "LOG"
    state_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ignite, "STATE_ROOT", state_root)
    monkeypatch.setattr(ignite, "LOG_ROOT", log_root)
    monkeypatch.setattr(ignite, "TOURNAMENT_ROSTER_PATH", state_root / "tactic_tournament_roster.json")
    monkeypatch.setattr(ignite, "MAIN_TACTICS_PATH", state_root / "main_program_tactics.json")
    monkeypatch.setattr(ignite, "MAIN_POLICY_PATH", state_root / "main_policy.json")
    return {"state_root": state_root, "log_root": log_root}


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, shell=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git -C {cwd} {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


@pytest.fixture
def tmp_repo(tmp_path):
    """tmp_path 下的一次性真实git仓库(参照 tests/test_git_merge_executor.py):
    main 分支带一个自身测试套件全绿的小package,供 GitMergeExecutor 在
    worktree 里真实跑 pytest。"""
    repo = tmp_path / "m8_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "pkg.py").write_text("def compute():\n    return 2\n", encoding="utf-8")
    (repo / "test_pkg.py").write_text(
        "from pkg import compute\n\n\ndef test_compute():\n    assert compute() == 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial commit on main")
    return repo


# ===========================================================================
# 1. 零LLM断言(§4.4第一条)
# ===========================================================================


def test_policy_branch_decides_through_scheduler_with_zero_llm_calls(tmp_path, log_root):
    """带policy_id的分支经 DispatchingTrader 走完整的
    scheduler.run_decision_cycle:决策产出自M7策略代码(momentum_v1),
    llm调用计数全程==0——这是M8存在的全部意义(前向决策零LLM)。"""
    policy_branch = "evo/20260715-momentum_v1"
    clock = FakeClock(BASE_TS)
    llm = CountingLLM()
    llm_trader = Trader(llm_client=llm, memory_store=FakeMemoryStore())

    # 明显上涨趋势:0.6%/bar复利,20bar动量约12%,远超momentum_v1的3%阈值。
    bars = _make_bars(40, base_price=50_000.0, drift_pct=0.006)
    dp = FakeDataPipeline({"BTC/USDT:USDT": bars})
    last_close = float(bars["close"].iloc[-1])

    dispatching = DispatchingTrader(
        llm_trader=llm_trader,
        policy_resolver=lambda b: "momentum_v1" if b == policy_branch else None,
        data_pipeline=dp,
    )

    sim = _make_sim(tmp_path, log_root, policy_branch)
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {policy_branch: sim}, dispatching,
        snapshot_provider=lambda ts: {"BTC/USDT:USDT": {"last": last_close}},
        next_bar_provider=lambda symbol, ts: {"open_time": ts + 1000, "open": last_close},
    )

    result = scheduler.run_decision_cycle(policy_branch)

    assert result["status"] == "decided"
    assert llm.call_count == 0  # 核心断言:纯代码路径,零LLM调用
    assert len(result["decisions"]) == 1
    decision = result["decisions"][0]
    assert decision.action == "open_long"  # 上涨趋势 -> momentum_v1顺势开多
    assert decision.branch == policy_branch  # branch归属由分发层盖章
    assert "动量" in decision.thesis or "K线" in decision.thesis  # 产自策略代码的文案
    assert decision.falsifier_condition is not None
    # 决策真的通过了Simulator的九步校验链(不是Rejection)
    assert result["executed"][0].__class__.__name__ == "Trade"
    # fetch_ohlcv 的 limit 带上了 REQUIRED_HISTORY_BARS + 缓冲
    assert dp.calls and dp.calls[0][2] >= 21


def test_branch_without_policy_id_falls_back_to_llm_path(tmp_path, log_root):
    """policy_resolver 返回None的分支原样委托给LLM Trader(计数==1),
    既有提示词分支(凉兮等)行为完全不受影响。"""
    clock = FakeClock(BASE_TS)
    llm = CountingLLM()
    llm_trader = Trader(llm_client=llm, memory_store=FakeMemoryStore())
    dispatching = DispatchingTrader(
        llm_trader=llm_trader,
        policy_resolver=lambda b: None,
        data_pipeline=FakeDataPipeline({}),
    )

    sim = _make_sim(tmp_path, log_root, "main")
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": sim}, dispatching,
        snapshot_provider=lambda ts: {"BTC/USDT:USDT": {"last": 50_000.0}},
    )

    result = scheduler.run_decision_cycle("main")

    assert result["status"] == "decided"
    assert llm.call_count == 1  # 恰好一次LLM调用,证明真的走了LLM路径
    assert result["decisions"][0].action == "hold"
    assert "LLM路径占位" in result["decisions"][0].thesis


def test_program_tactics_still_reaches_llm_trader_through_dispatcher(tmp_path, log_root):
    """回退路径必须全参数透传——program_tactics(锦标赛给纯提示词分支的
    战术文字)要能穿过 DispatchingTrader 抵达LLM prompt,老机制不受损。"""
    captured_prompts: list[str] = []

    def spying_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return CountingLLM()(prompt)

    llm_trader = Trader(llm_client=spying_llm, memory_store=FakeMemoryStore())
    dispatching = DispatchingTrader(
        llm_trader=llm_trader, policy_resolver=lambda b: None, data_pipeline=FakeDataPipeline({}),
    )
    decisions = dispatching.decide(
        ts=BASE_TS, positions=[], latest_snapshot={"BTC/USDT:USDT": {"last": 1.0}},
        program_tactics="UNIQUE_TACTICS_MARKER_m8_透传验证文本", branch="evo/prompt-style",
    )
    assert decisions[0].branch == "evo/prompt-style"
    assert any("UNIQUE_TACTICS_MARKER_m8_透传验证文本" in p for p in captured_prompts)


# ===========================================================================
# 2. 窗口拉长防误判(§4.4第二条)
# ===========================================================================


def _hourly_series(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return sorted(points)


def test_daily_tournament_does_not_promote_short_lived_spike(ignite_paths):
    """合成场景:分支前12小时偶然+1%(旧的小时级+4小时窗口逻辑会立刻判
    PROMOTE),但7天累计-2%;main 7天+1%。新的日级 evaluate_tactic_tournament
    (窗口168小时)必须不判它PROMOTE。"""
    branch = "evo/20260715-lucky-spike"
    created = BASE_TS
    now = created + 169 * HOUR_MS  # 刚过7天判定门槛

    # 分支小时级净值:0-11h 冲到 +1%,随后一路阴跌,7天末收于 -2%。
    branch_hourly = []
    for h in range(12):
        branch_hourly.append((created + h * HOUR_MS, 100.0 + h * (1.0 / 11)))  # ...101.0
    total_hours = 169
    for h in range(12, total_hours):
        frac = (h - 11) / (total_hours - 1 - 11)
        branch_hourly.append((created + h * HOUR_MS, 101.0 - frac * 3.0))  # 101 -> 98
    branch_hourly = _hourly_series(branch_hourly)

    # main小时级净值:7天线性 +1%。
    main_hourly = _hourly_series([
        (created + h * HOUR_MS, 100.0 + (h / (total_hours - 1)) * 1.0) for h in range(total_hours)
    ])

    # sanity(证明这个测试防的是真实风险):按旧口径,12小时窗口内分支+1%、
    # main约+0.065%,edge≈+0.93% > 0.5%的promote门槛——旧逻辑会判PROMOTE。
    early_branch = [nav for ts, nav in branch_hourly if ts <= created + 11 * HOUR_MS]
    early_main = [nav for ts, nav in main_hourly if ts <= created + 11 * HOUR_MS]
    early_edge = (early_branch[-1] / early_branch[0] - 1) * 100 - (early_main[-1] / early_main[0] - 1) * 100
    assert early_edge >= 0.5

    daily_branch = ignite._downsample_daily_last(branch_hourly)
    daily_main = ignite._downsample_daily_last(main_hourly)

    roster = {branch: {"tactics": "合成测试战术,不实际执行", "status": "active", "created_ms": created}}
    events = ignite.evaluate_tactic_tournament(
        roster, now, CONFIG["tactic_tournament"],
        daily_main_lookup=lambda: daily_main,
        daily_branch_lookup=lambda b: daily_branch if b == branch else [],
        hourly_branch_lookup=lambda b: branch_hourly if b == branch else [],
    )

    # 日级7天口径:分支-2% vs main+1%,edge=-3% —— 绝不能PROMOTE;
    # 回撤(101->98)约2.97% < 15%,也不该FAIL。正确结果是"无事件,继续观察"。
    assert events == []


def test_daily_tournament_still_promotes_genuine_sustained_edge(ignite_paths):
    """对照组:真实的持续优势(7天+4% vs main持平)在日级口径下照常PROMOTE,
    证明上面的防误判不是把晋升通道整个焊死。"""
    branch = "evo/20260715-genuine"
    created = BASE_TS
    now = created + 169 * HOUR_MS
    total_hours = 169

    branch_hourly = _hourly_series([
        (created + h * HOUR_MS, 100.0 + (h / (total_hours - 1)) * 4.0) for h in range(total_hours)
    ])
    main_hourly = _hourly_series([(created + h * HOUR_MS, 100.0) for h in range(total_hours)])

    roster = {branch: {"tactics": "合成测试战术,不实际执行", "status": "active", "created_ms": created}}
    events = ignite.evaluate_tactic_tournament(
        roster, now, CONFIG["tactic_tournament"],
        daily_main_lookup=lambda: ignite._downsample_daily_last(main_hourly),
        daily_branch_lookup=lambda b: ignite._downsample_daily_last(branch_hourly),
        hourly_branch_lookup=lambda b: branch_hourly,
    )
    assert len(events) == 1
    assert events[0]["decision"] == "PROMOTE"
    assert events[0]["branch"] == branch


def test_min_age_gate_blocks_judgment_before_seven_days(ignite_paths):
    """新的168小时门槛:不满7天的分支根本不进入评估,连'被误判'的机会都没有。"""
    branch = "evo/20260715-too-young"
    created = BASE_TS
    now = created + 12 * HOUR_MS  # 只活了12小时
    hourly = _hourly_series([(created + h * HOUR_MS, 100.0 + h * 0.1) for h in range(13)])

    roster = {branch: {"tactics": "合成测试战术,不实际执行", "status": "active", "created_ms": created}}
    events = ignite.evaluate_tactic_tournament(
        roster, now, CONFIG["tactic_tournament"],
        daily_main_lookup=lambda: ignite._downsample_daily_last(hourly),
        daily_branch_lookup=lambda b: ignite._downsample_daily_last(hourly),
        hourly_branch_lookup=lambda b: hourly,
    )
    assert events == []


def test_downsample_daily_last_keeps_last_point_per_utc_day():
    hourly = [
        (BASE_TS + 1 * HOUR_MS, 100.0),
        (BASE_TS + 23 * HOUR_MS, 105.0),  # day0 最后一个点
        (BASE_TS + DAY_MS + 2 * HOUR_MS, 99.0),  # day1 唯一点
    ]
    daily = ignite._downsample_daily_last(hourly)
    assert daily == [(BASE_TS, 105.0), (BASE_TS + DAY_MS, 99.0)]


def test_evaluate_cull_defaults_to_daily_series(ignite_paths):
    """末位斩杀的默认数据源也换成了日级(M8§4.2)——这里只验证接线:默认
    nav_series_lookup 是 read_branch_daily_nav_series。语义测试已由
    tests/test_ignite_tactic_generation.py 的注入式测试覆盖,不重复。"""
    import inspect

    src = inspect.getsource(ignite.evaluate_cull)
    assert "read_branch_daily_nav_series" in src


# ===========================================================================
# 3. 端到端:admit -> 锦标赛PROMOTE -> 真实judge -> 真实git merge
# ===========================================================================


def test_end_to_end_policy_promotion_through_real_git_merge(tmp_path, tmp_repo, ignite_paths, monkeypatch):
    """§4.4第三条 + §0.1诊断2的修复验证:这条链路上的每一环都必须是真的——
    EvolutionOrchestrator.judge() 真实产出PROMOTE(不再因为branch_navs交集
    过滤而恒为空),GitMergeExecutor.attempt_merge() 真实跑测试+真实merge
    (git log可见merge commit),两份LOCKED append-only日志有真实记录。"""
    monkeypatch.setattr(ignite, "PROJECT_ROOT", tmp_repo)
    log_root = ignite_paths["log_root"]

    clock = FakeClock(BASE_TS)
    policy_id = "momentum_v1"

    # --- 第1步:admit_policy_to_forward_pool(写名册 + 创建真实git分支)---
    roster, branch = ignite.admit_policy_to_forward_pool(policy_id, {}, clock.now_ms())
    assert branch.startswith("evo/") and branch.endswith(policy_id)
    assert roster[branch]["policy_id"] == policy_id
    assert roster[branch]["status"] == "active"
    # tactics = policy.DESCRIPTION + 生存规则后缀
    assert "动量战术" in roster[branch]["tactics"]
    assert "锦标赛生存规则" in roster[branch]["tactics"]
    # 名册立刻落盘(到monkeypatch后的tmp路径)
    on_disk = json.loads(ignite.TOURNAMENT_ROSTER_PATH.read_text(encoding="utf-8"))
    assert branch in on_disk
    # git分支真的存在
    assert branch in _git(tmp_repo, "branch", "--list", branch).stdout
    # 幂等:同名分支再admit一次直接拒绝(git侧已存在也不炸)
    with pytest.raises(ValueError):
        ignite.admit_policy_to_forward_pool(policy_id, roster, clock.now_ms())

    # 内环在该分支上产出真实的代码改动(否则merge --no-ff对同一commit是
    # "Already up to date",不会产生merge commit,断言无从谈起)。
    _git(tmp_repo, "checkout", branch)
    (tmp_repo / "policy_note.md").write_text(f"policy {policy_id} forward-pool notes\n", encoding="utf-8")
    _git(tmp_repo, "add", "-A")
    _git(tmp_repo, "commit", "-m", f"inner loop: land {policy_id} artifacts on {branch}")
    _git(tmp_repo, "checkout", "main")

    # --- 第2步:合成让它达标的日级净值(8天,+4% vs main持平)---
    created_ms = BASE_TS
    n_days = 9
    for d in range(n_days):
        ts = created_ms + d * DAY_MS + 12 * HOUR_MS
        log_writer.append_jsonl(
            "nav_intraday.jsonl", {"ts": ts, "nav_agent": 100.0, "nav_benchmark": 100.0, "nav_random": 100.0},
            root=log_root,
        )
        log_writer.append_jsonl(
            "nav_intraday_branches.jsonl", {"ts": ts, "branch": branch, "nav": 100.0 + d * 0.5},
            root=log_root,
        )
    # nav.tsv(scorer.daily_mark维护的官方日级三线历史,main持平)
    scorer = Scorer(CONFIG, log_root=log_root)
    for d in range(n_days):
        date = ignite._utc_date_str(created_ms + d * DAY_MS)
        scorer.daily_mark(nav_agent=100.0, nav_benchmark=100.0, nav_random=100.0, date=date)

    clock.set_ms(created_ms + 8 * DAY_MS + 13 * HOUR_MS)  # 存活满168小时
    now = clock.now_ms()

    # --- 第3步:锦标赛(日级默认lookup,读上面的tmp LOG)判PROMOTE ---
    events = ignite.evaluate_tactic_tournament(roster, now, CONFIG["tactic_tournament"])
    assert len(events) == 1
    assert events[0]["decision"] == "PROMOTE"
    assert events[0]["branch"] == branch

    # --- 第4步:promote_policy_branch 走真实链路 ---
    orchestrator = EvolutionOrchestrator(CONFIG, scorer=scorer, log_root=log_root)
    git_merge_executor = GitMergeExecutor(repo_path=tmp_repo, log_root=log_root)
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": _make_sim(tmp_path, log_root, "main")},
        trader=object(),  # 本测试不跑决策周期,trader不会被触碰
        scorer=scorer, evolution_orchestrator=orchestrator, git_merge_executor=git_merge_executor,
    )

    outcome = ignite.promote_policy_branch(
        branch=branch, policy_id=policy_id, roster=roster,
        evolution_orchestrator=orchestrator, scheduler=scheduler,
        today=ignite._utc_date_str(now),
    )

    # judge() 真实返回了 PROMOTE(不是被交集过滤成空)
    assert outcome["verdict"] == "PROMOTE"
    # GitMergeExecutor 真实执行了 merge
    assert outcome["merged"] is True
    assert scheduler.effective_main_branch == branch

    # git log 可见真实的 merge commit
    git_log = _git(tmp_repo, "log", "main", "--oneline").stdout
    assert f"PROMOTE {branch} into main" in git_log
    # merge真的带来了分支上的文件
    assert (tmp_repo / "policy_note.md").exists()

    # LOCKED append-only 日志有真实记录
    registrations = log_writer.read_jsonl("branch_registrations.jsonl", root=log_root)
    assert any(r["branch"] == branch for r in registrations)
    verdicts = log_writer.read_jsonl("ratchet_verdicts.jsonl", root=log_root)
    promote_records = [v for v in verdicts if v["branch"] == branch and v["decision"] == "PROMOTE"]
    assert len(promote_records) == 1
    merge_attempts = log_writer.read_jsonl("merge_attempts.jsonl", root=log_root)
    assert any(m["branch"] == branch and m["merged"] is True for m in merge_attempts)

    # main 的 policy 指向被更新:policy_resolver 对 "main" 从此返回该 policy_id
    assert ignite.load_main_policy_id() == policy_id
    # 人类可读的战术描述也同步更新为 policy.DESCRIPTION
    tactics_data = json.loads(ignite.MAIN_TACTICS_PATH.read_text(encoding="utf-8"))
    assert tactics_data["promoted_from"] == branch
    assert "动量战术" in tactics_data["tactics"]


def test_judge_can_veto_tournament_promotion_on_daily_data(tmp_path, tmp_repo, ignite_paths, monkeypatch):
    """两级晋升的"第二级真的有裁决权":锦标赛提名后,如果官方日级评分口径下
    edge不足(main也在涨),judge()返回ARCHIVE,merge根本不发生,main状态
    不被触碰。"""
    monkeypatch.setattr(ignite, "PROJECT_ROOT", tmp_repo)
    log_root = ignite_paths["log_root"]

    clock = FakeClock(BASE_TS)
    roster, branch = ignite.admit_policy_to_forward_pool("carry_v1", {}, clock.now_ms())

    scorer = Scorer(CONFIG, log_root=log_root)
    n_days = 9
    for d in range(n_days):
        ts = BASE_TS + d * DAY_MS + 12 * HOUR_MS
        # 官方nav.tsv口径:main每天+1,分支只略好一点点(edge < 0.5%)
        scorer.daily_mark(
            nav_agent=100.0 + d * 1.0, nav_benchmark=100.0, nav_random=100.0,
            date=ignite._utc_date_str(BASE_TS + d * DAY_MS),
        )
        log_writer.append_jsonl(
            "nav_intraday_branches.jsonl",
            {"ts": ts, "branch": branch, "nav": 100.0 + d * 1.01},
            root=log_root,
        )

    clock.set_ms(BASE_TS + 8 * DAY_MS + 13 * HOUR_MS)
    orchestrator = EvolutionOrchestrator(CONFIG, scorer=scorer, log_root=log_root)
    git_merge_executor = GitMergeExecutor(repo_path=tmp_repo, log_root=log_root)
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {"main": _make_sim(tmp_path, log_root, "main")},
        trader=object(), scorer=scorer,
        evolution_orchestrator=orchestrator, git_merge_executor=git_merge_executor,
    )

    outcome = ignite.promote_policy_branch(
        branch=branch, policy_id="carry_v1", roster=roster,
        evolution_orchestrator=orchestrator, scheduler=scheduler,
        today=ignite._utc_date_str(clock.now_ms()),
    )

    # 官方裁决口径:分支8天+8.08% vs main+8%,edge≈0.08% < 0.5% -> ARCHIVE
    assert outcome["verdict"] == "ARCHIVE"
    assert outcome["merged"] is False
    assert scheduler.effective_main_branch == "main"
    # main的policy/战术状态完全没有被触碰
    assert ignite.load_main_policy_id() is None
    assert not ignite.MAIN_TACTICS_PATH.exists()
    # git上没有任何merge commit
    git_log = _git(tmp_repo, "log", "main", "--oneline").stdout
    assert "PROMOTE" not in git_log


# ===========================================================================
# 4. monthly_report 新增"回测vs前向一致性"栏
# ===========================================================================


def _write_nav_tsv(log_root: Path) -> None:
    scorer = Scorer(CONFIG, log_root=log_root)
    scorer.daily_mark(nav_agent=100.0, nav_benchmark=100.0, nav_random=100.0, date="2026-06-01")
    scorer.daily_mark(nav_agent=105.0, nav_benchmark=102.0, nav_random=99.0, date="2026-06-30")


def test_monthly_report_backtest_forward_consistency_column(log_root):
    _write_nav_tsv(log_root)
    scorer = Scorer(CONFIG, log_root=log_root)
    pairs = [
        {"branch": "evo/20260715-momentum_v1", "backtest_holdout_edge_pct": 3.0, "forward_edge_pct": 1.2},
        {"branch": "evo/20260716-carry_v1", "backtest_holdout_edge_pct": 0.8, "forward_edge_pct": 1.0},
    ]
    report = scorer.monthly_report(backtest_forward_pairs=pairs)

    assert "Backtest vs Forward Consistency" in report
    # 差值 = forward - backtest,逐分支正确
    assert "| evo/20260715-momentum_v1 | 3.00% | 1.20% | -1.80% |" in report
    assert "| evo/20260716-carry_v1 | 0.80% | 1.00% | 0.20% |" in report
    # 平均gap = (-1.8 + 0.2) / 2 = -0.8
    assert "Average gap: -0.80% (n=2 branches)" in report
    # 体检性质写明,不是决策依据
    assert "体检指标" in report


def test_monthly_report_skips_consistency_column_when_no_pairs(log_root):
    _write_nav_tsv(log_root)
    scorer = Scorer(CONFIG, log_root=log_root)
    # 无记录:不传 / 传None / 传空列表,三种都必须优雅跳过、不报错
    for kwargs in ({}, {"backtest_forward_pairs": None}, {"backtest_forward_pairs": []}):
        report = scorer.monthly_report(**kwargs)
        assert "Backtest vs Forward Consistency" not in report
        assert "Monthly Report" in report  # 其余部分照常输出


# ===========================================================================
# 5. DispatchingTrader 安全行为
# ===========================================================================


def test_policy_with_insufficient_history_returns_empty_and_cycle_completes(tmp_path, log_root):
    """数据不足(K线根数 < REQUIRED_HISTORY_BARS)时策略按M7约定返回空列表,
    DispatchingTrader 不合成假hold,调度周期照常完成、不炸循环、零LLM。"""
    policy_branch = "evo/20260715-momentum_v1"
    clock = FakeClock(BASE_TS)
    llm = CountingLLM()
    llm_trader = Trader(llm_client=llm, memory_store=FakeMemoryStore())

    dp = FakeDataPipeline({"BTC/USDT:USDT": _make_bars(5, base_price=50_000.0, drift_pct=0.006)})
    dispatching = DispatchingTrader(
        llm_trader=llm_trader,
        policy_resolver=lambda b: "momentum_v1" if b == policy_branch else None,
        data_pipeline=dp,
    )

    sim = _make_sim(tmp_path, log_root, policy_branch)
    scheduler = _make_scheduler(
        tmp_path, log_root, clock, {policy_branch: sim}, dispatching,
        snapshot_provider=lambda ts: {"BTC/USDT:USDT": {"last": 50_000.0}},
    )

    result = scheduler.run_decision_cycle(policy_branch)

    assert result["status"] == "decided"
    assert result["decisions"] == []  # 空列表语义被保留,不合成假hold
    assert result["executed"] == []
    assert result["advice_path"] is None
    assert llm.call_count == 0  # 数据不足也不允许偷偷回退到LLM


def test_flat_market_policy_returns_empty_list(tmp_path, log_root):
    """横盘(动量低于阈值)同样返回空列表——'无信号'与'数据不足'是同一个
    安全语义:本周期不动仓位。"""
    dp = FakeDataPipeline({"BTC/USDT:USDT": _make_bars(40, base_price=50_000.0, drift_pct=0.0)})
    dispatching = DispatchingTrader(
        llm_trader=object(),  # 不该被触碰
        policy_resolver=lambda b: "momentum_v1",
        data_pipeline=dp,
    )
    decisions = dispatching.decide(
        ts=BASE_TS, positions=[], latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}},
    )
    assert decisions == []


def test_positions_accepted_as_list_or_dict():
    """main.py传的是 get_portfolio()['positions'](list[PerpPosition]),
    但dict形状也必须兼容——两种输入产生同一个 StrategyContext.positions。"""
    from LOCKED.schemas import PerpPosition

    pos = PerpPosition(symbol="BTC/USDT:USDT", side="long", notional=1000.0,
                       entry_price=50_000.0, margin=200.0, leverage=5)
    as_list = DispatchingTrader._positions_to_dict([pos])
    as_dict = DispatchingTrader._positions_to_dict({"BTC/USDT:USDT": pos})
    assert as_list == as_dict == {"BTC/USDT:USDT": pos}


def test_resolve_policy_id_for_branch_pure_function():
    roster = {
        "evo/a": {"policy_id": "momentum_v1", "status": "active"},
        "evo/b": {"status": "active"},  # 老式纯提示词分支,无policy_id字段
    }
    assert ignite.resolve_policy_id_for_branch("evo/a", roster) == "momentum_v1"
    assert ignite.resolve_policy_id_for_branch("evo/b", roster) is None
    assert ignite.resolve_policy_id_for_branch("evo/nonexistent", roster) is None


def test_policy_resolver_reads_main_policy_json_for_main_branch(ignite_paths):
    """make_policy_resolver 对 'main' 分支查 state/main_policy.json:晋升
    落盘后,main从下一个决策周期起自动走代码路径,不需要重启。"""
    clock = FakeClock(BASE_TS)
    resolver = ignite.make_policy_resolver(clock, {})
    assert resolver("main") is None  # 尚未有任何policy晋升
    ignite.save_main_policy_id("momentum_v1", "evo/20260715-momentum_v1", clock.now_ms())
    assert resolver("main") == "momentum_v1"  # 落盘立即生效,无需重启/重建resolver


# ===========================================================================
# M7资金费率感官前向注入:DispatchingTrader.decide() 里的 ctx.recent_funding
# ===========================================================================
#
# 下面几个helper都是离线fake,不发起任何真实网络请求。用一个"记录型"假
# policy模块(而不是真momentum_v1等种子策略)拦截decide()真正构造出来的
# ctx,这样能直接断言ctx.recent_funding的内容/缓存行为,而不需要通过某个
# 具体策略的交易逻辑去间接推断。


class _RecordingPolicyModule:
    """假冒 load_policy() 返回值的记录型policy:只记录传入的ctx,不产出任何
    决策,不实现真正的策略逻辑——本节测试只关心DispatchingTrader怎么构造
    ctx,不关心某个具体策略怎么用它。"""

    REQUIRED_HISTORY_BARS = 1
    DESCRIPTION = "recording stub policy for ctx.recent_funding injection tests"

    def __init__(self):
        self.seen_ctxs = []

    def decide(self, ctx):
        self.seen_ctxs.append(ctx)
        return []


class _FundingRecordingDataPipeline:
    """离线fake:fetch_ohlcv固定返回空K线(本节测试只关心funding注入,不
    关心K线内容),fetch_funding_rate_history记录每次调用的(symbol, since,
    limit),可选按symbol模拟拉取失败。"""

    def __init__(self, funding_by_symbol=None, raise_for=None):
        self.funding_by_symbol = funding_by_symbol or {}
        self.raise_for = raise_for or set()
        self.funding_calls: list[tuple[str, int, int]] = []

    def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000):
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        self.funding_calls.append((symbol, since, limit))
        if symbol in self.raise_for:
            raise RuntimeError(f"synthetic funding fetch failure for {symbol}")
        return self.funding_by_symbol.get(
            symbol, pd.DataFrame(columns=["timestamp", "funding_rate"])
        )


def _make_dispatcher_with_stub_policy(monkeypatch, dp, policy_module):
    """monkeypatch掉 ASSET.strategy.policy_trader.load_policy,让
    DispatchingTrader.decide() 拿到 _RecordingPolicyModule 而不是真的从磁盘
    加载种子策略——这样测试只关心ctx怎么被构造,不受某个具体策略实现变化
    影响。"""
    import ASSET.strategy.policy_trader as policy_trader_module

    monkeypatch.setattr(policy_trader_module, "load_policy", lambda policy_id: policy_module)
    return DispatchingTrader(
        llm_trader=None,  # policy_resolver永远返回policy_id,不会走到llm_trader
        policy_resolver=lambda branch: "stub_policy",
        data_pipeline=dp,
    )


def test_dispatching_trader_injects_recent_funding_into_ctx(monkeypatch):
    """基本注入验收:ctx.recent_funding里确实出现了data_pipeline返回的
    资金费率数据,且拉取参数(since/limit)符合"近30天"的设计。"""
    import ASSET.strategy.policy_trader as policy_trader_module

    funding_df = pd.DataFrame({"timestamp": [BASE_TS - 1000], "funding_rate": [0.0003]})
    dp = _FundingRecordingDataPipeline(funding_by_symbol={"BTC/USDT:USDT": funding_df})
    policy_module = _RecordingPolicyModule()
    dispatcher = _make_dispatcher_with_stub_policy(monkeypatch, dp, policy_module)

    dispatcher.decide(ts=BASE_TS, positions=[], latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}})

    assert len(policy_module.seen_ctxs) == 1
    ctx = policy_module.seen_ctxs[0]
    assert "BTC/USDT:USDT" in ctx.recent_funding
    pd.testing.assert_frame_equal(
        ctx.recent_funding["BTC/USDT:USDT"].reset_index(drop=True),
        funding_df.reset_index(drop=True),
    )
    assert dp.funding_calls == [
        (
            "BTC/USDT:USDT",
            BASE_TS - policy_trader_module._FUNDING_LOOKBACK_MS,
            policy_trader_module._FUNDING_FETCH_LIMIT,
        )
    ]


def test_dispatching_trader_funding_cache_hits_within_ten_minutes(monkeypatch):
    """同一个symbol在10分钟memo缓存窗口内只真正拉取一次;超过窗口后应该
    再拉一次——手法与 scripts/ignite.py::_trend_cache 相同但独立维护。"""
    dp = _FundingRecordingDataPipeline(
        funding_by_symbol={"BTC/USDT:USDT": pd.DataFrame(columns=["timestamp", "funding_rate"])}
    )
    policy_module = _RecordingPolicyModule()
    dispatcher = _make_dispatcher_with_stub_policy(monkeypatch, dp, policy_module)
    snapshot = {"BTC/USDT:USDT": {"last": 50_000.0}}

    dispatcher.decide(ts=BASE_TS, positions=[], latest_snapshot=snapshot)
    dispatcher.decide(ts=BASE_TS + 5 * 60_000, positions=[], latest_snapshot=snapshot)  # 5分钟后,仍在缓存窗口内
    assert len(dp.funding_calls) == 1, "expected the second call (5 minutes later) to hit the cache"

    dispatcher.decide(ts=BASE_TS + 11 * 60_000, positions=[], latest_snapshot=snapshot)  # 超过10分钟缓存窗口
    assert len(dp.funding_calls) == 2, "expected a fresh fetch once the 10-minute cache window has elapsed"


def test_dispatching_trader_funding_fetch_failure_yields_empty_dataframe_not_exception(monkeypatch):
    """data_pipeline.fetch_funding_rate_history 抛异常时,decide() 不应该
    跟着崩——funding是增强感官输入,不是决策周期能否完成的前提。"""
    dp = _FundingRecordingDataPipeline(raise_for={"BTC/USDT:USDT"})
    policy_module = _RecordingPolicyModule()
    dispatcher = _make_dispatcher_with_stub_policy(monkeypatch, dp, policy_module)

    result = dispatcher.decide(ts=BASE_TS, positions=[], latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}})

    assert result == []  # 记录型policy本来就不产出决策,但更重要的是没有异常往外抛
    ctx = policy_module.seen_ctxs[0]
    fdf = ctx.recent_funding["BTC/USDT:USDT"]
    assert fdf.empty
    assert list(fdf.columns) == ["timestamp", "funding_rate"]


def test_dispatching_trader_handles_data_pipeline_without_funding_method(monkeypatch):
    """向后兼容:data_pipeline 是本文件既有的 FakeDataPipeline(定义在本文件
    顶部,只实现了fetch_ohlcv,压根没有fetch_funding_rate_history方法)——
    这是本次升级前所有既有测试都在用的替身,升级后必须继续零改动可用。"""
    bars = _make_bars(5, base_price=50_000.0, drift_pct=0.0)
    dp = FakeDataPipeline({"BTC/USDT:USDT": bars})
    policy_module = _RecordingPolicyModule()
    dispatcher = _make_dispatcher_with_stub_policy(monkeypatch, dp, policy_module)

    dispatcher.decide(ts=BASE_TS, positions=[], latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}})

    ctx = policy_module.seen_ctxs[0]
    fdf = ctx.recent_funding["BTC/USDT:USDT"]
    assert fdf.empty
    assert list(fdf.columns) == ["timestamp", "funding_rate"]
