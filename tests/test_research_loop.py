"""
tests/test_research_loop.py —— M7 内环研究循环(scripts/research_loop.py)单测。

全部离线、确定性:
  - LLM 一律mock(见 make_llm_sequence),不真调任何API。
  - git 相关测试永远在 tmp_path 下 `git init` 出来的临时仓库里操作,绝不
    触碰本仓库本身(见 init_repo helper)。
  - BacktestEngine 在需要"跑一轮完整实验"的测试里被替换成 FakeBacktestEngine
    (返回预先构造好的合成 BacktestResult),不依赖真实历史数据/真实回测耗时。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.research_loop as rl
from LOCKED.backtest_engine import BacktestResult, BacktestWindow

STEP_MS = 4 * 3_600_000
BASE_TS = 1_700_000_000_000 - (1_700_000_000_000 % STEP_MS)

GOOD_POLICY_SOURCE = '''"""test-only policy: always returns an empty decision list."""
from __future__ import annotations

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "test policy: always returns empty list"


def decide(ctx: StrategyContext) -> list[Decision]:
    return []
'''

BAD_POLICY_SOURCE = '''"""test-only policy: violates the wall-clock ban (datetime.now())."""
from __future__ import annotations

import datetime

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "test policy: reads the wall clock, must be rejected by lint"


def decide(ctx: StrategyContext) -> list[Decision]:
    _ = datetime.datetime.now()
    return []
'''


def make_llm_sequence(responses):
    responses = list(responses)
    state = {"i": 0}

    def _client(prompt: str) -> str:
        idx = state["i"]
        state["i"] += 1
        return responses[idx]

    return _client


def _run_git(args, cwd):
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


def init_repo(tmp_path: Path) -> Path:
    """在 tmp_path 下建一个独立的、包含最小 ASSET/strategy 目录骨架的临时git
    仓库,并预置一个种子策略文件(供"改进型 vs 新建型"参照、以及冷启动
    seed_variant 想法来源使用)。全程只操作 tmp_path,绝不碰本仓库。"""
    repo = tmp_path / "repo"
    policies_dir = repo / "ASSET" / "strategy" / "policies"
    experiments_dir = repo / "ASSET" / "strategy" / "experiments"
    policies_dir.mkdir(parents=True)
    experiments_dir.mkdir(parents=True)
    (policies_dir / "aggressive_v1.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")

    _run_git(["init"], repo)
    _run_git(["config", "user.email", "test@example.com"], repo)
    _run_git(["config", "user.name", "test"], repo)
    _run_git(["add", "."], repo)
    _run_git(["commit", "-m", "init"], repo)
    return repo


def _git_log_subjects(repo: Path) -> list[str]:
    result = _run_git(["log", "--format=%H %s"], repo)
    return [line for line in result.stdout.strip().splitlines() if line]


class FakeBacktestEngine:
    """替身 BacktestEngine:run() 返回测试预先注入的合成结果,score() 复刻
    LOCKED.backtest_engine.BacktestEngine.score() 的公式(取最差验证窗口
    edge - 0.5*最差回撤,任意验证窗口爆仓则 -inf),不依赖真实历史数据/真实
    Simulator 回放,让"一轮完整实验"的测试保持快速且确定。"""

    injected_results: dict = {}

    def __init__(self, config, data_pipeline, scratch_root):
        self.config = config
        self.data_pipeline = data_pipeline
        self.scratch_root = scratch_root

    def run(self, strategy_fn, symbols, windows, experiment_id):
        return FakeBacktestEngine.injected_results

    def score(self, results):
        val = [r for label, r in results.items() if label != "train" and not r.window.is_holdout]
        if any(r.branch_dead for r in val):
            return float("-inf")
        edges = [r.edge_vs_benchmark_pct for r in val]
        return min(edges) - 0.5 * max(r.max_drawdown_pct for r in val)


def _make_result(
    label: str, edge_pct: float, max_dd_pct: float = 0.0, is_holdout: bool = False, trade_count: int = 6
) -> BacktestResult:
    """trade_count 默认给6——两个验证窗口(val_1/val_2)各6笔、总和12,高于
    min_val_trades默认门槛(10),这样"kept应该是kept"的既有测试不会被任务2
    新增的最小样本量门槛意外撞车(该门槛的专门测试会显式传trade_count覆盖
    这个默认值)。"""
    window = BacktestWindow(label=label, start_ts=BASE_TS, end_ts=BASE_TS + 10 * STEP_MS, is_holdout=is_holdout)
    return BacktestResult(
        window=window,
        nav_series=[(BASE_TS, 100.0)],
        benchmark_nav_series=[(BASE_TS, 100.0)],
        return_pct=edge_pct,
        max_drawdown_pct=max_dd_pct,
        edge_vs_benchmark_pct=edge_pct,
        branch_dead=False,
        trade_count=trade_count,
        rejection_count=0,
    )


def make_ctx(repo: Path, deep_llm) -> rl.LoopContext:
    experiments_dir = repo / "ASSET" / "strategy" / "experiments"
    return rl.LoopContext(
        config={"backtest": {"max_experiments_per_night": 50}},
        repo_path=repo,
        policies_dir=repo / "ASSET" / "strategy" / "policies",
        experiments_dir=experiments_dir,
        ledger_path=experiments_dir / "ledger.jsonl",
        protocols_dir=experiments_dir / "protocols",
        notes_dir=repo / "ASSET" / "research_notes",  # 不存在也没关系,冷启动走seed_variant
        deep_llm=deep_llm,
        data_pipeline=None,
        symbols=["BTC/USDT:USDT"],
        data_end_ts=BASE_TS,
        scratch_root=experiments_dir,
    )


# ---------------------------------------------------------------------------
# 1. build_default_windows:首尾相接、holdout最新、train最长
# ---------------------------------------------------------------------------


def test_build_default_windows_contiguous_holdout_newest_train_longest():
    config = {
        "backtest": {
            "windows": {"train_days": 365, "val_window_days": 90, "val_window_count": 2, "holdout_days": 90},
        }
    }
    data_end_ts = 2_000_000_000_000
    windows = rl.build_default_windows(config, data_end_ts)

    by_label = {w.label: w for w in windows}
    assert set(by_label) == {"train", "val_1", "val_2", "holdout"}

    train, val1, val2, holdout = by_label["train"], by_label["val_1"], by_label["val_2"], by_label["holdout"]

    # 首尾相接,不重叠不留缝隙
    assert train.end_ts == val1.start_ts
    assert val1.end_ts == val2.start_ts
    assert val2.end_ts == holdout.start_ts
    assert holdout.end_ts == data_end_ts

    # holdout 是时间上最新的一段,且唯一 is_holdout=True
    assert holdout.is_holdout is True
    assert train.is_holdout is False
    assert val1.is_holdout is False
    assert val2.is_holdout is False

    # train 跨度最长(365天 > 90天)
    assert (train.end_ts - train.start_ts) > (val1.end_ts - val1.start_ts)
    assert (train.end_ts - train.start_ts) > (holdout.end_ts - holdout.start_ts)


# ---------------------------------------------------------------------------
# 2. keep/revert 判定:结构性保证holdout不在签名里 + 分数更好才keep
# ---------------------------------------------------------------------------


def test_decide_keep_or_revert_signature_has_no_holdout_or_results_param():
    import inspect

    params = list(inspect.signature(rl.decide_keep_or_revert).parameters)
    assert params == ["new_score", "previous_best_score"]
    assert "results" not in params
    assert "holdout" not in " ".join(params).lower()


def test_decide_keep_or_revert_keeps_only_when_strictly_better():
    assert rl.decide_keep_or_revert(new_score=1.0, previous_best_score=0.0) == "kept"
    assert rl.decide_keep_or_revert(new_score=0.0, previous_best_score=0.0) == "reverted"  # 打平不算赢
    assert rl.decide_keep_or_revert(new_score=-1.0, previous_best_score=0.0) == "reverted"
    assert rl.decide_keep_or_revert(new_score=5.0, previous_best_score=3.0) == "kept"


def test_historical_best_score_defaults_to_zero_and_ignores_non_kept():
    ledger = [
        {"policy_id": "foo_v2", "status": "reverted", "val_edge_vs_benchmark_pct": 999.0, "val_max_drawdown_pct": 0.0},
        {"policy_id": "foo_v2", "status": "kept", "val_edge_vs_benchmark_pct": 2.0, "val_max_drawdown_pct": 1.0},
        {"policy_id": "bar_v2", "status": "kept", "val_edge_vs_benchmark_pct": 100.0, "val_max_drawdown_pct": 0.0},
    ]
    # 只看policy_id="foo_v2"且status=kept的那一条:2.0 - 0.5*1.0 = 1.5
    assert rl.historical_best_score(ledger, "foo_v2") == pytest.approx(1.5)
    # 从未出现过的policy_id -> 首轮与0比
    assert rl.historical_best_score(ledger, "never_seen") == 0.0


def test_compute_val_stats_excludes_train_and_holdout():
    results = {
        "train": _make_result("train", edge_pct=999.0),
        "val_1": _make_result("val_1", edge_pct=5.0, max_dd_pct=1.0),
        "val_2": _make_result("val_2", edge_pct=-2.0, max_dd_pct=4.0),
        "holdout": _make_result("holdout", edge_pct=-9999.0, max_dd_pct=50.0, is_holdout=True),
    }
    edge, dd = rl.compute_val_stats(results)
    assert edge == pytest.approx(-2.0)  # 最差的验证窗口edge,不含train/holdout
    assert dd == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 3. git 协议顺序:protocol commit 早于 result commit;revert时结果commit不存在、文件被恢复
# ---------------------------------------------------------------------------


def test_full_round_kept_commits_protocol_then_result_in_order(tmp_path, monkeypatch):
    repo = init_repo(tmp_path)
    monkeypatch.setattr(rl, "BacktestEngine", FakeBacktestEngine)
    FakeBacktestEngine.injected_results = {
        "train": _make_result("train", edge_pct=999.0),
        "val_1": _make_result("val_1", edge_pct=5.0, max_dd_pct=1.0),
        "val_2": _make_result("val_2", edge_pct=3.0, max_dd_pct=1.0),
        "holdout": _make_result("holdout", edge_pct=-50.0, is_holdout=True),
    }
    llm = make_llm_sequence([GOOD_POLICY_SOURCE])
    ctx = make_ctx(repo, llm)

    record = rl.run_single_experiment(ctx, experiment_index=0, dry_run=False, print_fn=lambda s: None)

    assert record["status"] == "kept"
    assert record["commit_sha_protocol"] is not None
    assert record["commit_sha_result"] is not None
    # holdout只记录,不参与判定分数,但确实原样写进了ledger
    assert record["holdout_edge_vs_benchmark_pct"] == pytest.approx(-50.0)

    subjects = _git_log_subjects(repo)  # 最新的在最前面 (git log 默认降序)
    protocol_idx = next(i for i, line in enumerate(subjects) if line.startswith(f"{record['commit_sha_protocol']} "))
    result_idx = next(i for i, line in enumerate(subjects) if line.startswith(f"{record['commit_sha_result']} "))
    assert result_idx < protocol_idx  # result commit更晚 -> 在log里排更前面

    assert any("research(protocol):" in line for line in subjects)
    assert any("research(results):" in line for line in subjects)

    policy_path = ctx.policies_dir / f"{record['policy_id']}.py"
    assert policy_path.exists()
    assert policy_path.read_text(encoding="utf-8") == GOOD_POLICY_SOURCE

    # ledger.jsonl 真的落盘了这条记录
    ledger_records = rl.read_ledger(ctx.ledger_path)
    assert len(ledger_records) == 1
    assert ledger_records[0]["experiment_id"] == record["experiment_id"]


def test_full_round_records_holdout_trade_count_and_return_pct_for_zero_trade_holdout(tmp_path, monkeypatch):
    """修"holdout躺赢"缺陷的核心验收:holdout窗口零交易(trade_count=0)时,
    ledger记录里的holdout_trade_count必须真实反映0,holdout_return_pct必须
    是holdout窗口的绝对收益(return_pct),而不是edge_vs_benchmark_pct——
    这两个字段是下游识别"这条holdout_edge是不是纯粹的基准镜像"的依据。"""
    repo = init_repo(tmp_path)
    monkeypatch.setattr(rl, "BacktestEngine", FakeBacktestEngine)
    FakeBacktestEngine.injected_results = {
        "train": _make_result("train", edge_pct=999.0),
        "val_1": _make_result("val_1", edge_pct=5.0, max_dd_pct=1.0),
        "val_2": _make_result("val_2", edge_pct=3.0, max_dd_pct=1.0),
        "holdout": _make_result("holdout", edge_pct=12.4177, is_holdout=True, trade_count=0),
    }
    llm = make_llm_sequence([GOOD_POLICY_SOURCE])
    ctx = make_ctx(repo, llm)

    record = rl.run_single_experiment(ctx, experiment_index=0, dry_run=False, print_fn=lambda s: None)

    assert record["holdout_trade_count"] == 0
    # _make_result 把 return_pct 也设成了 edge_pct(见helper注释,单测里两者
    # 同值),这里断言的是"确实取自return_pct这个字段"而不是巧合——用一个
    # edge与return不同的场景在下面的build_ledger_record单测里单独覆盖。
    assert record["holdout_return_pct"] == pytest.approx(12.4177)


def test_build_ledger_record_holdout_return_pct_is_absolute_return_not_edge():
    """holdout_return_pct 必须来自 BacktestResult.return_pct(绝对收益),
    不是 edge_vs_benchmark_pct——用两者不同值的场景直接验证不会被搞混。"""
    idea = rl.Idea(
        source="seed_variant", policy_id="momentum_v2", parent_policy_id="momentum_v1",
        hypothesis="test hypothesis", reference_policy_id="momentum_v1",
    )
    record = rl.build_ledger_record(
        experiment_id="momentum_v2_0000", idea=idea,
        commit_sha_protocol="aaa", commit_sha_result="bbb", status="kept",
        val_edge_vs_benchmark_pct=1.0, val_max_drawdown_pct=0.5,
        holdout_edge_vs_benchmark_pct=12.4177,  # 与真实BTC跌14%场景一致的镜像edge
        wall_time_seconds=3.0,
        holdout_trade_count=0, holdout_return_pct=-2.3,  # 策略自己的绝对收益,与edge不同
    )
    assert record["holdout_trade_count"] == 0
    assert record["holdout_return_pct"] == pytest.approx(-2.3)
    assert record["holdout_edge_vs_benchmark_pct"] == pytest.approx(12.4177)


def test_full_round_reverted_has_no_result_commit_and_file_restored(tmp_path, monkeypatch):
    repo = init_repo(tmp_path)
    monkeypatch.setattr(rl, "BacktestEngine", FakeBacktestEngine)
    FakeBacktestEngine.injected_results = {
        "train": _make_result("train", edge_pct=999.0),
        "val_1": _make_result("val_1", edge_pct=-5.0, max_dd_pct=1.0),
        "val_2": _make_result("val_2", edge_pct=-8.0, max_dd_pct=2.0),
        "holdout": _make_result("holdout", edge_pct=1.0, is_holdout=True),
    }
    llm = make_llm_sequence([GOOD_POLICY_SOURCE])
    ctx = make_ctx(repo, llm)

    record = rl.run_single_experiment(ctx, experiment_index=0, dry_run=False, print_fn=lambda s: None)

    assert record["status"] == "reverted"
    assert record["commit_sha_protocol"] is not None
    assert record["commit_sha_result"] is None
    # 样本量本身是够的(默认trade_count=6*2=12>=min_val_trades),真正的
    # revert原因是分数没打赢历史最优,不是任务2的样本量门槛
    assert record["revert_reason"] == "score_not_better"

    subjects = _git_log_subjects(repo)
    assert not any("research(results):" in line for line in subjects)  # 没有结果commit
    assert any("research(protocol):" in line for line in subjects)

    # 这是一个从未被提交过的新文件(policy_id 由 _next_policy_variant_id 生成,
    # 目标文件此前不存在),revert = 删除,而不是"恢复到上一版本"(没有上一版本)。
    policy_path = ctx.policies_dir / f"{record['policy_id']}.py"
    assert not policy_path.exists()


def test_git_revert_file_restores_previously_committed_content(tmp_path):
    """针对"改进型"(目标文件此前已被提交过)的revert路径单测:git checkout --
    应该精确恢复到上一次提交的内容,而不是删除文件。"""
    repo = init_repo(tmp_path)
    tracked_path = repo / "ASSET" / "strategy" / "policies" / "aggressive_v1.py"
    original_content = tracked_path.read_text(encoding="utf-8")

    tracked_path.write_text("this content should be discarded by revert", encoding="utf-8")
    rl.git_revert_file(repo, tracked_path)

    assert tracked_path.read_text(encoding="utf-8") == original_content


# ---------------------------------------------------------------------------
# 4. lint失败重写:第一次违规、第二次通过 -> 重试发生且最终kept
# ---------------------------------------------------------------------------


def test_generate_and_lint_policy_retries_after_lint_failure_then_succeeds():
    llm = make_llm_sequence([BAD_POLICY_SOURCE, GOOD_POLICY_SOURCE])
    source, attempts_log = rl.generate_and_lint_policy(llm, "irrelevant prompt", max_attempts=3)

    assert source == GOOD_POLICY_SOURCE
    assert len(attempts_log) == 2
    assert attempts_log[0] != []  # 第一次真的违规了(datetime.now())
    assert attempts_log[1] == []  # 第二次通过


def test_generate_and_lint_policy_gives_up_after_max_attempts_all_bad():
    llm = make_llm_sequence([BAD_POLICY_SOURCE, BAD_POLICY_SOURCE, BAD_POLICY_SOURCE])
    source, attempts_log = rl.generate_and_lint_policy(llm, "irrelevant prompt", max_attempts=3)

    assert source is None
    assert len(attempts_log) == 3
    assert all(v != [] for v in attempts_log)


def test_full_round_lint_failed_status_when_llm_never_produces_clean_source(tmp_path, monkeypatch):
    repo = init_repo(tmp_path)
    monkeypatch.setattr(rl, "BacktestEngine", FakeBacktestEngine)
    llm = make_llm_sequence([BAD_POLICY_SOURCE, BAD_POLICY_SOURCE, BAD_POLICY_SOURCE])
    ctx = make_ctx(repo, llm)

    record = rl.run_single_experiment(ctx, experiment_index=0, dry_run=False, print_fn=lambda s: None)

    assert record["status"] == "lint_failed"
    assert record["commit_sha_protocol"] is not None
    assert record["commit_sha_result"] is None
    # lint失败时文件从未落盘,策略目录里不应该出现半成品文件
    policy_path = ctx.policies_dir / f"{record['policy_id']}.py"
    assert not policy_path.exists()


def test_full_round_kept_after_one_lint_retry(tmp_path, monkeypatch):
    repo = init_repo(tmp_path)
    monkeypatch.setattr(rl, "BacktestEngine", FakeBacktestEngine)
    FakeBacktestEngine.injected_results = {
        "train": _make_result("train", edge_pct=999.0),
        "val_1": _make_result("val_1", edge_pct=5.0, max_dd_pct=0.0),
        "val_2": _make_result("val_2", edge_pct=4.0, max_dd_pct=0.0),
        "holdout": _make_result("holdout", edge_pct=1.0, is_holdout=True),
    }
    llm = make_llm_sequence([BAD_POLICY_SOURCE, GOOD_POLICY_SOURCE])
    ctx = make_ctx(repo, llm)

    record = rl.run_single_experiment(ctx, experiment_index=0, dry_run=False, print_fn=lambda s: None)

    assert record["status"] == "kept"
    policy_path = ctx.policies_dir / f"{record['policy_id']}.py"
    assert policy_path.read_text(encoding="utf-8") == GOOD_POLICY_SOURCE


# ---------------------------------------------------------------------------
# 5. 路径安全:policy_id="../evil" 被拒绝
# ---------------------------------------------------------------------------


def test_validate_policy_id_rejects_path_traversal():
    with pytest.raises(ValueError):
        rl.validate_policy_id("../evil")


def test_safe_policy_path_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        rl.safe_policy_path(tmp_path, "../evil")


def test_safe_policy_path_rejects_other_unsafe_ids(tmp_path):
    for bad_id in ["../../etc/passwd", "Aggressive", "a", "a b", "a/b", "1abc", "__init__"]:
        with pytest.raises(ValueError):
            rl.safe_policy_path(tmp_path, bad_id)


def test_safe_policy_path_accepts_valid_id(tmp_path):
    path = rl.safe_policy_path(tmp_path, "momentum_v2")
    assert path == (tmp_path / "momentum_v2.py").resolve()


# ---------------------------------------------------------------------------
# 6. webui /api/experiments 端点
# ---------------------------------------------------------------------------


@pytest.fixture
def webui_client(tmp_path, monkeypatch):
    import webui.app as webui_app
    from fastapi.testclient import TestClient

    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(webui_app, "EXPERIMENTS_LEDGER_PATH", ledger_path)
    return TestClient(webui_app.app), ledger_path


def test_experiments_endpoint_has_no_data_when_ledger_missing(webui_client):
    client, _ledger_path = webui_client
    data = client.get("/api/experiments").json()
    assert data["has_data"] is False
    assert data["experiments"] == []


def test_experiments_endpoint_returns_synthetic_ledger_records(webui_client):
    client, ledger_path = webui_client
    records = [
        {
            "experiment_id": "momentum_v2_0001", "policy_id": "momentum_v2", "parent_policy_id": "momentum_v1",
            "hypothesis": "test hypothesis 1", "commit_sha_protocol": "aaa", "commit_sha_result": "bbb",
            "status": "kept", "val_edge_vs_benchmark_pct": 3.5, "val_max_drawdown_pct": 1.0,
            "holdout_edge_vs_benchmark_pct": 2.0, "wall_time_seconds": 1.23,
        },
        {
            "experiment_id": "momentum_v3_0000", "policy_id": "momentum_v3", "parent_policy_id": "momentum_v2",
            "hypothesis": "test hypothesis 2", "commit_sha_protocol": "ccc", "commit_sha_result": None,
            "status": "reverted", "val_edge_vs_benchmark_pct": -1.0, "val_max_drawdown_pct": 2.0,
            "holdout_edge_vs_benchmark_pct": None, "wall_time_seconds": 0.5,
        },
    ]
    with open(ledger_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    data = client.get("/api/experiments").json()
    assert data["has_data"] is True
    assert len(data["experiments"]) == 2
    # 按experiment_id排序
    assert [e["experiment_id"] for e in data["experiments"]] == ["momentum_v2_0001", "momentum_v3_0000"]
    assert data["experiments"][0]["status"] == "kept"


# ---------------------------------------------------------------------------
# 7. ledger schema:每行记录包含全部必需字段
# ---------------------------------------------------------------------------


REQUIRED_LEDGER_FIELDS = {
    "experiment_id", "policy_id", "parent_policy_id", "hypothesis",
    "commit_sha_protocol", "commit_sha_result", "status",
    "val_edge_vs_benchmark_pct", "val_max_drawdown_pct",
    "holdout_edge_vs_benchmark_pct", "wall_time_seconds",
    "revert_reason", "strategy_class",  # 向后兼容字段(此前一次改动新增)
    "holdout_trade_count", "holdout_return_pct",  # 本次改动新增:修"holdout躺赢"缺陷
}


def test_build_ledger_record_has_all_required_fields():
    idea = rl.Idea(
        source="seed_variant", policy_id="momentum_v2", parent_policy_id="momentum_v1",
        hypothesis="test hypothesis", reference_policy_id="momentum_v1",
    )
    record = rl.build_ledger_record(
        experiment_id="momentum_v2_0000", idea=idea,
        commit_sha_protocol="aaa", commit_sha_result="bbb", status="kept",
        val_edge_vs_benchmark_pct=1.0, val_max_drawdown_pct=0.5,
        holdout_edge_vs_benchmark_pct=2.0, wall_time_seconds=3.0,
    )
    assert set(record.keys()) == REQUIRED_LEDGER_FIELDS


def test_ledger_round_trip_preserves_all_required_fields(tmp_path):
    idea = rl.Idea(
        source="research_note", policy_id="research_idea_v1", parent_policy_id=None,
        hypothesis="another test hypothesis", reference_policy_id="diversified_v1",
    )
    record = rl.build_ledger_record(
        experiment_id="research_idea_v1_0000", idea=idea,
        commit_sha_protocol="aaa", commit_sha_result=None, status="reverted",
        val_edge_vs_benchmark_pct=-1.0, val_max_drawdown_pct=2.0,
        holdout_edge_vs_benchmark_pct=None, wall_time_seconds=0.1,
    )
    ledger_path = tmp_path / "ledger.jsonl"
    rl.append_ledger(ledger_path, record)
    loaded = rl.read_ledger(ledger_path)
    assert len(loaded) == 1
    assert REQUIRED_LEDGER_FIELDS.issubset(set(loaded[0].keys()))
    assert loaded[0] == record


# ---------------------------------------------------------------------------
# 8. 想法挑选:冷启动从种子策略开始;轮转顺序
# ---------------------------------------------------------------------------


def test_select_idea_source_cold_start_forces_seed_variant():
    assert rl.select_idea_source(experiment_index=0, ledger_has_kept=False) == "seed_variant"
    assert rl.select_idea_source(experiment_index=1, ledger_has_kept=False) == "seed_variant"
    assert rl.select_idea_source(experiment_index=2, ledger_has_kept=False) == "seed_variant"


def test_select_idea_source_rotates_when_ledger_has_kept_entries():
    assert rl.select_idea_source(experiment_index=0, ledger_has_kept=True) == "kept_variant"
    assert rl.select_idea_source(experiment_index=1, ledger_has_kept=True) == "research_note"
    assert rl.select_idea_source(experiment_index=2, ledger_has_kept=True) == "seed_variant"
    assert rl.select_idea_source(experiment_index=3, ledger_has_kept=True) == "kept_variant"


def test_select_idea_cold_start_picks_a_fresh_seed_variant_id(tmp_path):
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    (policies_dir / "momentum_v1.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    idea = rl.select_idea(0, ledger_entries=[], policies_dir=policies_dir, notes_dir=tmp_path / "notes")
    assert idea.source == "seed_variant"
    assert idea.parent_policy_id in rl._SEED_POLICY_IDS
    assert not (policies_dir / f"{idea.policy_id}.py").exists()  # 目标是一个全新文件名


def test_next_policy_variant_id_avoids_collisions_and_strips_existing_suffix(tmp_path):
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    (policies_dir / "momentum_v1.py").write_text("x", encoding="utf-8")
    (policies_dir / "momentum_v2.py").write_text("x", encoding="utf-8")
    candidate = rl._next_policy_variant_id("momentum_v1", policies_dir)
    assert candidate == "momentum_v3"
    rl.validate_policy_id(candidate)  # 必须本身也是合法policy_id


# ---------------------------------------------------------------------------
# 9. 任务2:最小样本量门槛(min_val_trades)——总交易笔数不够,一律revert,
#    不管分数多好,revert_reason 区分"样本不够"和"分数没打赢"
# ---------------------------------------------------------------------------


def test_min_val_trades_gate_reverts_even_with_good_score(tmp_path, monkeypatch):
    repo = init_repo(tmp_path)
    monkeypatch.setattr(rl, "BacktestEngine", FakeBacktestEngine)
    FakeBacktestEngine.injected_results = {
        "train": _make_result("train", edge_pct=999.0, trade_count=0),
        "val_1": _make_result("val_1", edge_pct=5.0, max_dd_pct=0.0, trade_count=2),
        "val_2": _make_result("val_2", edge_pct=4.0, max_dd_pct=0.0, trade_count=1),
        "holdout": _make_result("holdout", edge_pct=1.0, is_holdout=True, trade_count=0),
    }
    llm = make_llm_sequence([GOOD_POLICY_SOURCE])
    ctx = make_ctx(repo, llm)

    record = rl.run_single_experiment(ctx, experiment_index=0, dry_run=False, print_fn=lambda s: None)

    # 分数其实是正的、严格好于0的历史最优基线,换成任务2之前的老逻辑本该kept——
    # 但两个验证窗口的trade_count总和只有3(<min_val_trades默认10),一律revert
    assert record["status"] == "reverted"
    assert record["revert_reason"] == "insufficient_trades"
    assert record["commit_sha_result"] is None
    policy_path = ctx.policies_dir / f"{record['policy_id']}.py"
    assert not policy_path.exists()  # 从未提交过的新文件,revert=删除


def test_min_val_trades_gate_allows_keep_when_sample_size_sufficient(tmp_path, monkeypatch):
    repo = init_repo(tmp_path)
    monkeypatch.setattr(rl, "BacktestEngine", FakeBacktestEngine)
    FakeBacktestEngine.injected_results = {
        "train": _make_result("train", edge_pct=999.0, trade_count=0),
        "val_1": _make_result("val_1", edge_pct=5.0, max_dd_pct=0.0, trade_count=8),
        "val_2": _make_result("val_2", edge_pct=4.0, max_dd_pct=0.0, trade_count=7),
        "holdout": _make_result("holdout", edge_pct=1.0, is_holdout=True, trade_count=0),
    }
    llm = make_llm_sequence([GOOD_POLICY_SOURCE])
    ctx = make_ctx(repo, llm)

    record = rl.run_single_experiment(ctx, experiment_index=0, dry_run=False, print_fn=lambda s: None)

    # trade_count总和=15(>=10)且分数(4.0,底线是min(5,4)-0*0.5=4.0)严格好于
    # 历史最优0.0 -> kept,revert_reason为None(kept时该字段规定为null)
    assert record["status"] == "kept"
    assert record["revert_reason"] is None
    assert record["commit_sha_result"] is not None


# ---------------------------------------------------------------------------
# 10. 任务1:外环研究总监——mock deep_llm,校验落盘/重试耗尽后返回None
# ---------------------------------------------------------------------------


def test_run_research_director_writes_memo_and_archive_on_valid_response(tmp_path):
    repo = init_repo(tmp_path)
    valid_response = json.dumps(
        {
            "trajectory_insights": [
                "目前只有aggressive这一个真实实现的策略类别,轨迹样本还很少",
            ],
            "directions": [
                {"class": "funding_rate_carry", "hypothesis": "资金费率carry方向的具体假设", "rationale": "从未出现过的类别"},
                {"class": "pairs_trading", "hypothesis": "配对交易方向的具体假设", "rationale": "分散化"},
                {"class": "aggressive", "hypothesis": "在aggressive基础上进一步调参", "rationale": "复用已验证类别"},
            ],
        },
        ensure_ascii=False,
    )
    llm = make_llm_sequence([valid_response])
    ctx = make_ctx(repo, llm)

    memo = rl.run_research_director(ctx, print_fn=lambda s: None)

    assert memo is not None
    assert len(memo["directions"]) == 3
    assert memo["ts"] == ctx.data_end_ts  # ts复用ctx.data_end_ts,不读墙钟

    memo_path = ctx.experiments_dir / "direction_memo.json"
    md_path = ctx.experiments_dir / "direction_memo.md"
    archive_path = ctx.experiments_dir / "memos" / f"memo_{ctx.data_end_ts}.json"
    assert memo_path.exists()
    assert md_path.exists()
    assert archive_path.exists()

    on_disk = json.loads(memo_path.read_text(encoding="utf-8"))
    assert on_disk == memo
    assert json.loads(archive_path.read_text(encoding="utf-8")) == memo
    assert "funding_rate_carry" in md_path.read_text(encoding="utf-8")


def test_run_research_director_returns_none_after_repeated_invalid_responses(tmp_path):
    repo = init_repo(tmp_path)
    llm = make_llm_sequence(["not json at all", "{}", "still not valid json"])
    ctx = make_ctx(repo, llm)
    printed: list[str] = []

    memo = rl.run_research_director(ctx, print_fn=printed.append)

    assert memo is None
    assert not (ctx.experiments_dir / "direction_memo.json").exists()
    assert printed  # 打印了失败信息告知调用方,但没有抛异常


def test_build_director_prompt_contains_trajectory_and_death_archive_and_classes():
    ledger_entries = [
        {"policy_id": "momentum_v2", "hypothesis": "追涨杀跌", "status": "kept",
         "val_edge_vs_benchmark_pct": 3.0, "val_max_drawdown_pct": 1.0, "holdout_edge_vs_benchmark_pct": 2.0},
    ]
    death_events = [
        {"branch": "evo/20260701-carry", "decision": "FAIL", "edge_vs_main_pct": -2.0,
         "max_drawdown_pct": 18.0, "reason": "drawdown blew through fail line", "ts": 123},
    ]
    prompt = rl.build_director_prompt(
        ledger_entries, death_events, {"momentum_v1": "动量战术"}, known_classes=["momentum"],
    )
    assert "momentum_v2" in prompt
    assert "evo/20260701-carry" in prompt
    assert "momentum" in prompt
    assert "directions" in prompt


def test_build_director_prompt_warns_about_zero_trade_holdout_being_a_mirror_score():
    """修"holdout躺赢"缺陷:prompt里必须包含holdout_trade_count/
    holdout_return_pct这两个字段,以及提醒总监"holdout_trade_count=0的策略
    其holdout edge只是基准涨跌镜像"的那句话——否则总监会被8个零交易策略
    "完全相同的高edge"这种假象误导,认为它们都是有效策略。"""
    ledger_entries = [
        {"policy_id": "aggressive_v1", "hypothesis": "零交易躺赢", "status": "reverted",
         "val_edge_vs_benchmark_pct": -1.0, "val_max_drawdown_pct": 0.0,
         "holdout_edge_vs_benchmark_pct": 12.4177, "holdout_trade_count": 0, "holdout_return_pct": 0.0},
    ]
    prompt = rl.build_director_prompt(
        ledger_entries, [], {"aggressive_v1": "激进战术"}, known_classes=["aggressive"],
    )
    assert "holdout_trade_count" in prompt
    assert "holdout_return_pct" in prompt
    assert "holdout_trade_count=0" in prompt
    assert "镜像" in prompt


# ---------------------------------------------------------------------------
# 11. 任务1:想法来源接线——总监备忘录消费 + 新颖性槽位
# ---------------------------------------------------------------------------


def test_select_idea_prefers_director_memo_when_present(tmp_path):
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    (policies_dir / "momentum_v1.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    memo = {
        "trajectory_insights": ["..."],
        "directions": [
            {"class": "funding_rate_carry", "hypothesis": "假设A", "rationale": "r"},
            {"class": "pairs_trading", "hypothesis": "假设B", "rationale": "r"},
        ],
        "ts": 123,
    }
    (experiments_dir / "direction_memo.json").write_text(json.dumps(memo), encoding="utf-8")

    idea0 = rl.select_idea(
        0, ledger_entries=[], policies_dir=policies_dir, notes_dir=tmp_path / "notes", experiments_dir=experiments_dir,
    )
    assert idea0.source == "director"
    assert idea0.hypothesis == "假设A"
    assert idea0.strategy_class == "funding_rate_carry"

    idea1 = rl.select_idea(
        1, ledger_entries=[], policies_dir=policies_dir, notes_dir=tmp_path / "notes", experiments_dir=experiments_dir,
    )
    assert idea1.hypothesis == "假设B"  # 轮转:experiment_index % len(directions)


def test_select_idea_falls_back_to_old_rotation_when_memo_absent(tmp_path):
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    (policies_dir / "momentum_v1.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    # experiments_dir 存在但没有 direction_memo.json -> 视同不存在,不应该报错
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()

    idea = rl.select_idea(
        0, ledger_entries=[], policies_dir=policies_dir, notes_dir=tmp_path / "notes", experiments_dir=experiments_dir,
    )
    assert idea.source == "seed_variant"  # 冷启动老逻辑不受影响


def test_select_idea_forces_novelty_every_fifth_experiment_regardless_of_memo(tmp_path):
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    (policies_dir / "momentum_v1.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    memo = {"directions": [{"class": "funding_rate_carry", "hypothesis": "假设A", "rationale": "r"}], "ts": 1}
    (experiments_dir / "direction_memo.json").write_text(json.dumps(memo), encoding="utf-8")

    for index in (4, 9, 14):  # experiment_index % 5 == 4
        idea = rl.select_idea(
            index, ledger_entries=[], policies_dir=policies_dir, notes_dir=tmp_path / "notes",
            experiments_dir=experiments_dir,
        )
        assert idea.source == "novelty"  # 新颖性槽位凌驾于备忘录方向之上

    idea_non_slot = rl.select_idea(
        0, ledger_entries=[], policies_dir=policies_dir, notes_dir=tmp_path / "notes", experiments_dir=experiments_dir,
    )
    assert idea_non_slot.source == "director"  # 非第5个实验时正常消费备忘录


def test_build_researcher_prompt_novelty_form_includes_known_classes():
    idea = rl.Idea(
        source="novelty", policy_id="novelty_idea_v1", parent_policy_id=None,
        hypothesis=rl._NOVELTY_HYPOTHESIS, reference_policy_id="diversified_v1", strategy_class=None,
    )
    prompt = rl.build_researcher_prompt(
        idea, "novelty_idea_v1", "new", "reference source here", known_classes=["momentum", "carry"],
    )
    assert "新颖性槽位" in prompt
    assert "momentum" in prompt and "carry" in prompt  # 已存在类别清单必须出现在prompt里
    assert "配对/价差回归" in prompt  # 经典类别菜单(仅作启发)


def test_build_researcher_prompt_director_form_includes_target_class():
    idea = rl.Idea(
        source="director", policy_id="director_idea_v1", parent_policy_id=None,
        hypothesis="假设文本", reference_policy_id="diversified_v1", strategy_class="funding_rate_carry",
    )
    prompt = rl.build_researcher_prompt(idea, "director_idea_v1", "new", "reference source here")
    assert "研究总监指定方向" in prompt
    assert "funding_rate_carry" in prompt


# ---------------------------------------------------------------------------
# 12. run_research_loop 收尾自动调研究总监一次;dry_run 时跳过
# ---------------------------------------------------------------------------


def test_run_research_loop_calls_research_director_once_when_not_dry_run(tmp_path, monkeypatch):
    repo = init_repo(tmp_path)
    monkeypatch.setattr(rl, "BacktestEngine", FakeBacktestEngine)
    FakeBacktestEngine.injected_results = {
        "train": _make_result("train", edge_pct=999.0),
        "val_1": _make_result("val_1", edge_pct=5.0, max_dd_pct=0.0),
        "val_2": _make_result("val_2", edge_pct=4.0, max_dd_pct=0.0),
        "holdout": _make_result("holdout", edge_pct=1.0, is_holdout=True),
    }
    calls = []
    monkeypatch.setattr(rl, "run_research_director", lambda ctx, print_fn=print: calls.append(ctx))
    llm = make_llm_sequence([GOOD_POLICY_SOURCE, GOOD_POLICY_SOURCE])
    ctx = make_ctx(repo, llm)

    rl.run_research_loop(ctx, max_experiments=2, dry_run=False, print_fn=lambda s: None)

    assert len(calls) == 1


def test_run_research_loop_skips_research_director_in_dry_run(tmp_path, monkeypatch):
    repo = init_repo(tmp_path)
    calls = []
    monkeypatch.setattr(rl, "run_research_director", lambda ctx, print_fn=print: calls.append(ctx))
    llm = make_llm_sequence([GOOD_POLICY_SOURCE, GOOD_POLICY_SOURCE])
    ctx = make_ctx(repo, llm)

    rl.run_research_loop(ctx, max_experiments=2, dry_run=True, print_fn=lambda s: None)

    assert calls == []
