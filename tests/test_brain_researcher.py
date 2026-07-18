"""
tests/test_brain_researcher.py —— scripts/brain_researcher.py 单测。

全部离线、确定性,与 tests/test_research_loop.py 同一套隔离纪律:
  - LLM 一律mock(make_llm_sequence),不真调任何API。
  - 回测引擎(BacktestEngine)/行情管道(DataPipeline)/数据终点推导
    (determine_data_end_ts)在需要"跑一次回测关卡"的测试里整体monkeypatch
    掉,不依赖真实历史数据/真实网络/真实data_cache缓存文件。
  - 所有磁盘写入(policies/LOG)全部落在 tmp_path 下,绝不触碰本仓库真实的
    ASSET/strategy/policies 或 LOG 目录。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.brain_researcher as br
from LOCKED.backtest_engine import BacktestResult, BacktestWindow
from LOCKED.log_writer import read_jsonl

STEP_MS = 4 * 3_600_000
BASE_TS = 1_700_000_000_000 - (1_700_000_000_000 % STEP_MS)
HOUR_MS = 3_600_000

GOOD_POLICY_SOURCE = '''"""test-only incumbent policy: always returns an empty decision list."""
from __future__ import annotations

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "incumbent test policy: always empty"


def decide(ctx: StrategyContext) -> list[Decision]:
    return []
'''

CANDIDATE_POLICY_SOURCE = '''"""test-only candidate policy: always returns an empty decision list."""
from __future__ import annotations

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "candidate test policy: always empty"


def decide(ctx: StrategyContext) -> list[Decision]:
    return []
'''

BAD_POLICY_SOURCE = '''"""test-only policy: violates the wall-clock ban (datetime.now())."""
from __future__ import annotations

import datetime

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "bad policy: reads the wall clock, must be rejected by lint"


def decide(ctx: StrategyContext) -> list[Decision]:
    _ = datetime.datetime.now()
    return []
'''


def make_llm_sequence(responses):
    """依次返回预设响应的假LLM callable,响应耗尽后抛IndexError(测试如果
    意外多调用一次会立刻暴露,而不是静默返回None导致更难定位的失败)。"""
    responses = list(responses)
    state = {"i": 0}

    def _client(prompt: str) -> str:
        idx = state["i"]
        state["i"] += 1
        return responses[idx]

    return _client


class FakeMemoryStore:
    """记录每次write()调用的假记忆库,不接触真实sqlite文件。"""

    def __init__(self):
        self.calls: list[dict] = []

    def write(self, content, ts, layer, branch=None, importance=1.0):
        self.calls.append({"content": content, "ts": ts, "layer": layer, "branch": branch})


class FakeDataPipeline:
    """回测关卡里注入的假行情管道,不做任何真实IO/网络请求。"""

    def __init__(self, exchange_id):
        self.exchange_id = exchange_id


class FakeGateEngine:
    """替身 BacktestEngine:run() 按 experiment_id 查表返回预先注入的合成
    结果,score() 复刻 LOCKED.backtest_engine.BacktestEngine.score() 的公式
    (与 tests/test_research_loop.py::FakeBacktestEngine 同一手法)。"""

    injected: dict[str, dict] = {}

    def __init__(self, config, data_pipeline, scratch_root):
        self.config = config
        self.data_pipeline = data_pipeline
        self.scratch_root = scratch_root

    def run(self, strategy_fn, symbols, windows, experiment_id):
        return FakeGateEngine.injected[experiment_id]

    def score(self, results):
        val = [r for label, r in results.items() if label != "train" and not r.window.is_holdout]
        if any(r.branch_dead for r in val):
            return float("-inf")
        edges = [r.edge_vs_benchmark_pct for r in val]
        return min(edges) - 0.5 * max(r.max_drawdown_pct for r in val)


def _make_result(
    label: str, edge_pct: float, max_dd_pct: float = 0.0, is_holdout: bool = False, trade_count: int = 6
) -> BacktestResult:
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


def make_config(**overrides):
    cfg = {
        "backtest": {
            "scratch_root": "ASSET/strategy/experiments",
            "windows": {"train_days": 365, "val_window_days": 90, "val_window_count": 2, "holdout_days": 90},
            "min_val_trades": 10,
        },
        "data": {"exchange": "okx", "timeframe": "4h"},
        "quant_derby": {"proposal_cooldown_hours": 4},
    }
    cfg.update(overrides)
    return cfg


def make_env(tmp_path: Path):
    """project_root就是tmp_path本身——policies_dir=tmp_path/ASSET/strategy/policies,
    与 brain_researcher._infer_project_root(policies_dir) 的parents[2]算法对齐。"""
    policies_dir = tmp_path / "ASSET" / "strategy" / "policies"
    policies_dir.mkdir(parents=True)
    log_root = tmp_path / "LOG"
    return policies_dir, log_root


BRANCH = "evo/20260719-quant-ark-kimi"


def base_meta(policy_id="legacy_carry_v3", provider="ARK-Kimi"):
    return {"llm_provider": provider, "policy_id": policy_id, "created_ms": BASE_TS, "status": "active"}


# ---------------------------------------------------------------------------
# 1. keep 路径:写日志 + 写记忆,不触发回测关卡
# ---------------------------------------------------------------------------


def test_keep_path_writes_journal_log_and_memory(tmp_path):
    policies_dir, log_root = make_env(tmp_path)
    (policies_dir / "legacy_carry_v3.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    meta = base_meta()
    llm = make_llm_sequence(
        [json.dumps({"journal": "本小时净值平稳,继续观察资金费率信号。", "action": "keep", "policy_code": ""})]
    )
    mem = FakeMemoryStore()
    now_ms = BASE_TS + STEP_MS

    result = br.run_brain_review(
        BRANCH, meta, llm, now_ms,
        config=make_config(), memory_store=mem,
        nav_series_lookup=lambda b: [(BASE_TS, 100.0), (now_ms, 101.5)],
        market_context="BTC横盘整理,资金费率转负。",
        log_root=log_root, policies_dir=policies_dir, state={},
    )

    assert result["proposed_policy_id"] is None
    assert result["gate_result"] is None
    assert "净值平稳" in result["journal"]
    assert result["state"]["last_review_ms"] == now_ms
    assert "version_counter" not in result["state"] or result["state"].get("version_counter", 0) == 0

    journal_records = read_jsonl("brain_journals.jsonl", root=log_root)
    assert len(journal_records) == 1
    rec = journal_records[0]
    assert rec["branch"] == BRANCH
    assert rec["provider"] == "ARK-Kimi"
    assert rec["action"] == "keep"
    assert rec["policy_id"] == "legacy_carry_v3"
    assert "净值平稳" in rec["journal"]

    assert len(mem.calls) == 1
    assert mem.calls[0]["layer"] == "L2"
    assert mem.calls[0]["branch"] == BRANCH
    assert mem.calls[0]["ts"] == now_ms

    # 没有任何策略候选文件被写出
    assert sorted(p.name for p in policies_dir.glob("*.py")) == ["legacy_carry_v3.py"]
    assert not (log_root / "policy_gate.jsonl").exists()


# ---------------------------------------------------------------------------
# 2. propose -> lint失败 -> 不上线,但文件留档
# ---------------------------------------------------------------------------


def test_propose_lint_failure_is_not_shipped_but_file_kept_on_disk(tmp_path):
    policies_dir, log_root = make_env(tmp_path)
    (policies_dir / "legacy_carry_v3.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    meta = base_meta()
    llm = make_llm_sequence(
        [json.dumps({"journal": "尝试引入墙钟抖动信号。", "action": "propose", "policy_code": BAD_POLICY_SOURCE})]
    )
    mem = FakeMemoryStore()
    now_ms = BASE_TS + STEP_MS

    result = br.run_brain_review(
        BRANCH, meta, llm, now_ms,
        config=make_config(), memory_store=mem,
        nav_series_lookup=lambda b: [(BASE_TS, 100.0)],
        market_context="(无)",
        log_root=log_root, policies_dir=policies_dir, state={"version_counter": 0},
    )

    assert result["proposed_policy_id"] is None
    assert result["gate_result"]["verdict"] == "lint_failed"
    assert result["gate_result"]["violations"]  # 非空违规清单
    assert result["state"]["version_counter"] == 1  # 版本号已经消耗,不会被下次复用

    candidate_path = policies_dir / "ark_kimi_v1.py"
    assert candidate_path.exists()  # "策略文件保留(留档)"——不删除
    assert candidate_path.read_text(encoding="utf-8") == BAD_POLICY_SOURCE

    gate_records = read_jsonl("policy_gate.jsonl", root=log_root)
    assert len(gate_records) == 1
    assert gate_records[0]["verdict"] == "lint_failed"
    assert gate_records[0]["candidate_id"] == "ark_kimi_v1"
    assert gate_records[0]["incumbent_id"] == "legacy_carry_v3"

    journal_records = read_jsonl("brain_journals.jsonl", root=log_root)
    assert journal_records[0]["action"] == "propose"
    assert "静态审查" in journal_records[0]["journal"]


# ---------------------------------------------------------------------------
# 3/4. propose -> 回测关卡:分数不如现任 -> rejected;分数更优 -> accepted
# ---------------------------------------------------------------------------


def _run_propose_with_fake_gate(tmp_path, monkeypatch, candidate_edge, incumbent_edge, min_val_trades=10, trade_count=6):
    policies_dir, log_root = make_env(tmp_path)
    (policies_dir / "legacy_carry_v3.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    meta = base_meta()
    llm = make_llm_sequence(
        [json.dumps({"journal": "提交一版新策略。", "action": "propose", "policy_code": CANDIDATE_POLICY_SOURCE})]
    )
    mem = FakeMemoryStore()
    now_ms = BASE_TS + STEP_MS

    FakeGateEngine.injected = {
        "gate_ark_kimi_v1": {
            "train": _make_result("train", edge_pct=999.0),
            "val_1": _make_result("val_1", edge_pct=candidate_edge, max_dd_pct=1.0, trade_count=trade_count),
            "val_2": _make_result("val_2", edge_pct=candidate_edge, max_dd_pct=1.0, trade_count=trade_count),
            "holdout": _make_result("holdout", edge_pct=0.0, is_holdout=True),
        },
        "gate_incumbent_legacy_carry_v3": {
            "train": _make_result("train", edge_pct=999.0),
            "val_1": _make_result("val_1", edge_pct=incumbent_edge, max_dd_pct=1.0),
            "val_2": _make_result("val_2", edge_pct=incumbent_edge, max_dd_pct=1.0),
            "holdout": _make_result("holdout", edge_pct=0.0, is_holdout=True),
        },
    }
    monkeypatch.setattr(br, "BacktestEngine", FakeGateEngine)
    monkeypatch.setattr(br, "DataPipeline", FakeDataPipeline)
    monkeypatch.setattr(br, "determine_data_end_ts", lambda config, cache_dir: BASE_TS)

    result = br.run_brain_review(
        BRANCH, meta, llm, now_ms,
        config=make_config(),
        memory_store=mem,
        nav_series_lookup=lambda b: [(BASE_TS, 100.0)],
        market_context="(无)",
        log_root=log_root, policies_dir=policies_dir, state={"version_counter": 0},
    )
    return result, policies_dir, log_root


def test_propose_rejected_when_candidate_score_not_better_than_incumbent(tmp_path, monkeypatch):
    result, policies_dir, log_root = _run_propose_with_fake_gate(
        tmp_path, monkeypatch, candidate_edge=1.0, incumbent_edge=5.0
    )

    assert result["proposed_policy_id"] is None
    assert result["gate_result"]["verdict"] == "rejected"
    assert result["gate_result"]["reason"] == "score_not_better"
    assert result["gate_result"]["candidate_score"] < result["gate_result"]["incumbent_score"]

    # 文件留档,但incumbent的policy_id并未被顶替(调用方接线时不会看到proposed_policy_id)
    assert (policies_dir / "ark_kimi_v1.py").exists()
    assert (policies_dir / "legacy_carry_v3.py").exists()

    gate_records = read_jsonl("policy_gate.jsonl", root=log_root)
    assert gate_records[0]["verdict"] == "rejected"


def test_propose_accepted_when_candidate_score_strictly_better_than_incumbent(tmp_path, monkeypatch):
    result, policies_dir, log_root = _run_propose_with_fake_gate(
        tmp_path, monkeypatch, candidate_edge=9.0, incumbent_edge=1.0
    )

    assert result["proposed_policy_id"] == "ark_kimi_v1"
    assert result["gate_result"]["verdict"] == "accepted"
    assert result["gate_result"]["candidate_score"] > result["gate_result"]["incumbent_score"]
    assert (policies_dir / "ark_kimi_v1.py").read_text(encoding="utf-8") == CANDIDATE_POLICY_SOURCE

    gate_records = read_jsonl("policy_gate.jsonl", root=log_root)
    assert gate_records[0]["verdict"] == "accepted"
    assert gate_records[0]["candidate_id"] == "ark_kimi_v1"
    assert gate_records[0]["incumbent_id"] == "legacy_carry_v3"


def test_propose_rejected_when_tied_score_is_not_a_win(tmp_path, monkeypatch):
    """平手=不换——decide_keep_or_revert的棘轮纪律必须原样透传到本模块的
    回测关卡判定。"""
    result, _, _ = _run_propose_with_fake_gate(tmp_path, monkeypatch, candidate_edge=3.0, incumbent_edge=3.0)
    assert result["gate_result"]["verdict"] == "rejected"
    assert result["proposed_policy_id"] is None


def test_propose_rejected_when_val_trade_count_below_min_threshold(tmp_path, monkeypatch):
    """即使候选分数远高于现任,验证窗口总交易笔数不足门槛也一律rejected——
    与research_loop.min_val_trades同一条纪律(防止2-3笔交易靠运气拿高分)。"""
    result, _, _ = _run_propose_with_fake_gate(
        tmp_path, monkeypatch, candidate_edge=100.0, incumbent_edge=1.0, trade_count=2
    )
    assert result["gate_result"]["verdict"] == "rejected"
    assert result["gate_result"]["reason"] == "insufficient_trades"
    assert result["proposed_policy_id"] is None


# ---------------------------------------------------------------------------
# 5. 提案冷却:冷却期内propose被搁置,按keep处理,不消耗version_counter
# ---------------------------------------------------------------------------


def test_propose_during_cooldown_is_shelved_as_keep(tmp_path):
    policies_dir, log_root = make_env(tmp_path)
    (policies_dir / "legacy_carry_v3.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    meta = base_meta()
    llm = make_llm_sequence(
        [json.dumps({"journal": "还是想改进一下。", "action": "propose", "policy_code": CANDIDATE_POLICY_SOURCE})]
    )
    mem = FakeMemoryStore()
    last_proposal_ms = BASE_TS
    now_ms = BASE_TS + 2 * HOUR_MS  # 冷却期4小时,2小时后仍在冷却中

    result = br.run_brain_review(
        BRANCH, meta, llm, now_ms,
        config=make_config(), memory_store=mem,
        nav_series_lookup=lambda b: [(BASE_TS, 100.0)],
        market_context="(无)",
        log_root=log_root, policies_dir=policies_dir,
        state={"last_proposal_ms": last_proposal_ms, "version_counter": 1},
    )

    assert result["proposed_policy_id"] is None
    assert result["gate_result"] is None
    assert "冷却" in result["journal"]
    # 冷却期内被搁置的提案不消耗version_counter/不刷新last_proposal_ms
    assert result["state"]["version_counter"] == 1
    assert result["state"]["last_proposal_ms"] == last_proposal_ms
    assert result["state"]["last_review_ms"] == now_ms

    journal_records = read_jsonl("brain_journals.jsonl", root=log_root)
    assert journal_records[0]["action"] == "keep"
    assert not any(policies_dir.glob("ark_kimi_*.py"))
    assert not (log_root / "policy_gate.jsonl").exists()


# ---------------------------------------------------------------------------
# 6. LLM输出坏JSON:重试后仍失败 -> 降级为keep
# ---------------------------------------------------------------------------


def test_bad_json_output_retries_then_degrades_to_keep(tmp_path):
    policies_dir, log_root = make_env(tmp_path)
    (policies_dir / "legacy_carry_v3.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    meta = base_meta()
    llm = make_llm_sequence(["not json at all", "{\"journal\": \"缺少action字段\"}", "still not json"])
    mem = FakeMemoryStore()
    now_ms = BASE_TS + STEP_MS

    result = br.run_brain_review(
        BRANCH, meta, llm, now_ms,
        config=make_config(), memory_store=mem,
        nav_series_lookup=lambda b: [(BASE_TS, 100.0)],
        market_context="(无)",
        log_root=log_root, policies_dir=policies_dir, state={},
    )

    assert result["proposed_policy_id"] is None
    assert result["gate_result"] is None
    assert "解析失败" in result["journal"]

    journal_records = read_jsonl("brain_journals.jsonl", root=log_root)
    assert journal_records[0]["action"] == "keep"
    assert len(mem.calls) == 1  # 降级路径依然写记忆,留痕


# ---------------------------------------------------------------------------
# 7. 异常不外抛:LLM调用本身抛异常 / 净值查询抛异常
# ---------------------------------------------------------------------------


def test_llm_raising_exception_does_not_propagate(tmp_path):
    policies_dir, log_root = make_env(tmp_path)
    (policies_dir / "legacy_carry_v3.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    meta = base_meta()

    def _raising_llm(prompt: str) -> str:
        raise RuntimeError("API超时")

    now_ms = BASE_TS + STEP_MS
    result = br.run_brain_review(
        BRANCH, meta, _raising_llm, now_ms,
        config=make_config(), memory_store=FakeMemoryStore(),
        nav_series_lookup=lambda b: [(BASE_TS, 100.0)],
        market_context="(无)",
        log_root=log_root, policies_dir=policies_dir, state={},
    )
    assert result["proposed_policy_id"] is None
    assert result["gate_result"] is None
    assert "解析失败" in result["journal"] or "异常" in result["journal"]


def test_nav_series_lookup_raising_exception_does_not_propagate(tmp_path):
    policies_dir, log_root = make_env(tmp_path)
    (policies_dir / "legacy_carry_v3.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    meta = base_meta()
    llm = make_llm_sequence([json.dumps({"journal": "净值查询挂了也要能跑完。", "action": "keep", "policy_code": ""})])

    def _raising_lookup(branch: str):
        raise ConnectionError("nav服务不可达")

    now_ms = BASE_TS + STEP_MS
    result = br.run_brain_review(
        BRANCH, meta, llm, now_ms,
        config=make_config(), memory_store=FakeMemoryStore(),
        nav_series_lookup=_raising_lookup,
        market_context="(无)",
        log_root=log_root, policies_dir=policies_dir, state={},
    )
    assert result["gate_result"] is None
    assert result["proposed_policy_id"] is None
    assert "净值查询挂了也要能跑完" in result["journal"]


def test_backtest_gate_exception_falls_back_to_keep_like_gate_error(tmp_path, monkeypatch):
    policies_dir, log_root = make_env(tmp_path)
    (policies_dir / "legacy_carry_v3.py").write_text(GOOD_POLICY_SOURCE, encoding="utf-8")
    meta = base_meta()
    llm = make_llm_sequence(
        [json.dumps({"journal": "提交新版本。", "action": "propose", "policy_code": CANDIDATE_POLICY_SOURCE})]
    )

    def _boom(config, cache_dir):
        raise FileNotFoundError("data_cache未就绪")

    monkeypatch.setattr(br, "determine_data_end_ts", _boom)
    now_ms = BASE_TS + STEP_MS

    result = br.run_brain_review(
        BRANCH, meta, llm, now_ms,
        config=make_config(), memory_store=FakeMemoryStore(),
        nav_series_lookup=lambda b: [(BASE_TS, 100.0)],
        market_context="(无)",
        log_root=log_root, policies_dir=policies_dir, state={"version_counter": 0},
    )
    assert result["proposed_policy_id"] is None
    assert result["gate_result"]["verdict"] == "error"
    # 候选文件依然落盘留档(先写文件后跑回测是既定顺序)
    assert (policies_dir / "ark_kimi_v1.py").exists()


# ---------------------------------------------------------------------------
# 8. 纯函数小工具
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider, expected_prefix",
    [
        ("ARK-Kimi", "ark_kimi"),
        ("deepseek-v3", "deepseek_v3"),
        ("智谱GLM", "glm"),  # 非ASCII字符被清洗掉,只留下拉丁字母部分
        ("###", "b_"),  # 清洗后为空/不以字母开头 -> 补"b_"前缀兜底
        (None, "brain"),
    ],
)
def test_provider_slug_produces_valid_policy_id_prefix(provider, expected_prefix):
    slug = br._provider_slug(provider)
    assert slug.startswith(expected_prefix) or slug == "brain"
    # 结果本身要能通过 policy_id 白名单校验(拼上_v1后缀)
    from scripts.research_loop import validate_policy_id

    validate_policy_id(f"{slug}_v1")


def test_next_candidate_policy_id_skips_existing_files(tmp_path):
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    (policies_dir / "ark_kimi_v1.py").write_text("x", encoding="utf-8")
    (policies_dir / "ark_kimi_v2.py").write_text("x", encoding="utf-8")

    policy_id, new_counter = br._next_candidate_policy_id("ark_kimi", version_counter=0, policies_dir=policies_dir)
    assert policy_id == "ark_kimi_v3"
    assert new_counter == 3


def test_cooldown_status_pure_function():
    in_cd, remaining = br._cooldown_status({"last_proposal_ms": 0}, now_ms=1 * HOUR_MS, cooldown_hours=4)
    assert in_cd is True
    assert remaining == pytest.approx(3.0)

    in_cd2, remaining2 = br._cooldown_status({"last_proposal_ms": 0}, now_ms=5 * HOUR_MS, cooldown_hours=4)
    assert in_cd2 is False
    assert remaining2 == 0.0

    in_cd3, _ = br._cooldown_status({}, now_ms=999, cooldown_hours=4)
    assert in_cd3 is False  # 从未提案过 -> 不在冷却期


# ---------------------------------------------------------------------------
# 9. 真实小窗口冒烟(标记slow,默认跳过)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "真实冒烟需要项目根目录下真实的data_cache OHLCV parquet缓存"
        "(scripts/backfill_history.py产出)与真实DataPipeline联网/离线缓存,"
        "跑一次完整train+val*2+holdout四窗口回测在CI/常规单测环境下过慢"
        "(分钟级)。本文件其余用例已经用FakeGateEngine覆盖了run_policy_gate"
        "的全部判定分支(lint_failed/rejected/accepted/insufficient_trades/"
        "error),这里只留一个显式跳过的占位,说明冒烟应该怎么跑:准备好真实"
        "data_cache后去掉本装饰器,直接调用"
        "scripts.brain_researcher.run_policy_gate(...)不做任何monkeypatch。"
    )
)
def test_real_backtest_engine_smoke():
    pass
