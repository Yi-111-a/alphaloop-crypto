"""覆盖 scripts/ignite.py 里"量化研究员德比"(quant_derby)这段——第五代
架构决定(2026-07-18用户最终指令):"废除大模型直接交易——所有交易由确定性
策略代码执行,大模型转岗为'量化研究员'"。

测试范围:
  - tournament_mode(config):三态(quant_derby / brain_derby / 缺省
    tactic_evolution),旧模式行为不受新增第三态影响。
  - build_quant_derby_defaults(rotation_pool, now_ms, seed_policy_id):
    每个大脑一个分支、分支id含"-quant-"段、policy_id/policy_history预置
    种子值、kind='brain'、tactics任务简介含关键规则字样、空池退化为默认
    供应商单分支。
  - ASSET/strategy/policies/flat_v1.py:通过 policy_lint 静态审查,
    decide() 恒返回空列表(且REQUIRED_HISTORY_BARS/DESCRIPTION契约完整,
    load_policy 能正常加载)。
  - respawn_quant_branch(roster, dead_branch, decision, now_ms):FAIL/
    CULLED/PROMOTE 累计规则与respawn_brain_branch一致,但新分支必须继承
    死者的policy_id和policy_history(研究血统不随账户清零);非大脑分支
    返回None不复活;policy_history缺失时从policy_id兜底重建。
  - PROMOTE 落盘:save_main_policy_id + save_main_brain round-trip(main()
    主循环里quant_mode的PROMOTE分支直接调用这两个既有函数,不走M8
    promote_policy_branch的git-merge链路——这里测的是它们组合使用后
    main分支状态的最终形态)。
  - 旧模式(brain_derby/tactic_evolution)在新增quant_derby模式后行为
    不受影响:load_tournament_roster的str/dict-defaults兼容性、
    respawn_brain_branch(不继承policy_id/policy_history)均维持原样。

完全离线:不真调任何API,不碰真实state/目录。
"""
from __future__ import annotations

import json

import pytest

import scripts.ignite as ignite
from ASSET.strategy.policies import StrategyContext, load_policy
from ASSET.strategy.policy_lint import lint_policy_file


_NOW_MS = 1_760_000_000_000  # 2025-10-09 附近的一个固定UTC时刻,日期部分确定


@pytest.fixture(autouse=True)
def _isolate_state_root(tmp_path, monkeypatch):
    """所有测试都不能碰真实项目的 state/ 目录(名册/main_brain/main_policy文件)。"""
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ignite, "STATE_ROOT", state_root)
    monkeypatch.setattr(ignite, "TOURNAMENT_ROSTER_PATH", state_root / "tactic_tournament_roster.json")
    monkeypatch.setattr(ignite, "MAIN_BRAIN_PATH", state_root / "main_brain.json")
    monkeypatch.setattr(ignite, "MAIN_POLICY_PATH", state_root / "main_policy.json")
    return state_root


# ---------------------------------------------------------------------------
# tournament_mode:三态
# ---------------------------------------------------------------------------


def test_tournament_mode_reads_quant_derby():
    assert ignite.tournament_mode({"tactic_tournament": {"mode": "quant_derby"}}) == "quant_derby"


def test_tournament_mode_old_modes_unaffected_by_third_state():
    assert ignite.tournament_mode({}) == "tactic_evolution"
    assert ignite.tournament_mode({"tactic_tournament": {"mode": "brain_derby"}}) == "brain_derby"


# ---------------------------------------------------------------------------
# build_quant_derby_defaults
# ---------------------------------------------------------------------------


def test_quant_derby_defaults_one_branch_per_provider_with_seed_policy():
    pool = ["ark-kimi", "ark-glm", "ark-doubao"]
    defaults = ignite.build_quant_derby_defaults(pool, _NOW_MS, "flat_v1")

    assert len(defaults) == 3
    providers = sorted(m["llm_provider"] for m in defaults.values())
    assert providers == sorted(pool)

    for branch, meta in defaults.items():
        assert branch == f"evo/20251009-quant-{meta['llm_provider']}"
        assert meta["kind"] == "brain"
        assert meta["policy_id"] == "flat_v1"
        assert meta["policy_history"] == ["flat_v1"]
        assert meta["generation"] == 1
        assert meta["deaths"] == 0
        assert meta["promotions"] == 0

    # 任务简介(不是交易LLM任务书)必须一字不差——所有分支的说明文字统一。
    tactics = {m["tactics"] for m in defaults.values()}
    assert len(tactics) == 1
    assert next(iter(tactics)) == ignite._QUANT_DERBY_BRIEFING


def test_quant_derby_defaults_empty_pool_degrades_to_default_provider():
    defaults = ignite.build_quant_derby_defaults([], _NOW_MS, "flat_v1")
    assert len(defaults) == 1
    (meta,) = defaults.values()
    assert meta["llm_provider"] == ignite._DEFAULT_LLM_PROVIDER
    assert meta["policy_id"] == "flat_v1"


def test_quant_derby_briefing_mentions_core_rules():
    """任务简介必须写明:代码交易、大脑研究、改码须过回测关卡、
    斩杀/复活/接管规则不变——这是给面板/巡检看的,不是交易prompt。"""
    text = ignite._QUANT_DERBY_BRIEFING
    for must_have in ("确定性", "研究员", "回测", "斩杀", "复活", "main"):
        assert must_have in text, f"任务简介缺少关键说明: {must_have!r}"


def test_load_roster_accepts_quant_defaults_and_stamps_status():
    defaults = ignite.build_quant_derby_defaults(["ark-glm", "ark-kimi"], _NOW_MS, "flat_v1")
    roster = ignite.load_tournament_roster(_NOW_MS, defaults)

    assert set(roster) == set(defaults)
    for branch, meta in roster.items():
        assert meta["status"] == "active"
        assert meta["created_ms"] == _NOW_MS
        assert meta["policy_id"] == "flat_v1"
        assert meta["policy_history"] == ["flat_v1"]

    # 首次建档必须立刻落盘(与str/brain-defaults同一条纪律)。
    on_disk = json.loads(ignite.TOURNAMENT_ROSTER_PATH.read_text(encoding="utf-8"))
    assert set(on_disk) == set(defaults)


# ---------------------------------------------------------------------------
# ASSET/strategy/policies/flat_v1.py —— 种子策略
# ---------------------------------------------------------------------------


def test_flat_v1_passes_policy_lint():
    from ASSET.strategy.policies import POLICIES_DIR

    violations = lint_policy_file(POLICIES_DIR / "flat_v1.py")
    assert violations == []


def test_flat_v1_decide_returns_empty_list():
    module = load_policy("flat_v1")
    assert isinstance(module.DESCRIPTION, str) and module.DESCRIPTION
    assert isinstance(module.REQUIRED_HISTORY_BARS, int)

    ctx = StrategyContext(
        ts=_NOW_MS,
        positions={},
        snapshot={"BTC/USDT:USDT": {"last": 60000.0}},
        recent_bars={},
    )
    assert module.decide(ctx) == []


# ---------------------------------------------------------------------------
# respawn_quant_branch
# ---------------------------------------------------------------------------


def _dead_quant_roster() -> dict:
    return {
        "evo/20251009-quant-ark-kimi": {
            "tactics": ignite._QUANT_DERBY_BRIEFING, "llm_provider": "ark-kimi", "kind": "brain",
            "status": "failed", "created_ms": 1, "resolved_ms": 2,
            "generation": 1, "deaths": 0, "promotions": 0,
            "policy_id": "carry_v9", "policy_history": ["flat_v1", "carry_v9"],
        },
    }


def test_respawn_quant_inherits_policy_id_and_history():
    roster, new_id = ignite.respawn_quant_branch(
        _dead_quant_roster(), "evo/20251009-quant-ark-kimi", "FAIL", _NOW_MS
    )
    assert new_id == "evo/20251009-quant-ark-kimi-g2"
    meta = roster[new_id]
    assert meta["llm_provider"] == "ark-kimi"
    assert meta["status"] == "active"
    assert meta["generation"] == 2
    assert meta["deaths"] == 1
    assert meta["promotions"] == 0
    assert meta["respawned_from"] == "evo/20251009-quant-ark-kimi"
    # 核心断言:研究血统(policy_id/policy_history)必须原样继承,不是
    # 退回种子策略——这是quant derby与brain derby复活规则的核心差异。
    assert meta["policy_id"] == "carry_v9"
    assert meta["policy_history"] == ["flat_v1", "carry_v9"]
    # 死者条目原样保留(公示历史)。
    assert roster["evo/20251009-quant-ark-kimi"]["status"] == "failed"


def test_respawn_quant_on_promote_increments_promotions_not_deaths():
    roster, new_id = ignite.respawn_quant_branch(
        _dead_quant_roster(), "evo/20251009-quant-ark-kimi", "PROMOTE", _NOW_MS
    )
    meta = roster[new_id]
    assert meta["deaths"] == 0
    assert meta["promotions"] == 1
    assert meta["policy_id"] == "carry_v9"


def test_respawn_quant_avoids_branch_id_collision():
    roster = _dead_quant_roster()
    roster["evo/20251009-quant-ark-kimi-g2"] = {
        "tactics": "x", "llm_provider": "ark-kimi", "kind": "brain",
        "status": "culled", "created_ms": 3, "generation": 2, "deaths": 1, "promotions": 0,
        "policy_id": "carry_v9", "policy_history": ["flat_v1", "carry_v9"],
    }
    new_roster, new_id = ignite.respawn_quant_branch(
        roster, "evo/20251009-quant-ark-kimi", "FAIL", _NOW_MS
    )
    assert new_id == "evo/20251009-quant-ark-kimi-g3"
    assert new_roster["evo/20251009-quant-ark-kimi-g2"]["tactics"] == "x"


def test_respawn_quant_skips_non_brain_branches():
    roster = {
        "evo/20251009-carry_v9": {
            "policy_id": "carry_v9", "tactics": "policy描述", "status": "culled",
            "created_ms": 1, "llm_provider": "ark-glm",
        },
    }
    same_roster, new_id = ignite.respawn_quant_branch(
        roster, "evo/20251009-carry_v9", "CULLED", _NOW_MS
    )
    assert new_id is None
    assert same_roster is roster


def test_respawn_quant_rebuilds_history_when_missing():
    """防御性兜底:policy_history字段缺失时,从当前policy_id重建单元素
    历史,而不是丢失policy_id本身(见respawn_quant_branch docstring)。"""
    roster = {
        "evo/20251009-quant-ark-glm": {
            "tactics": "t", "llm_provider": "ark-glm", "kind": "brain",
            "status": "failed", "created_ms": 1, "generation": 1, "deaths": 0, "promotions": 0,
            "policy_id": "momentum_v7",  # 没有 policy_history 字段
        },
    }
    new_roster, new_id = ignite.respawn_quant_branch(
        roster, "evo/20251009-quant-ark-glm", "FAIL", _NOW_MS
    )
    assert new_roster[new_id]["policy_id"] == "momentum_v7"
    assert new_roster[new_id]["policy_history"] == ["momentum_v7"]


# ---------------------------------------------------------------------------
# PROMOTE 落盘:main分支切换policy_id + 大脑接管研究权
# (main()主循环quant_mode的PROMOTE分支直接调用这两个既有函数,不经过
# M8 promote_policy_branch的git-merge链路——这里验证组合调用后的最终状态)
# ---------------------------------------------------------------------------


def test_quant_promote_writes_main_policy_and_main_brain():
    assert ignite.load_main_policy_id() is None
    assert ignite.load_main_brain() is None

    winning_branch = "evo/20251009-quant-ark-kimi"
    ignite.save_main_policy_id("carry_v9", winning_branch, _NOW_MS)
    ignite.save_main_brain("ark-kimi", winning_branch, _NOW_MS)

    assert ignite.load_main_policy_id() == "carry_v9"
    assert ignite.load_main_brain() == "ark-kimi"

    policy_payload = json.loads(ignite.MAIN_POLICY_PATH.read_text(encoding="utf-8"))
    assert policy_payload["promoted_from"] == winning_branch
    brain_payload = json.loads(ignite.MAIN_BRAIN_PATH.read_text(encoding="utf-8"))
    assert brain_payload["source_branch"] == winning_branch


# ---------------------------------------------------------------------------
# 旧模式(brain_derby/tactic_evolution)在新增quant_derby之后行为不受影响
# ---------------------------------------------------------------------------


def test_brain_derby_defaults_unaffected_by_quant_derby_addition():
    defaults = ignite.build_brain_derby_defaults(["ark-glm", "ark-kimi"], _NOW_MS)
    for meta in defaults.values():
        assert "policy_id" not in meta
        assert meta["tactics"] == ignite._BRAIN_DERBY_MANDATE + ignite._BRAIN_DERBY_STAKES_SUFFIX


def test_load_roster_still_accepts_legacy_str_defaults():
    roster = ignite.load_tournament_roster(_NOW_MS, {"evo/x": "老式战术文字"})
    assert roster["evo/x"]["tactics"] == "老式战术文字"
    assert roster["evo/x"]["status"] == "active"


def test_respawn_brain_branch_does_not_carry_policy_fields():
    """brain_derby的复活规则维持原样:不写policy_id/policy_history,
    与respawn_quant_branch的核心差异必须继续成立。"""
    roster = {
        "evo/20251009-brain-ark-kimi": {
            "tactics": "任务书", "llm_provider": "ark-kimi", "kind": "brain",
            "status": "failed", "created_ms": 1, "generation": 1, "deaths": 0, "promotions": 0,
        },
    }
    new_roster, new_id = ignite.respawn_brain_branch(
        roster, "evo/20251009-brain-ark-kimi", "FAIL", _NOW_MS
    )
    assert "policy_id" not in new_roster[new_id]
    assert "policy_history" not in new_roster[new_id]
