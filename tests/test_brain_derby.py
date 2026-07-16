"""覆盖 scripts/ignite.py 里"大脑德比"这段(用户2026-07-16最终指令:
"你不要改成各个不同性格的分支了,直接改成不同的ai大脑,然后重新来")。

测试范围:
  - tournament_mode(config):brain_derby / 缺省tactic_evolution。
  - build_brain_derby_defaults(rotation_pool, now_ms):每个大脑一个分支、
    任务书一字不差、llm_provider预置、空池退化为默认供应商单分支。
  - load_tournament_roster(now_ms, defaults):defaults的value兼容
    str(历史模式)/dict(德比模式)两种形态,dict的meta字段原样进名册。
  - respawn_brain_branch(roster, dead_branch, decision, now_ms):FAIL/CULLED
    累计deaths、PROMOTE累计promotions,新分支id世代号递增且防冲突,
    非大脑分支(policy分支)返回None不复活。
  - load_main_brain / save_main_brain + make_llm_resolver对"main"的解析:
    main_brain.json覆盖默认脑,指向未配置供应商时回退。
  - assign_providers_to_roster不重排德比名册里预置的llm_provider(幂等
    纪律对预置字段同样成立)。

完全离线:不真调任何API,不碰真实state/目录。
"""
from __future__ import annotations

import json

import pytest

import scripts.ignite as ignite


_NOW_MS = 1_760_000_000_000  # 2025-10-09 附近的一个固定UTC时刻,日期部分确定


@pytest.fixture(autouse=True)
def _isolate_state_root(tmp_path, monkeypatch):
    """所有测试都不能碰真实项目的 state/ 目录(名册/main_brain文件)。"""
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ignite, "STATE_ROOT", state_root)
    monkeypatch.setattr(ignite, "TOURNAMENT_ROSTER_PATH", state_root / "tactic_tournament_roster.json")
    monkeypatch.setattr(ignite, "MAIN_BRAIN_PATH", state_root / "main_brain.json")
    return state_root


# ---------------------------------------------------------------------------
# tournament_mode
# ---------------------------------------------------------------------------


def test_tournament_mode_defaults_to_tactic_evolution():
    assert ignite.tournament_mode({}) == "tactic_evolution"
    assert ignite.tournament_mode({"tactic_tournament": {}}) == "tactic_evolution"
    assert ignite.tournament_mode({"tactic_tournament": None}) == "tactic_evolution"


def test_tournament_mode_reads_brain_derby():
    assert ignite.tournament_mode({"tactic_tournament": {"mode": "brain_derby"}}) == "brain_derby"


# ---------------------------------------------------------------------------
# build_brain_derby_defaults
# ---------------------------------------------------------------------------


def test_brain_derby_defaults_one_branch_per_provider_identical_mandate():
    pool = ["ark-kimi", "ark-glm", "ark-doubao", "ark-minimax", "ark-deepseek"]
    defaults = ignite.build_brain_derby_defaults(pool, _NOW_MS)

    assert len(defaults) == 5
    providers = sorted(m["llm_provider"] for m in defaults.values())
    assert providers == sorted(pool)

    # 任务书必须一字不差——德比的全部意义就在于"唯一变量是大脑"。
    tactics = {m["tactics"] for m in defaults.values()}
    assert len(tactics) == 1
    only = next(iter(tactics))
    assert only == ignite._BRAIN_DERBY_MANDATE + ignite._BRAIN_DERBY_STAKES_SUFFIX

    for branch, meta in defaults.items():
        assert branch == f"evo/20251009-brain-{meta['llm_provider']}"
        assert meta["kind"] == "brain"
        assert meta["generation"] == 1
        assert meta["deaths"] == 0
        assert meta["promotions"] == 0


def test_brain_derby_defaults_empty_pool_degrades_to_default_provider():
    defaults = ignite.build_brain_derby_defaults([], _NOW_MS)
    assert len(defaults) == 1
    (meta,) = defaults.values()
    assert meta["llm_provider"] == ignite._DEFAULT_LLM_PROVIDER


def test_brain_derby_mandate_has_no_style_hints():
    """任务书刻意不给风格暗示——出现历史模式的性格关键词说明有人把旧战术
    文字混了进来。"""
    text = ignite._BRAIN_DERBY_MANDATE
    # 注意"分散/集中"这类词在任务书里以"集中还是分散,全部由你自己判断"的
    # 中性列举形式出现是允许的——禁的是旧性格战术的定向短语。
    for banned in ("进取战术", "保守战术", "动量战术", "carry战术", "分散战术",
                   "凉兮", "滚仓", "梭哈", "广撒网", "顺势", "吃资金费率"):
        assert banned not in text, f"中性任务书里不该出现风格暗示词: {banned!r}"


# ---------------------------------------------------------------------------
# load_tournament_roster 的 dict-defaults 兼容
# ---------------------------------------------------------------------------


def test_load_roster_accepts_dict_defaults_and_stamps_status():
    defaults = ignite.build_brain_derby_defaults(["ark-glm", "ark-kimi"], _NOW_MS)
    roster = ignite.load_tournament_roster(_NOW_MS, defaults)

    assert set(roster) == set(defaults)
    for branch, meta in roster.items():
        assert meta["status"] == "active"
        assert meta["created_ms"] == _NOW_MS
        assert meta["llm_provider"] == defaults[branch]["llm_provider"]
        assert meta["kind"] == "brain"

    # 必须已经落盘(与str-defaults同一条"首次建档立刻写文件"纪律)。
    on_disk = json.loads(ignite.TOURNAMENT_ROSTER_PATH.read_text(encoding="utf-8"))
    assert set(on_disk) == set(defaults)


def test_load_roster_still_accepts_legacy_str_defaults():
    roster = ignite.load_tournament_roster(_NOW_MS, {"evo/x": "老式战术文字"})
    assert roster["evo/x"]["tactics"] == "老式战术文字"
    assert roster["evo/x"]["status"] == "active"


def test_load_roster_prefers_existing_file_over_defaults():
    """名册文件已存在时defaults完全不参与——切到德比模式不会隐式改写
    正在跑的旧名册(gen4重开靠显式清零,不靠这里)。"""
    ignite.save_tournament_roster({"evo/old": {"tactics": "t", "status": "active", "created_ms": 1}})
    roster = ignite.load_tournament_roster(
        _NOW_MS, ignite.build_brain_derby_defaults(["ark-glm"], _NOW_MS)
    )
    assert set(roster) == {"evo/old"}


# ---------------------------------------------------------------------------
# respawn_brain_branch
# ---------------------------------------------------------------------------


def _dead_brain_roster() -> dict:
    return {
        "evo/20251009-brain-ark-kimi": {
            "tactics": "任务书", "llm_provider": "ark-kimi", "kind": "brain",
            "status": "failed", "created_ms": 1, "resolved_ms": 2,
            "generation": 1, "deaths": 0, "promotions": 0,
        },
    }


def test_respawn_on_fail_increments_deaths():
    roster, new_id = ignite.respawn_brain_branch(
        _dead_brain_roster(), "evo/20251009-brain-ark-kimi", "FAIL", _NOW_MS
    )
    assert new_id == "evo/20251009-brain-ark-kimi-g2"
    meta = roster[new_id]
    assert meta["llm_provider"] == "ark-kimi"
    assert meta["status"] == "active"
    assert meta["generation"] == 2
    assert meta["deaths"] == 1
    assert meta["promotions"] == 0
    assert meta["respawned_from"] == "evo/20251009-brain-ark-kimi"
    assert meta["tactics"] == ignite._BRAIN_DERBY_MANDATE + ignite._BRAIN_DERBY_STAKES_SUFFIX
    # 死者的条目原样保留(公示历史),不被复活覆盖。
    assert roster["evo/20251009-brain-ark-kimi"]["status"] == "failed"


def test_respawn_on_culled_increments_deaths():
    _, new_id = ignite.respawn_brain_branch(
        _dead_brain_roster(), "evo/20251009-brain-ark-kimi", "CULLED", _NOW_MS
    )
    assert new_id is not None


def test_respawn_on_promote_increments_promotions_not_deaths():
    roster, new_id = ignite.respawn_brain_branch(
        _dead_brain_roster(), "evo/20251009-brain-ark-kimi", "PROMOTE", _NOW_MS
    )
    meta = roster[new_id]
    assert meta["deaths"] == 0
    assert meta["promotions"] == 1


def test_respawn_counters_accumulate_across_generations():
    roster = _dead_brain_roster()
    branch = "evo/20251009-brain-ark-kimi"
    for expected_deaths in (1, 2, 3):
        roster, branch = ignite.respawn_brain_branch(roster, branch, "FAIL", _NOW_MS)
        assert roster[branch]["deaths"] == expected_deaths
        roster[branch]["status"] = "failed"  # 模拟下一轮又死了
    assert roster[branch]["generation"] == 4


def test_respawn_avoids_branch_id_collision():
    roster = _dead_brain_roster()
    # 同一天同一个大脑的g2已经存在(极端情况):世代号必须继续+1,不覆盖。
    roster["evo/20251009-brain-ark-kimi-g2"] = {
        "tactics": "x", "llm_provider": "ark-kimi", "kind": "brain",
        "status": "culled", "created_ms": 3, "generation": 2, "deaths": 1, "promotions": 0,
    }
    new_roster, new_id = ignite.respawn_brain_branch(
        roster, "evo/20251009-brain-ark-kimi", "FAIL", _NOW_MS
    )
    assert new_id == "evo/20251009-brain-ark-kimi-g3"
    assert new_roster["evo/20251009-brain-ark-kimi-g2"]["tactics"] == "x"


def test_respawn_skips_non_brain_branches():
    roster = {
        "evo/20251009-carry_v9": {
            "policy_id": "carry_v9", "tactics": "policy描述", "status": "culled",
            "created_ms": 1, "llm_provider": "ark-glm",  # 轮转分配可能盖过章,但kind不是brain
        },
    }
    same_roster, new_id = ignite.respawn_brain_branch(
        roster, "evo/20251009-carry_v9", "CULLED", _NOW_MS
    )
    assert new_id is None
    assert same_roster is roster


# ---------------------------------------------------------------------------
# main_brain.json + make_llm_resolver 对 main 的解析
# ---------------------------------------------------------------------------


def test_load_main_brain_roundtrip():
    assert ignite.load_main_brain() is None
    ignite.save_main_brain("ark-kimi", "evo/20251009-brain-ark-kimi", _NOW_MS)
    assert ignite.load_main_brain() == "ark-kimi"
    payload = json.loads(ignite.MAIN_BRAIN_PATH.read_text(encoding="utf-8"))
    assert payload["source_branch"] == "evo/20251009-brain-ark-kimi"
    assert payload["promoted_ms"] == _NOW_MS


def test_load_main_brain_corrupt_file_returns_none():
    ignite.MAIN_BRAIN_PATH.write_text("not json", encoding="utf-8")
    assert ignite.load_main_brain() is None


def _clients(*names):
    return {name: (f"routine-{name}", f"deep-{name}") for name in names}


def test_resolver_main_uses_default_brain_without_override():
    resolver = ignite.make_llm_resolver(
        _clients(ignite._DEFAULT_LLM_PROVIDER, "ark-kimi"), lambda: {}
    )
    assert resolver("main") == f"routine-{ignite._DEFAULT_LLM_PROVIDER}"


def test_resolver_main_honors_main_brain_override():
    ignite.save_main_brain("ark-kimi", "evo/x", _NOW_MS)
    resolver = ignite.make_llm_resolver(
        _clients(ignite._DEFAULT_LLM_PROVIDER, "ark-kimi"), lambda: {}
    )
    assert resolver("main") == "routine-ark-kimi"


def test_resolver_main_falls_back_when_override_provider_unavailable():
    """main_brain.json指向的供应商没配key(被build_provider_clients跳过)时
    回退默认脑,不抛异常、不返回None——main的决策周期不能因此失败。"""
    ignite.save_main_brain("ark-vanished", "evo/x", _NOW_MS)
    resolver = ignite.make_llm_resolver(
        _clients(ignite._DEFAULT_LLM_PROVIDER, "ark-kimi"), lambda: {}
    )
    assert resolver("main") == f"routine-{ignite._DEFAULT_LLM_PROVIDER}"


def test_resolver_evo_branch_unaffected_by_main_brain_override():
    ignite.save_main_brain("ark-kimi", "evo/x", _NOW_MS)
    roster = {"evo/b": {"status": "active", "llm_provider": "ark-glm"}}
    resolver = ignite.make_llm_resolver(
        _clients(ignite._DEFAULT_LLM_PROVIDER, "ark-kimi", "ark-glm"), lambda: roster
    )
    assert resolver("evo/b") == "routine-ark-glm"


# ---------------------------------------------------------------------------
# 德比名册的预置llm_provider不被轮转重排
# ---------------------------------------------------------------------------


def test_assign_providers_keeps_preassigned_brain_identity():
    defaults = ignite.build_brain_derby_defaults(["ark-glm", "ark-kimi", "ark-minimax"], _NOW_MS)
    roster = ignite.load_tournament_roster(_NOW_MS, defaults)
    before = {b: m["llm_provider"] for b, m in roster.items()}
    after_roster = ignite.assign_providers_to_roster(roster, ["ark-doubao", "ark-glm"])
    after = {b: m["llm_provider"] for b, m in after_roster.items()}
    assert before == after
