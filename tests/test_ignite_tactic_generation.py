"""覆盖 scripts/ignite.py 里"锦标赛自动生成替补战术"这一段(2026-07-14用户
要求补上的闭环)。只测试纯函数(_validate_tactic_generation_response /
generate_replacement_tactic),不测试main()里的调度循环本身——那部分只能
靠真实点火时手动验证(见其他ignite.py代码的既有测试覆盖策略)。"""
from __future__ import annotations

import json

import scripts.ignite as ignite


def make_llm_sequence(responses):
    calls = {"count": 0}
    responses = list(responses)

    def _client(prompt: str) -> str:
        calls["count"] += 1
        return responses[calls["count"] - 1]

    _client.calls = calls
    return _client


SAMPLE_EVENT = {
    "branch": "evo/20260714-carry",
    "decision": "FAIL",
    "edge_vs_main_pct": -3.2,
    "max_drawdown_pct": 17.5,
    "reason": "intraday drawdown 17.50% > fail_drawdown_pct 15.00%",
}

SAMPLE_ROSTER = {
    "evo/20260714-aggressive": {"tactics": "进取战术:...", "status": "active", "created_ms": 0},
    "evo/20260714-carry": {"tactics": "carry战术:...", "status": "failed", "created_ms": 0},
}


class TestValidateTacticGenerationResponse:
    def test_valid_response_passes(self):
        raw = json.dumps({
            "branch_id": "evo/20260715-vol-target",
            "tactics": "波动率目标战术:按最近实现波动率反向缩放仓位规模,高波动标的降低仓位、低波动标的提高仓位。",
        })
        result, error = ignite._validate_tactic_generation_response(raw, SAMPLE_ROSTER)
        assert error is None
        assert result["branch_id"] == "evo/20260715-vol-target"
        assert "波动率" in result["tactics"]

    def test_invalid_json_rejected(self):
        result, error = ignite._validate_tactic_generation_response("not json {{{", SAMPLE_ROSTER)
        assert result is None
        assert error == "response_not_valid_json"

    def test_non_object_json_rejected(self):
        result, error = ignite._validate_tactic_generation_response("[1,2,3]", SAMPLE_ROSTER)
        assert result is None
        assert error == "response_not_a_json_object"

    def test_missing_branch_id_rejected(self):
        raw = json.dumps({"tactics": "一个足够长的战术描述文本用于通过长度校验测试"})
        result, error = ignite._validate_tactic_generation_response(raw, SAMPLE_ROSTER)
        assert result is None
        assert "branch_id_invalid" in error

    def test_branch_id_not_starting_with_evo_rejected(self):
        raw = json.dumps({
            "branch_id": "aggressive-v2",
            "tactics": "一个足够长的战术描述文本用于通过长度校验测试",
        })
        result, error = ignite._validate_tactic_generation_response(raw, SAMPLE_ROSTER)
        assert result is None
        assert "branch_id_invalid" in error

    def test_branch_id_collision_with_existing_roster_rejected(self):
        raw = json.dumps({
            "branch_id": "evo/20260714-aggressive",  # 已存在于名册里
            "tactics": "一个足够长的战术描述文本用于通过长度校验测试",
        })
        result, error = ignite._validate_tactic_generation_response(raw, SAMPLE_ROSTER)
        assert result is None
        assert "branch_id_collision" in error

    def test_tactics_too_short_rejected(self):
        raw = json.dumps({"branch_id": "evo/20260715-short", "tactics": "太短"})
        result, error = ignite._validate_tactic_generation_response(raw, SAMPLE_ROSTER)
        assert result is None
        assert "tactics_invalid" in error

    def test_tactics_not_a_string_rejected(self):
        raw = json.dumps({"branch_id": "evo/20260715-x", "tactics": 12345})
        result, error = ignite._validate_tactic_generation_response(raw, SAMPLE_ROSTER)
        assert result is None
        assert "tactics_invalid" in error


class TestGenerateReplacementTactic:
    def test_succeeds_on_first_try(self):
        good = json.dumps({
            "branch_id": "evo/20260715-vol-target",
            "tactics": "波动率目标战术:按最近实现波动率反向缩放仓位规模。",
        })
        llm_client = make_llm_sequence([good])
        result = ignite.generate_replacement_tactic(llm_client, SAMPLE_EVENT, SAMPLE_ROSTER, ["BTC/USDT:USDT"])
        assert result is not None
        assert result["branch_id"] == "evo/20260715-vol-target"
        assert llm_client.calls["count"] == 1

    def test_retries_after_invalid_then_succeeds(self):
        bad = "not json at all"
        good = json.dumps({
            "branch_id": "evo/20260715-vol-target",
            "tactics": "波动率目标战术:按最近实现波动率反向缩放仓位规模。",
        })
        llm_client = make_llm_sequence([bad, good])
        result = ignite.generate_replacement_tactic(llm_client, SAMPLE_EVENT, SAMPLE_ROSTER, ["BTC/USDT:USDT"])
        assert result is not None
        assert llm_client.calls["count"] == 2
        # 第二次调用的prompt里应该带上第一次失败的错误反馈,便于agent纠正
        # (与Trader._format_prompt的retry_feedback机制同一套设计)。

    def test_exhausts_retries_returns_none_not_a_fake_placeholder(self):
        bad = "not json at all"
        llm_client = make_llm_sequence([bad, bad, bad])
        result = ignite.generate_replacement_tactic(
            llm_client, SAMPLE_EVENT, SAMPLE_ROSTER, ["BTC/USDT:USDT"], max_retries=3
        )
        assert result is None
        assert llm_client.calls["count"] == 3

    def test_prompt_includes_active_tactics_to_avoid_duplication(self):
        captured_prompts = []

        def _client(prompt):
            captured_prompts.append(prompt)
            return json.dumps({
                "branch_id": "evo/20260715-vol-target",
                "tactics": "波动率目标战术:按最近实现波动率反向缩放仓位规模。",
            })

        ignite.generate_replacement_tactic(_client, SAMPLE_EVENT, SAMPLE_ROSTER, ["BTC/USDT:USDT"])
        assert "evo/20260714-aggressive" in captured_prompts[0]

    def test_prompt_includes_resolution_reason(self):
        captured_prompts = []

        def _client(prompt):
            captured_prompts.append(prompt)
            return json.dumps({
                "branch_id": "evo/20260715-vol-target",
                "tactics": "波动率目标战术:按最近实现波动率反向缩放仓位规模。",
            })

        ignite.generate_replacement_tactic(_client, SAMPLE_EVENT, SAMPLE_ROSTER, ["BTC/USDT:USDT"])
        assert "evo/20260714-carry" in captured_prompts[0]
        assert "FAIL" in captured_prompts[0]


# ---------------------------------------------------------------------------
# 末位斩杀 evaluate_cull(用户2026-07-15):按窗口收益排名斩最差,带新分支
# 保护期和最小可比数量保护。
# ---------------------------------------------------------------------------

HOUR_MS = 3_600_000
CULL_CFG = {"cull_interval_hours": 12, "cull_min_age_hours": 12}


def _series(*navs, start_ms=0, step_ms=HOUR_MS):
    return [(start_ms + i * step_ms, nav) for i, nav in enumerate(navs)]


class TestEvaluateCull:
    def test_kills_lowest_return_even_if_profitable(self):
        now = 100 * HOUR_MS
        roster = {
            "evo/a": {"status": "active", "created_ms": 0},
            "evo/b": {"status": "active", "created_ms": 0},
            "evo/c": {"status": "active", "created_ms": 0},
        }
        lookup = {
            "evo/a": _series(100, 110, start_ms=now - 11 * HOUR_MS),  # +10%
            "evo/b": _series(100, 103, start_ms=now - 11 * HOUR_MS),  # +3%  <- 最差但在盈利
            "evo/c": _series(100, 105, start_ms=now - 11 * HOUR_MS),  # +5%
        }
        event = ignite.evaluate_cull(roster, now, CULL_CFG, nav_series_lookup=lookup.get)
        assert event is not None
        assert event["branch"] == "evo/b"
        assert event["decision"] == "CULLED"

    def test_young_branch_is_protected(self):
        now = 100 * HOUR_MS
        roster = {
            "evo/old-loser": {"status": "active", "created_ms": 0},
            "evo/old-winner": {"status": "active", "created_ms": 0},
            "evo/newborn": {"status": "active", "created_ms": now - 2 * HOUR_MS},  # 才2小时
        }
        lookup = {
            "evo/old-loser": _series(100, 95, start_ms=now - 11 * HOUR_MS),   # -5%
            "evo/old-winner": _series(100, 105, start_ms=now - 11 * HOUR_MS), # +5%
            "evo/newborn": _series(100, 80, start_ms=now - 2 * HOUR_MS),      # -20%但受保护
        }
        event = ignite.evaluate_cull(roster, now, CULL_CFG, nav_series_lookup=lookup.get)
        assert event is not None
        assert event["branch"] == "evo/old-loser"  # 新分支垫底也不斩,斩老的最差

    def test_no_cull_when_fewer_than_two_eligible(self):
        now = 100 * HOUR_MS
        roster = {
            "evo/only": {"status": "active", "created_ms": 0},
            "evo/baby": {"status": "active", "created_ms": now - HOUR_MS},
        }
        lookup = {
            "evo/only": _series(100, 90, start_ms=now - 11 * HOUR_MS),
            "evo/baby": _series(100, 120, start_ms=now - HOUR_MS),
        }
        event = ignite.evaluate_cull(roster, now, CULL_CFG, nav_series_lookup=lookup.get)
        assert event is None  # 只有1个可排名对象,排名无意义,不斩

    def test_non_active_branches_ignored(self):
        now = 100 * HOUR_MS
        roster = {
            "evo/dead": {"status": "failed", "created_ms": 0},
            "evo/a": {"status": "active", "created_ms": 0},
            "evo/b": {"status": "active", "created_ms": 0},
        }
        lookup = {
            "evo/dead": _series(100, 1, start_ms=now - 11 * HOUR_MS),
            "evo/a": _series(100, 101, start_ms=now - 11 * HOUR_MS),
            "evo/b": _series(100, 99, start_ms=now - 11 * HOUR_MS),
        }
        event = ignite.evaluate_cull(roster, now, CULL_CFG, nav_series_lookup=lookup.get)
        assert event["branch"] == "evo/b"

    def test_branch_without_enough_window_data_skipped(self):
        now = 100 * HOUR_MS
        roster = {
            "evo/a": {"status": "active", "created_ms": 0},
            "evo/b": {"status": "active", "created_ms": 0},
            "evo/no-data": {"status": "active", "created_ms": 0},
        }
        lookup = {
            "evo/a": _series(100, 102, start_ms=now - 11 * HOUR_MS),
            "evo/b": _series(100, 101, start_ms=now - 11 * HOUR_MS),
            "evo/no-data": [(now - HOUR_MS, 100.0)],  # 窗口内只有1个点,无法算收益
        }
        event = ignite.evaluate_cull(roster, now, CULL_CFG, nav_series_lookup=lookup.get)
        assert event["branch"] == "evo/b"


# ---------------------------------------------------------------------------
# horizon到期强平(2026-07-16用户要求"特定时间平仓"):论点有效期从装饰
# 变成真承诺。
# ---------------------------------------------------------------------------


class TestHorizonExpiry:
    def test_parse_horizon_ms(self):
        assert ignite.parse_horizon_ms("4h") == 4 * 3_600_000
        assert ignite.parse_horizon_ms("30m") == 30 * 60_000
        assert ignite.parse_horizon_ms("1d") == 86_400_000
        assert ignite.parse_horizon_ms("1.5h") == int(1.5 * 3_600_000)
        assert ignite.parse_horizon_ms("0h") is None       # 0=不强平
        assert ignite.parse_horizon_ms("永远") is None      # 解析不了不误杀
        assert ignite.parse_horizon_ms("") is None
        assert ignite.parse_horizon_ms(None) is None

    def test_expired_position_selected(self):
        now = 100 * HOUR_MS
        open_pos = {"main": {"BTC/USDT:USDT"}}
        latest = {("main", "BTC/USDT:USDT"): {"ts": now - 13 * HOUR_MS, "horizon": "12h"}}
        expired = ignite.find_horizon_expired_positions(open_pos, latest, now)
        assert len(expired) == 1
        assert expired[0][0] == "main" and expired[0][1] == "BTC/USDT:USDT"

    def test_unexpired_position_kept(self):
        now = 100 * HOUR_MS
        open_pos = {"main": {"BTC/USDT:USDT"}}
        latest = {("main", "BTC/USDT:USDT"): {"ts": now - 5 * HOUR_MS, "horizon": "12h"}}
        assert ignite.find_horizon_expired_positions(open_pos, latest, now) == []

    def test_adjust_renews_the_clock(self):
        """adjust也算新论点声明——最近一笔open/adjust的ts起算,不是最初开仓时间。"""
        now = 100 * HOUR_MS
        open_pos = {"evo/x": {"ETH/USDT:USDT"}}
        # 开仓在20小时前(horizon 12h早过了),但6小时前adjust续期过
        latest = {("evo/x", "ETH/USDT:USDT"): {"ts": now - 6 * HOUR_MS, "horizon": "12h"}}
        assert ignite.find_horizon_expired_positions(open_pos, latest, now) == []

    def test_unparseable_horizon_skipped(self):
        now = 100 * HOUR_MS
        open_pos = {"main": {"BTC/USDT:USDT"}}
        latest = {("main", "BTC/USDT:USDT"): {"ts": 0, "horizon": "看情况"}}
        assert ignite.find_horizon_expired_positions(open_pos, latest, now) == []

    def test_position_without_decision_record_skipped(self):
        now = 100 * HOUR_MS
        open_pos = {"main": {"BTC/USDT:USDT"}}
        assert ignite.find_horizon_expired_positions(open_pos, {}, now) == []

    def test_parse_horizon_freeform_variants(self):
        """2026-07-16实测扩容:模型用自由文本写有效期,17/29笔持仓逃检。"""
        W = 604_800_000
        assert ignite.parse_horizon_ms("1w") == W
        assert ignite.parse_horizon_ms("2 weeks") == 2 * W
        assert ignite.parse_horizon_ms("1 day") == 86_400_000
        assert ignite.parse_horizon_ms("3 days") == 3 * 86_400_000
        assert ignite.parse_horizon_ms("2-4 weeks") == 4 * W  # 范围取上界=承诺最多4周
        assert ignite.parse_horizon_ms("48 hours") == 48 * 3_600_000
        assert ignite.parse_horizon_ms("soon") is None  # 纯口语还是不认
