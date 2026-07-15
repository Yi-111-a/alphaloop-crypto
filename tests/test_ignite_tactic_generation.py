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
