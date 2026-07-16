from __future__ import annotations

import json

import pytest

from LOCKED.schemas import Decision
from LOCKED.simulator import MIN_THESIS_LEN
from ASSET.strategy.trader import (
    MAX_LEVERAGE,
    MIN_LEVERAGE,
    VALID_ACTIONS,
    Trader,
    references_hypothesis,
)


# ---------------------------------------------------------------------------
# 公共测试夹具 / 假对象
# ---------------------------------------------------------------------------


class FakeMemoryStore:
    """记录每次 retrieve() 调用的参数,用于断言 query_ts 没有被墙钟时间污染。"""

    def __init__(self, results=None):
        self.calls = []
        self._results = results if results is not None else []

    def retrieve(self, query, query_ts, top_k=5, layers=None):
        self.calls.append(
            dict(query=query, query_ts=query_ts, top_k=top_k, layers=layers)
        )
        return self._results


class RecordRecord:
    """带 .content 属性的假 MemoryRecord。"""

    def __init__(self, content):
        self.content = content


def make_llm_sequence(responses):
    """返回一个 callable,每次调用按顺序弹出一个预设响应;记录调用次数。"""
    calls = {"count": 0}
    responses = list(responses)

    def _client(prompt: str) -> str:
        calls["count"] += 1
        idx = calls["count"] - 1
        return responses[idx]

    _client.calls = calls
    return _client


VALID_DECISION_DICT = dict(
    ts=1_700_000_000_000,
    symbol="BTC/USDT:USDT",
    action="open_long",
    target_notional_pct=50.0,
    leverage=3,
    thesis="基于H3:BTC波动率处于历史低位,预期突破概率上升,值得建仓",
    falsifier="若4h收盘价跌破前低支撑位则判断此假设失效,应立即平仓",
    horizon="12h",
    falsifier_condition="price<48000",  # M3: 非hold决策必须携带机器可读证伪条件
)


def valid_response_json(overrides=None, branch=None):
    d = dict(VALID_DECISION_DICT)
    if overrides:
        d.update(overrides)
    if branch is not None:
        d["branch"] = branch
    return json.dumps([d])


# ---------------------------------------------------------------------------
# 1. 输入拼装顺序 + query_ts 时间边界测试(最高优先级)
# ---------------------------------------------------------------------------


class TestBuildContextOrderingAndTimeBoundary:
    def test_memory_store_called_with_decision_ts_not_wallclock(self):
        memory_store = FakeMemoryStore(results=[RecordRecord("some memory")])
        trader = Trader(llm_client=lambda p: "[]", memory_store=memory_store)

        decision_ts = 1_650_000_000_000  # 明显不是"现在"的时间戳
        trader.build_context(
            positions=[],
            ts=decision_ts,
            latest_snapshot={"BTC/USDT:USDT": 60000.0},
            last_reflection_summary="上一轮反思摘要",
            program_tactics="谨慎试仓",
            memory_query_text="BTC trend",
            top_k=5,
        )

        assert len(memory_store.calls) == 1
        call = memory_store.calls[0]
        assert call["query_ts"] == decision_ts
        # 显式排除常见的"墙钟时间"错误:不是当前真实时间,也不是0/None之类占位符
        assert call["query_ts"] != 0
        assert call["top_k"] == 5
        assert call["query"] == "BTC trend"

    def test_context_contains_all_five_parts_in_order(self):
        memory_store = FakeMemoryStore(results=[RecordRecord("mem1")])
        trader = Trader(llm_client=lambda p: "[]", memory_store=memory_store)

        positions = [{"symbol": "BTC/USDT:USDT", "notional": 1000}]
        snapshot = {"BTC/USDT:USDT": 61000.0}
        reflection = "上次反思:趋势跟随策略表现良好"
        tactics = "本周期优先做多主流币"

        context = trader.build_context(
            positions=positions,
            ts=123,
            latest_snapshot=snapshot,
            last_reflection_summary=reflection,
            program_tactics=tactics,
            memory_query_text="q",
            top_k=3,
        )

        assert context["_order"] == [
            "positions",
            "memory_results",
            "latest_snapshot",
            "last_reflection_summary",
            "program_tactics",
        ]
        assert context["positions"] == positions
        assert context["memory_results"] == ["mem1"]
        assert context["latest_snapshot"] == snapshot
        assert context["last_reflection_summary"] == reflection
        assert context["program_tactics"] == tactics

    def test_memory_store_duck_typing_tuple_shape(self):
        """MemoryStore.retrieve 也可能返回 (record, score) 元组,Trader 需要防御性解包。"""

        class TupleMemoryStore:
            def retrieve(self, query, query_ts, top_k=5, layers=None):
                return [(RecordRecord("tuple-wrapped memory"), 0.9)]

        trader = Trader(llm_client=lambda p: "[]", memory_store=TupleMemoryStore())
        context = trader.build_context(
            positions=[],
            ts=1,
            latest_snapshot={},
            last_reflection_summary=None,
            program_tactics=None,
            memory_query_text="q",
        )
        assert context["memory_results"] == ["tuple-wrapped memory"]

    def test_memory_store_duck_typing_dict_shape(self):
        class DictMemoryStore:
            def retrieve(self, query, query_ts, top_k=5, layers=None):
                return [{"content": "dict-shaped memory"}]

        trader = Trader(llm_client=lambda p: "[]", memory_store=DictMemoryStore())
        context = trader.build_context(
            positions=[],
            ts=1,
            latest_snapshot={},
            last_reflection_summary=None,
            program_tactics=None,
            memory_query_text="q",
        )
        assert context["memory_results"] == ["dict-shaped memory"]


# ---------------------------------------------------------------------------
# 2. 首次成功
# ---------------------------------------------------------------------------


class TestDecideFirstTrySuccess:
    def test_valid_llm_output_returns_decisions_and_calls_llm_once(self):
        memory_store = FakeMemoryStore()
        llm_client = make_llm_sequence([valid_response_json()])
        trader = Trader(llm_client=llm_client, memory_store=memory_store)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={"BTC/USDT:USDT": 60000.0},
            memory_query_text="q",
        )

        assert llm_client.calls["count"] == 1
        assert len(decisions) == 1
        assert isinstance(decisions[0], Decision)
        assert decisions[0].action == "open_long"
        assert decisions[0].thesis == VALID_DECISION_DICT["thesis"]
        assert decisions[0].branch == "main"


# ---------------------------------------------------------------------------
# 3. 重试后成功
# ---------------------------------------------------------------------------


class TestRetryThenSucceed:
    def test_two_failures_then_valid_on_third_attempt(self):
        memory_store = FakeMemoryStore()
        bad1 = json.dumps([{**VALID_DECISION_DICT, "falsifier": ""}])  # missing/empty falsifier
        bad2 = json.dumps(
            [{**VALID_DECISION_DICT, "falsifier": "too short"}]
        )  # falsifier too short
        good = valid_response_json()
        llm_client = make_llm_sequence([bad1, bad2, good])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )

        assert llm_client.calls["count"] == 3
        assert len(decisions) == 1
        assert decisions[0].action == "open_long"


# ---------------------------------------------------------------------------
# 4. 重试耗尽 -> 默认hold
# ---------------------------------------------------------------------------


class TestRetryExhaustedFallsBackToHold:
    @pytest.mark.parametrize(
        "bad_overrides",
        [
            {"thesis": None},  # missing-ish thesis (invalid type)
            {"falsifier": "short"},  # too short
            {"action": "definitely_not_valid"},  # invalid literal
            {"leverage": "three"},  # non-int leverage
            {"leverage": 99},  # out-of-range leverage
            {"falsifier_condition": None},  # missing machine-readable condition on a non-hold decision
            {"falsifier_condition": "if price drops a lot"},  # unparseable free text
        ],
        ids=[
            "missing_thesis",
            "falsifier_too_short",
            "invalid_action",
            "leverage_non_int",
            "leverage_out_of_range",
            "missing_falsifier_condition",
            "unparseable_falsifier_condition",
        ],
    )
    def test_always_invalid_defaults_to_hold_after_max_retries(self, bad_overrides):
        memory_store = FakeMemoryStore()
        bad_response = json.dumps([{**VALID_DECISION_DICT, **bad_overrides}])
        llm_client = make_llm_sequence([bad_response, bad_response, bad_response])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )

        assert llm_client.calls["count"] == 3
        assert len(decisions) == 1
        d = decisions[0]
        assert isinstance(d, Decision)
        assert d.action == "hold"
        assert len(d.thesis.strip()) >= MIN_THESIS_LEN
        assert len(d.falsifier.strip()) >= MIN_THESIS_LEN

    def test_missing_thesis_field_entirely(self):
        memory_store = FakeMemoryStore()
        bad = dict(VALID_DECISION_DICT)
        del bad["thesis"]
        bad_response = json.dumps([bad])
        llm_client = make_llm_sequence([bad_response] * 3)
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=42,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )
        assert llm_client.calls["count"] == 3
        assert decisions[0].action == "hold"


class TestFalsifierConditionRequiredForNonHold:
    """M3 裁决:falsifier 的证伪判定不能全靠 LLM 自评，Trader 必须为非hold决策
    额外产出机器可读的 falsifier_condition，供 Reflector 用确定性代码判定。"""

    def test_valid_falsifier_condition_accepted_on_first_try(self):
        memory_store = FakeMemoryStore()
        llm_client = make_llm_sequence([valid_response_json()])
        trader = Trader(llm_client=llm_client, memory_store=memory_store)

        decisions = trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q"
        )
        assert llm_client.calls["count"] == 1
        assert decisions[0].falsifier_condition == "price<48000"
        from LOCKED.schemas import parse_falsifier_condition
        assert parse_falsifier_condition(decisions[0].falsifier_condition) is not None

    def test_hold_decision_does_not_require_falsifier_condition(self):
        memory_store = FakeMemoryStore()
        hold_dict = dict(VALID_DECISION_DICT)
        hold_dict["action"] = "hold"
        hold_dict["target_notional_pct"] = 0.0
        del hold_dict["falsifier_condition"]  # absent entirely, not even null
        llm_client = make_llm_sequence([json.dumps([hold_dict])])
        trader = Trader(llm_client=llm_client, memory_store=memory_store)

        decisions = trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q"
        )
        assert llm_client.calls["count"] == 1, "a hold decision missing falsifier_condition must NOT trigger a retry"
        assert decisions[0].action == "hold"
        assert decisions[0].falsifier_condition is None

    def test_missing_falsifier_condition_on_open_long_triggers_retry_then_fallback(self):
        memory_store = FakeMemoryStore()
        bad = dict(VALID_DECISION_DICT)
        del bad["falsifier_condition"]
        bad_response = json.dumps([bad])
        llm_client = make_llm_sequence([bad_response, bad_response, bad_response])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q"
        )
        assert llm_client.calls["count"] == 3
        assert decisions[0].action == "hold"  # fell back to safe default


# ---------------------------------------------------------------------------
# 5. Trader校验规则镜像 Simulator 自己的校验规则
# ---------------------------------------------------------------------------


class TestValidationMirrorsSimulator:
    def test_trader_produced_decision_passes_simulator_style_checks(self):
        memory_store = FakeMemoryStore()
        llm_client = make_llm_sequence([valid_response_json()])
        trader = Trader(llm_client=llm_client, memory_store=memory_store)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )
        d = decisions[0]

        # 重新推导 LOCKED/simulator.py execute() 校验链步骤3、5 用到的规则,
        # 而不是只信任 Trader 自己的 schema。
        assert len(d.thesis.strip()) >= MIN_THESIS_LEN
        assert len(d.falsifier.strip()) >= MIN_THESIS_LEN
        assert d.action in VALID_ACTIONS
        assert d.action in {"open_long", "open_short", "close", "adjust", "hold"}
        assert isinstance(d.leverage, int) and not isinstance(d.leverage, bool)
        assert MIN_LEVERAGE <= d.leverage <= MAX_LEVERAGE <= 10
        assert isinstance(d.target_notional_pct, (int, float))

    def test_fallback_hold_also_passes_simulator_style_checks(self):
        memory_store = FakeMemoryStore()
        bad_response = "not json at all {{{"
        llm_client = make_llm_sequence([bad_response] * 3)
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )
        d = decisions[0]
        assert len(d.thesis.strip()) >= MIN_THESIS_LEN
        assert len(d.falsifier.strip()) >= MIN_THESIS_LEN
        assert d.action in VALID_ACTIONS
        assert isinstance(d.leverage, int)
        assert MIN_LEVERAGE <= d.leverage <= MAX_LEVERAGE


# ---------------------------------------------------------------------------
# 5b. 语义级校验(2026-07-14新增):不只是"类型对不对",而是"这个值在这个
# 业务场景下讲不讲得通"——都是本session里签入agent反复手动拦下过的真实
# 错误类别,把人工审查的判断标准搬进自动重试校验里。
# ---------------------------------------------------------------------------


class TestSemanticValidationCatchesRealErrorClasses:
    def test_shortened_symbol_not_in_snapshot_is_rejected_and_retried(self):
        memory_store = FakeMemoryStore()
        bad = valid_response_json(overrides={"symbol": "BTC"})  # 截断写法,不是"BTC/USDT:USDT"
        good = valid_response_json()  # 第二次改用完整交易对
        llm_client = make_llm_sequence([bad, good])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}},
            memory_query_text="q",
        )
        assert llm_client.calls["count"] == 2  # 第一次真的被拒绝、触发了重试
        assert decisions[0].symbol == "BTC/USDT:USDT"

    def test_symbol_outside_this_cycle_universe_is_rejected(self):
        memory_store = FakeMemoryStore()
        bad = valid_response_json(overrides={"symbol": "DOGE/USDT:USDT"})
        good = valid_response_json()
        llm_client = make_llm_sequence([bad, good])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            # 本轮快照只覆盖 BTC,不覆盖 DOGE——即使 DOGE/USDT:USDT 格式完全
            # 合法,也应该被拒绝,因为它不在"这一刻真实可交易"的集合里。
            latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}},
            memory_query_text="q",
        )
        assert llm_client.calls["count"] == 2
        assert decisions[0].symbol == "BTC/USDT:USDT"

    def test_symbol_check_skipped_when_snapshot_empty(self):
        """行情快照为空(比如数据源暂时拉不到)时不该把这个新校验变成
        "全部拒绝"——应该退化为跳过symbol语义检查,只做类型/范围校验。"""
        memory_store = FakeMemoryStore()
        llm_client = make_llm_sequence([valid_response_json()])
        trader = Trader(llm_client=llm_client, memory_store=memory_store)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )
        assert llm_client.calls["count"] == 1
        assert decisions[0].symbol == "BTC/USDT:USDT"

    def test_negative_target_notional_pct_is_rejected_and_retried(self):
        memory_store = FakeMemoryStore()
        bad = valid_response_json(overrides={"target_notional_pct": -16.0})
        good = valid_response_json()
        llm_client = make_llm_sequence([bad, good])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )
        assert llm_client.calls["count"] == 2
        assert decisions[0].target_notional_pct >= 0

    def test_target_notional_pct_over_100_is_rejected(self):
        memory_store = FakeMemoryStore()
        bad = valid_response_json(overrides={"target_notional_pct": 500.0})
        good = valid_response_json()
        llm_client = make_llm_sequence([bad, good])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )
        assert llm_client.calls["count"] == 2
        assert decisions[0].target_notional_pct <= 100


# ---------------------------------------------------------------------------
# 6. references_hypothesis()
# ---------------------------------------------------------------------------


class TestReferencesHypothesis:
    @pytest.mark.parametrize(
        "thesis,expected",
        [
            ("基于H3的判断,BTC即将突破", True),
            ("H12abc trending", True),
            ("基于H15的假设,资金费率将转负", True),
            ("单纯的直觉,没有任何依据", False),
            ("", False),
            ("H", False),  # H 后面没有数字,不算引用
            ("这是关于H的讨论,但没有编号", False),
            ("参考 h3 假设(小写)", False),  # 大小写敏感,正则要求大写H
        ],
    )
    def test_references_hypothesis(self, thesis, expected):
        assert references_hypothesis(thesis) is expected

    def test_none_thesis_returns_false(self):
        assert references_hypothesis(None) is False


# ---------------------------------------------------------------------------
# 7. 彻底无法解析的 JSON(而不仅仅是schema不合格)
# ---------------------------------------------------------------------------


class TestMalformedJsonHandledAsValidationFailure:
    def test_unparseable_json_triggers_retry_not_exception(self):
        memory_store = FakeMemoryStore()
        malformed = "this is { not : valid json at all"
        good = valid_response_json()
        llm_client = make_llm_sequence([malformed, malformed, good])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        # 不应该抛出任何异常
        decisions = trader.decide(
            ts=1_700_000_000_000,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )
        assert llm_client.calls["count"] == 3
        assert decisions[0].action == "open_long"

    def test_unparseable_json_all_attempts_defaults_to_hold_no_exception(self):
        memory_store = FakeMemoryStore()
        malformed = "{{{not json"
        llm_client = make_llm_sequence([malformed, malformed, malformed])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )
        assert llm_client.calls["count"] == 3
        assert len(decisions) == 1
        assert decisions[0].action == "hold"

    def test_json_that_is_not_a_list_treated_as_invalid(self):
        memory_store = FakeMemoryStore()
        not_a_list = json.dumps({"action": "hold"})  # valid JSON, wrong top-level shape
        good = valid_response_json()
        llm_client = make_llm_sequence([not_a_list, not_a_list, good])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1,
            positions=[],
            latest_snapshot={},
            memory_query_text="q",
        )
        assert llm_client.calls["count"] == 3
        assert decisions[0].action == "open_long"


class TestMinimumMeaningfulPositionSize:
    """2026-07-15:真实观察到LLM给出0.02%/0.1%的仓位(100U本金下连手续费都
    覆盖不了),开仓/调仓最低1个百分点,低于即拒绝重试并在反馈里讲清单位。"""

    def test_dust_position_rejected_then_retried(self):
        memory_store = FakeMemoryStore()
        bad = valid_response_json(overrides={"target_notional_pct": 0.1})  # 0.1% dust
        good = valid_response_json(overrides={"target_notional_pct": 10.0})
        llm_client = make_llm_sequence([bad, good])
        trader = Trader(llm_client=llm_client, memory_store=memory_store, max_retries=3)

        decisions = trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q",
        )
        assert llm_client.calls["count"] == 2
        assert decisions[0].target_notional_pct >= 1.0

    def test_hold_with_zero_pct_still_valid(self):
        memory_store = FakeMemoryStore()
        hold = valid_response_json(overrides={
            "action": "hold", "target_notional_pct": 0.0, "falsifier_condition": None,
        })
        llm_client = make_llm_sequence([hold])
        trader = Trader(llm_client=llm_client, memory_store=memory_store)

        decisions = trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q",
        )
        assert llm_client.calls["count"] == 1
        assert decisions[0].action == "hold"


class TestMarketRegimeLine:
    """2026-07-15:快照带趋势字段时,prompt里必须出现大势判断行(API模式下
    模型无状态,没有这行它不知道全场在跌还是在涨)。"""

    def test_regime_line_appears_with_trend_fields(self):
        from ASSET.strategy.trader import _market_regime_line

        snap = {
            "A/USDT:USDT": {"last": 1.0, "chg_24h_pct": -5.0, "chg_7d_pct": -12.0},
            "B/USDT:USDT": {"last": 2.0, "chg_24h_pct": -3.0, "chg_7d_pct": -8.0},
            "C/USDT:USDT": {"last": 3.0, "chg_24h_pct": 1.0, "chg_7d_pct": 2.0},
        }
        line = _market_regime_line(snap)
        assert "2/3 symbols are DOWN" in line
        assert "open_short" in line

    def test_no_trend_fields_returns_empty(self):
        from ASSET.strategy.trader import _market_regime_line

        assert _market_regime_line({"A/USDT:USDT": {"last": 1.0}}) == ""
        assert _market_regime_line({"A/USDT:USDT": 60000.0}) == ""  # 旧式标量快照不崩

    def test_prompt_contains_regime_when_snapshot_enriched(self):
        memory_store = FakeMemoryStore()
        trader = Trader(llm_client=lambda p: "[]", memory_store=memory_store)
        ctx = trader.build_context(
            positions=[], ts=1,
            latest_snapshot={"A/USDT:USDT": {"last": 1.0, "chg_24h_pct": -9.9}},
            last_reflection_summary=None, program_tactics=None, memory_query_text="q",
        )
        prompt = trader._format_prompt(ctx, None, ts=1)
        assert "MARKET REGIME" in prompt


# ---------------------------------------------------------------------------
# 6. llm_client_resolver(2026-07-16,分支级认知多样性):不同分支用不同的
#    llm_client,打破单一模型的思维趋同。改动最小化的验收标准就是这里的
#    最后一个测试:不传resolver时,行为必须与改造前逐字节相同。
# ---------------------------------------------------------------------------


class TestLlmClientResolver:
    def test_resolver_routes_different_branches_to_different_clients(self):
        memory_store = FakeMemoryStore()
        client_a = make_llm_sequence([valid_response_json(branch="evo/a")])
        client_b = make_llm_sequence([valid_response_json(branch="evo/b")])
        default_client = make_llm_sequence(["should never be called"])

        def resolver(branch):
            return {"evo/a": client_a, "evo/b": client_b}.get(branch)

        trader = Trader(
            llm_client=default_client, memory_store=memory_store, llm_client_resolver=resolver,
        )

        trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q",
            branch="evo/a",
        )
        trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q",
            branch="evo/b",
        )

        assert client_a.calls["count"] == 1
        assert client_b.calls["count"] == 1
        assert default_client.calls["count"] == 0  # 两个分支都被resolver接管了,default完全没被调用

    def test_resolver_returning_none_falls_back_to_default_client(self):
        memory_store = FakeMemoryStore()
        default_client = make_llm_sequence([valid_response_json(branch="evo/unrouted")])

        def resolver(branch):
            return None  # 查不到该分支对应的client

        trader = Trader(
            llm_client=default_client, memory_store=memory_store, llm_client_resolver=resolver,
        )
        decisions = trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q",
            branch="evo/unrouted",
        )

        assert default_client.calls["count"] == 1
        assert decisions[0].branch == "evo/unrouted"

    def test_resolver_raising_exception_falls_back_to_default_client(self):
        """resolver本身出故障(比如名册文件损坏)不应该让decide()崩溃——分支
        路由是锦上添花的功能,不能拖累决策周期本身的可用性。"""
        memory_store = FakeMemoryStore()
        default_client = make_llm_sequence([valid_response_json()])

        def broken_resolver(branch):
            raise RuntimeError("roster file corrupted")

        trader = Trader(
            llm_client=default_client, memory_store=memory_store, llm_client_resolver=broken_resolver,
        )
        decisions = trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q",
        )

        assert default_client.calls["count"] == 1
        assert decisions[0].action == "open_long"

    def test_no_resolver_behaves_identically_to_before(self):
        """没有传llm_client_resolver时(默认None),行为必须与改造前完全一致
        ——只使用构造时传入的self.llm_client,不做任何分支查询。"""
        memory_store = FakeMemoryStore()
        llm_client = make_llm_sequence([valid_response_json()])
        trader = Trader(llm_client=llm_client, memory_store=memory_store)

        assert trader.llm_client_resolver is None
        decisions = trader.decide(
            ts=1_700_000_000_000, positions=[], latest_snapshot={}, memory_query_text="q",
        )
        assert llm_client.calls["count"] == 1
        assert decisions[0].action == "open_long"
