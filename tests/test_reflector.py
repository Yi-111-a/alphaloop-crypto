"""
tests/test_reflector.py —— M3 反思模块验收测试。

覆盖 spec §6 M3 两条验收标准:
  - "反思输入被工程隔离(单测:Reflector函数无法接收新闻类参数)" -> 结构性
    封死测试 1/2。
  - "falsifier触发的决策被正确标注'证伪'并进入L3" -> determine_thesis_status
    纯函数测试 + 端到端 L3 晋升测试(test 6,视为本文件优先级最高的测试)。

以及复审要求的"LLM 绝不能覆盖确定性判定"这条最重要的性质(test 4)。
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import LOCKED.reflector as reflector_module
from LOCKED import log_writer
from LOCKED.reflector import Reflector, determine_thesis_status, parse_horizon_to_ms
from LOCKED.schemas import Decision, ThesisMark

_MODULE_SOURCE = Path(reflector_module.__file__).read_text(encoding="utf-8")
_MODULE_TREE = ast.parse(_MODULE_SOURCE)


# ---------------------------------------------------------------------------
# 公共测试夹具 / 假对象
# ---------------------------------------------------------------------------


class FakeMemoryStore:
    """只记录 write() 调用,没有 retrieve() 方法(如果 Reflector 代码路径里
    不小心调了 .retrieve(...),这个假对象会用 AttributeError 直接暴露问题,
    是 AST 静态检查之外的一道运行时冗余防线)。"""

    def __init__(self):
        self.writes: list[dict] = []

    def write(self, content, ts, layer, importance=1.0):
        record = dict(content=content, ts=ts, layer=layer, importance=importance)
        self.writes.append(record)
        return record


def make_llm_recorder(response: str = "经验摘要占位符,概括本轮反思的教训。"):
    calls: list[str] = []

    def _client(prompt: str) -> str:
        calls.append(prompt)
        return response

    _client.calls = calls
    return _client


def _decision_dict(**overrides) -> dict:
    base = dict(
        ts=1_700_000_000_000,
        symbol="BTC/USDT:USDT",
        action="open_long",
        target_notional_pct=50.0,
        leverage=3,
        thesis="基于H1:BTC波动率处于历史低位,预期突破概率上升,值得建仓",
        falsifier="若4h收盘价跌破前低支撑位则判断此假设失效,应立即平仓",
        horizon="12h",
        branch="main",
        falsifier_condition="price<48000",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. 结构性封死测试 #1(AST):禁止新闻/检索类 import;reflect() 签名严格封死
# ---------------------------------------------------------------------------


def test_no_news_or_search_imports_anywhere_in_reflector():
    forbidden_substrings = {"news", "search", "web", "fetch", "arxiv", "github"}
    offenders: list[str] = []

    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name_lower = alias.name.lower()
                if any(token in name_lower for token in forbidden_substrings):
                    offenders.append(f"import {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            mod_lower = (node.module or "").lower()
            if any(token in mod_lower for token in forbidden_substrings):
                offenders.append(f"from {node.module} import ... (line {node.lineno})")

    assert offenders == [], f"forbidden news/search-ish imports found: {offenders}"


def test_reflect_signature_is_exactly_sealed():
    """reflect()'s parameter list must be EXACTLY {self, now_ts, branch, window,
    price_lookup} -- no *args/**kwargs, no extra params that could smuggle in a
    news/research_notes/free-text-query channel."""
    reflect_def = None
    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.ClassDef) and node.name == "Reflector":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "reflect":
                    reflect_def = item
    assert reflect_def is not None, "could not locate Reflector.reflect in AST"

    args = reflect_def.args
    param_names = {a.arg for a in args.args}
    assert param_names == {"self", "now_ts", "branch", "window", "price_lookup"}, param_names
    assert args.vararg is None, "reflect() must not accept *args"
    assert args.kwarg is None, "reflect() must not accept **kwargs"
    assert args.kwonlyargs == [], "reflect() must not accept extra keyword-only params"

    # Cross-check with the live signature too (belt and suspenders).
    sig = inspect.signature(Reflector.reflect)
    assert set(sig.parameters.keys()) == {"self", "now_ts", "branch", "window", "price_lookup"}


# ---------------------------------------------------------------------------
# 2. 结构性封死测试 #2(AST):全文件不允许出现 .retrieve(...) 调用
# ---------------------------------------------------------------------------


def test_no_retrieve_call_anywhere_in_reflector():
    """Reflector may WRITE to memory (memory_store.write(...)) but must never
    READ/query it. Catches memory_store.retrieve(...) or any other
    `.retrieve(...)` call anywhere in the module."""
    offenders: list[int] = []
    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "retrieve":
            offenders.append(node.lineno)
    assert offenders == [], f"forbidden .retrieve(...) call(s) found at line(s): {offenders}"


# ---------------------------------------------------------------------------
# 7. 无墙钟调用(AST 回归护栏,与 ASSET/memory/engine.py 同款)
# ---------------------------------------------------------------------------


def test_no_wallclock_calls_anywhere_in_reflector():
    forbidden_modules = {"time", "datetime"}
    offenders: list[str] = []

    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    offenders.append(f"import {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.module in forbidden_modules:
                offenders.append(f"from {node.module} import ... (line {node.lineno})")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"time", "now", "utcnow", "today"}:
                offenders.append(f"call .{func.attr}(...) (line {node.lineno})")

    assert offenders == [], f"forbidden wall-clock references found in reflector.py: {offenders}"


# ---------------------------------------------------------------------------
# 3. determine_thesis_status —— 纯函数、确定性判定测试
# ---------------------------------------------------------------------------


def test_determine_thesis_status_falsified_when_price_dips_within_horizon():
    decision = Decision(
        ts=1_700_000_000_000,
        symbol="BTC/USDT:USDT",
        action="open_long",
        target_notional_pct=50.0,
        leverage=3,
        thesis="占位thesis,用于纯函数判定测试,长度需要达标",
        falsifier="占位falsifier,用于纯函数判定测试,长度需要达标",
        horizon="12h",
        falsifier_condition="price<48000",
    )
    horizon_ms = parse_horizon_to_ms("12h")
    price_samples = [
        (decision.ts, 51_000.0),
        (decision.ts + horizon_ms // 2, 47_000.0),  # dips below threshold mid-horizon
        (decision.ts + horizon_ms, 52_000.0),  # recovers by the end
    ]
    status, reason = determine_thesis_status(decision, price_samples, now_ts=decision.ts + horizon_ms)
    assert status == "证伪"
    assert "47000" in reason


def test_determine_thesis_status_confirmed_when_price_stays_above_threshold_and_horizon_elapsed():
    decision = Decision(
        ts=1_700_000_000_000,
        symbol="ETH/USDT:USDT",
        action="open_long",
        target_notional_pct=50.0,
        leverage=3,
        thesis="占位thesis,用于纯函数判定测试,长度需要达标",
        falsifier="占位falsifier,用于纯函数判定测试,长度需要达标",
        horizon="12h",
        falsifier_condition="price<48000",
    )
    horizon_ms = parse_horizon_to_ms("12h")
    price_samples = [
        (decision.ts, 51_000.0),
        (decision.ts + horizon_ms // 2, 50_500.0),
        (decision.ts + horizon_ms, 52_000.0),
    ]
    status, reason = determine_thesis_status(decision, price_samples, now_ts=decision.ts + horizon_ms)
    assert status == "应验"


def test_determine_thesis_status_undetermined_when_horizon_not_elapsed():
    decision = Decision(
        ts=1_700_000_000_000,
        symbol="SOL/USDT:USDT",
        action="open_long",
        target_notional_pct=50.0,
        leverage=3,
        thesis="占位thesis,用于纯函数判定测试,长度需要达标",
        falsifier="占位falsifier,用于纯函数判定测试,长度需要达标",
        horizon="12h",
        falsifier_condition="price<48000",
    )
    horizon_ms = parse_horizon_to_ms("12h")
    price_samples = [
        (decision.ts, 51_000.0),
        (decision.ts + horizon_ms // 4, 50_500.0),
    ]
    now_ts = decision.ts + horizon_ms // 2  # horizon has NOT fully elapsed yet
    status, reason = determine_thesis_status(decision, price_samples, now_ts=now_ts)
    assert status == "未决"
    assert "horizon not yet elapsed" in reason


def test_determine_thesis_status_undetermined_when_falsifier_condition_missing_or_unparseable():
    horizon_ms = parse_horizon_to_ms("12h")
    for bad_condition in (None, "", "not a condition", "price??48000"):
        decision = Decision(
            ts=1_700_000_000_000,
            symbol="BTC/USDT:USDT",
            action="open_long",
            target_notional_pct=50.0,
            leverage=3,
            thesis="占位thesis,用于纯函数判定测试,长度需要达标",
            falsifier="占位falsifier,用于纯函数判定测试,长度需要达标",
            horizon="12h",
            falsifier_condition=bad_condition,
        )
        # Even wildly falsifying prices must not matter -- there's no machine
        # readable condition to evaluate them against.
        price_samples = [(decision.ts, 1.0), (decision.ts + horizon_ms, 1.0)]
        status, reason = determine_thesis_status(
            decision, price_samples, now_ts=decision.ts + horizon_ms
        )
        assert status == "未决", f"failed for condition={bad_condition!r}"
        assert "falsifier_condition" in reason


def test_determine_thesis_status_boundary_price_exactly_at_threshold_not_triggered():
    """Matches evaluate_falsifier_condition's own strict-< semantics: price
    exactly AT the threshold does not trigger "price<48000"."""
    decision = Decision(
        ts=1_700_000_000_000,
        symbol="BTC/USDT:USDT",
        action="open_long",
        target_notional_pct=50.0,
        leverage=3,
        thesis="占位thesis,用于纯函数判定测试,长度需要达标",
        falsifier="占位falsifier,用于纯函数判定测试,长度需要达标",
        horizon="12h",
        falsifier_condition="price<48000",
    )
    horizon_ms = parse_horizon_to_ms("12h")
    price_samples = [
        (decision.ts, 48_000.0),
        (decision.ts + horizon_ms, 48_000.0),
    ]
    status, _ = determine_thesis_status(decision, price_samples, now_ts=decision.ts + horizon_ms)
    assert status == "应验"  # never strictly below 48000 -> not falsified, horizon elapsed


def test_parse_horizon_to_ms_supports_hours_and_days_and_rejects_garbage():
    assert parse_horizon_to_ms("12h") == 12 * 3_600_000
    assert parse_horizon_to_ms("3d") == 3 * 86_400_000
    for bad in ("12", "1w", "abc", "-1h", "0d", ""):
        try:
            parse_horizon_to_ms(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for horizon={bad!r}")


# ---------------------------------------------------------------------------
# 4. LLM 绝不能被问"这个决策对不对" —— 只能在判定结果之上写摘要
# ---------------------------------------------------------------------------


def test_llm_is_never_asked_to_judge_only_to_summarize(tmp_path):
    log_root = tmp_path / "LOG"

    falsified = _decision_dict(
        ts=1_700_000_000_000, symbol="BTC/USDT:USDT", horizon="12h",
        falsifier_condition="price<48000",
    )
    confirmed = _decision_dict(
        ts=1_700_010_000_000, symbol="ETH/USDT:USDT", horizon="12h",
        falsifier_condition="price<1000",
    )
    hold = _decision_dict(
        ts=1_700_020_000_000, symbol="SOL/USDT:USDT", action="hold",
        falsifier_condition=None,
    )
    for d in (falsified, confirmed, hold):
        log_writer.append_jsonl("decisions.jsonl", d, root=log_root)

    horizon_ms = parse_horizon_to_ms("12h")
    now_ts = confirmed["ts"] + horizon_ms  # both horizons fully elapsed

    def price_lookup(symbol, ts):
        if symbol == "BTC/USDT:USDT":
            return 40_000.0  # always below 48000 -> falsified
        return 5_000.0  # always above 1000 -> confirmed

    llm = make_llm_recorder()
    memory = FakeMemoryStore()
    reflector = Reflector(llm_client=llm, memory_store=memory, log_root=log_root)

    marks = reflector.reflect(now_ts=now_ts, branch="main", window=20, price_lookup=price_lookup)

    status_by_symbol = {m.symbol: m.thesis_status for m in marks}
    assert status_by_symbol["BTC/USDT:USDT"] == "证伪"
    assert status_by_symbol["ETH/USDT:USDT"] == "应验"
    assert "SOL/USDT:USDT" not in status_by_symbol  # hold decisions are never judged

    # The LLM is called AT MOST ONCE per reflect() call, for the aggregate
    # summary -- not once per decision asking "is this right".
    assert len(llm.calls) == 1
    prompt = llm.calls[0]

    # The verdict must already be baked into the prompt text -- proof the LLM
    # is structurally incapable of being the one who decides it.
    assert "证伪" in prompt
    assert "应验" in prompt
    assert "ALREADY JUDGED" in prompt
    assert "Do NOT re-judge" in prompt


# ---------------------------------------------------------------------------
# 5. ThesisMark 持久化
# ---------------------------------------------------------------------------


def test_thesis_marks_persisted_and_match_standalone_determination(tmp_path):
    log_root = tmp_path / "LOG"

    d1 = _decision_dict(ts=1_700_000_000_000, symbol="BTC/USDT:USDT", horizon="12h",
                         falsifier_condition="price<48000")
    d2 = _decision_dict(ts=1_700_005_000_000, symbol="ETH/USDT:USDT", horizon="12h",
                         falsifier_condition="price<1000")
    for d in (d1, d2):
        log_writer.append_jsonl("decisions.jsonl", d, root=log_root)

    horizon_ms = parse_horizon_to_ms("12h")
    now_ts = d2["ts"] + horizon_ms

    def price_lookup(symbol, ts):
        return 40_000.0 if symbol == "BTC/USDT:USDT" else 5_000.0

    memory = FakeMemoryStore()
    reflector = Reflector(llm_client=make_llm_recorder(), memory_store=memory, log_root=log_root)
    marks = reflector.reflect(now_ts=now_ts, branch="main", window=20, price_lookup=price_lookup)
    assert len(marks) == 2

    persisted = log_writer.read_jsonl("reflections/marks.jsonl", root=log_root)
    assert len(persisted) == 2

    persisted_by_symbol = {row["symbol"]: row for row in persisted}

    # Recompute standalone via determine_thesis_status and cross-check.
    for raw in (d1, d2):
        decision_obj = Decision(
            ts=raw["ts"], symbol=raw["symbol"], action=raw["action"],
            target_notional_pct=raw["target_notional_pct"], leverage=raw["leverage"],
            thesis=raw["thesis"], falsifier=raw["falsifier"], horizon=raw["horizon"],
            branch=raw["branch"], falsifier_condition=raw["falsifier_condition"],
        )
        expected_status, _ = determine_thesis_status(
            decision_obj,
            [(raw["ts"], price_lookup(raw["symbol"], raw["ts"])),
             (raw["ts"] + horizon_ms, price_lookup(raw["symbol"], raw["ts"] + horizon_ms))],
            now_ts=now_ts,
        )
        assert persisted_by_symbol[raw["symbol"]]["thesis_status"] == expected_status
        assert persisted_by_symbol[raw["symbol"]]["decision_ts"] == raw["ts"]


# ---------------------------------------------------------------------------
# 6. THE END-TO-END L3 PROMOTION TEST(最高优先级)
# ---------------------------------------------------------------------------


def test_falsified_decision_promotes_l3_lesson_and_trader_retrieves_it_respecting_time_boundary(tmp_path):
    from ASSET.memory.engine import MemoryStore
    from ASSET.strategy.trader import Trader

    log_root = tmp_path / "LOG"
    memory = MemoryStore(db_path=tmp_path / "memory.db")

    probe = "PROBEWORD9421ALPHALOOP"
    decision_ts = 1_700_000_000_000
    horizon = "12h"
    horizon_ms = parse_horizon_to_ms(horizon)

    decision_dict = _decision_dict(
        ts=decision_ts,
        symbol="BTC/USDT:USDT",
        horizon=horizon,
        thesis=f"基于H5:BTC筑底反弹,预期突破前高阻力位,{probe},值得逢低加多",
        falsifier="若价格跌破48000则判断此假设失效,应立即平仓离场",
        falsifier_condition="price<48000",
    )
    log_writer.append_jsonl("decisions.jsonl", decision_dict, root=log_root)

    now_ts = decision_ts + horizon_ms  # horizon fully elapsed by reflection time

    dip_lo = decision_ts + int(horizon_ms * 0.3)
    dip_hi = decision_ts + int(horizon_ms * 0.7)

    def price_lookup(symbol, ts):
        if dip_lo <= ts <= dip_hi:
            return 47_000.0  # dips below the falsifier threshold mid-horizon
        return 51_000.0

    reflector = Reflector(
        llm_client=make_llm_recorder("止损纪律有效,应继续要求非hold决策携带机器可读证伪条件。"),
        memory_store=memory,
        log_root=log_root,
    )
    marks = reflector.reflect(now_ts=now_ts, branch="main", window=20, price_lookup=price_lookup)
    assert len(marks) == 1
    assert marks[0].thesis_status == "证伪"

    # --- L3 record written to a REAL MemoryStore, content references the lesson ---
    l3_results = memory.retrieve(probe, query_ts=now_ts, top_k=5, layers=["L3"])
    assert len(l3_results) >= 1, "expected a L3 lesson record to be retrievable right after reflection"
    lesson_content = l3_results[0][0].content
    assert "BTC/USDT:USDT" in lesson_content
    assert "price<48000" in lesson_content
    assert "47000" in lesson_content  # the concrete price that broke the thesis

    # --- Trader wired to the SAME MemoryStore retrieves the lesson AFTER it was written ---
    trader = Trader(
        llm_client=lambda p: json.dumps([{
            "ts": now_ts + 10_000,
            "symbol": "BTC/USDT:USDT",
            "action": "hold",
            "target_notional_pct": 0.0,
            "leverage": 1,
            "thesis": "占位thesis用于满足最小长度要求测试端到端L3检索链路问题",
            "falsifier": "占位falsifier用于满足最小长度要求测试端到端检索问题",
            "horizon": "4h",
        }]),
        memory_store=memory,
    )

    context_after = trader.build_context(
        positions={},
        ts=now_ts + 10_000,  # strictly AFTER the L3 write's ts (now_ts)
        latest_snapshot={},
        last_reflection_summary=None,
        program_tactics=None,
        memory_query_text=probe,
        top_k=5,
    )
    assert any(probe in item for item in context_after["memory_results"]), (
        "Trader should retrieve the L3 lesson via the same MemoryStore after it was written"
    )

    # --- M2 time boundary still holds through this new write path ---
    context_before = trader.build_context(
        positions={},
        ts=decision_ts,  # strictly BEFORE the L3 write's ts (now_ts)
        latest_snapshot={},
        last_reflection_summary=None,
        program_tactics=None,
        memory_query_text=probe,
        top_k=5,
    )
    assert not any(probe in item for item in context_before["memory_results"]), (
        "a L3 lesson written by Reflector leaked into a query with ts BEFORE its own ts "
        "-- MemoryStore.retrieve's ts <= query_ts rule must still hold through this path"
    )
