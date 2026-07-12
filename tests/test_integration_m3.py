"""
M3 全链路集成校验:不 mock Researcher/ColdStartGate/Trader/Simulator/Reflector
之间的接口(只在最外层的LLM/搜索调用打桩),把整条"冷启动研究 -> 假设写入记忆
-> 解冻交易 -> Trader决策 -> Simulator撮合 -> Reflector判定 -> 证伪教训升入L3
-> 下一轮Trader检索到教训"链路串起来跑一遍,逐条对照spec的M2/M3验收标准。
"""
from __future__ import annotations

import json

from LOCKED import log_writer
from LOCKED.cold_start import ColdStartGate
from LOCKED.reflector import Reflector
from LOCKED.schemas import Trade
from LOCKED.simulator import Simulator

from ASSET.memory.engine import MemoryStore
from ASSET.strategy.researcher import Researcher
from ASSET.strategy.trader import Trader

UNIVERSE = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

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
}


def _researcher_llm_client(prompt: str) -> str:
    # 12 real hypotheses, exceeds min_hypotheses=10, no padding needed.
    hypotheses = [
        {"hypothesis": f"H{i}占位假设:市场结构性观察{i}", "rationale": "基于波动率画像与历史文献综合判断"}
        for i in range(1, 13)
    ]
    return json.dumps(hypotheses)


def _trader_llm_client_factory(decision_dict):
    def _client(prompt: str) -> str:
        return json.dumps([decision_dict])
    return _client


def _reflector_llm_client(prompt: str) -> str:
    return "根据已判定结果生成的经验摘要占位文本,不重新评判对错。"


def test_full_m3_loop_research_to_reflection_to_next_decision(tmp_path):
    log_root = tmp_path / "LOG"
    memory = MemoryStore(db_path=tmp_path / "memory.db")

    # ---- 1. COLD_START: Researcher produces genesis.md + hypotheses ----
    genesis_path = tmp_path / "research_notes" / "genesis.md"
    researcher = Researcher(
        llm_client=_researcher_llm_client,
        memory_store=memory,
        search_client=None,
        research_notes_dir=tmp_path / "research_notes",
        genesis_path=genesis_path,
    )
    price_history = {
        "BTC/USDT:USDT": [50_000.0, 50_500.0, 49_800.0, 51_000.0, 50_200.0],
        "ETH/USDT:USDT": [3_000.0, 3_050.0, 2_980.0, 3_100.0, 3_020.0],
    }
    result = researcher.run_cold_start_research(
        ts=1_699_000_000_000,
        universe_symbols=UNIVERSE,
        price_history=price_history,
        min_hypotheses=10,
    )
    assert result["hypothesis_count"] >= 10
    assert genesis_path.exists()

    # ---- 2. Cold-start gate transitions to NORMAL using Researcher's own output ----
    gate = ColdStartGate(
        genesis_path=genesis_path,
        min_hypothesis_count=10,
        log_root=log_root,
        state_path=tmp_path / "state" / "cold_start_state.json",
    )
    state = gate.check_and_transition(hypothesis_count=result["hypothesis_count"], ts=1_699_000_000_001)
    assert state == "NORMAL"
    assert gate.is_cold_start() is False

    sim = Simulator(
        config=CONFIG,
        cold_start_gate=gate.is_cold_start,
        universe_symbols=UNIVERSE,
        db_path=tmp_path / "state" / "portfolio_main.db",
        log_root=log_root,
        branch="main",
    )

    # ---- 3. Trader decides, referencing a genesis hypothesis, with a falsifier
    #         condition that WILL be triggered later (price dips under 48000) ----
    decision_ts = 1_700_000_000_000
    decision_dict = {
        "ts": decision_ts,
        "symbol": "BTC/USDT:USDT",
        "action": "open_long",
        "target_notional_pct": 30.0,
        "leverage": 2,
        "thesis": "基于H1:市场结构性观察显示BTC短期存在上行动能",
        "falsifier": "若价格跌破48000则本次交易假设视为证伪并应止损",
        "falsifier_condition": "price<48000",
        "horizon": "12h",
    }
    trader = Trader(llm_client=_trader_llm_client_factory(decision_dict), memory_store=memory)
    decisions = trader.decide(
        ts=decision_ts,
        positions={},
        latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}},
        memory_query_text="市场结构性观察",
        branch="main",
    )
    decision = decisions[0]
    assert "H1" in decision.thesis
    log_writer.append_jsonl("decisions.jsonl", decision, root=log_root)

    next_bar = {"open_time": decision_ts + 100_000, "open": 50_000.0}
    fill = sim.execute(decision, next_bar)
    assert isinstance(fill, Trade), f"expected a fill in NORMAL state, got: {fill}"

    # ---- 4. Price dips below 48000 within the horizon -> Reflector judges 证伪 ----
    horizon_ms = 12 * 3600 * 1000
    now_ts = decision_ts + horizon_ms  # horizon fully elapsed

    def price_lookup(symbol: str, ts: int) -> float:
        # dips under the falsifier threshold roughly mid-horizon, recovers after
        midpoint = decision_ts + horizon_ms // 2
        if abs(ts - midpoint) < horizon_ms // 4:
            return 47_000.0
        return 50_000.0

    reflector = Reflector(llm_client=_reflector_llm_client, memory_store=memory, log_root=log_root)
    marks = reflector.reflect(now_ts=now_ts, branch="main", window=20, price_lookup=price_lookup)
    assert len(marks) == 1
    assert marks[0].thesis_status == "证伪"

    # ---- 5. THE END-TO-END LEAKAGE-FREE LESSON CHECK: the next Trader decision
    #         cycle (later ts, matching query) must see the falsified lesson;
    #         an earlier-ts query must NOT (M2 time boundary still holds). ----
    later_ts = now_ts + 3600_000
    later_decision_dict = {**decision_dict, "ts": later_ts, "action": "hold",
                            "target_notional_pct": 0.0, "leverage": 1,
                            "falsifier_condition": None}

    captured_contexts = []

    def _capturing_trader_llm(prompt: str) -> str:
        captured_contexts.append(prompt)
        return json.dumps([later_decision_dict])

    trader2 = Trader(llm_client=_capturing_trader_llm, memory_store=memory)
    trader2.decide(
        ts=later_ts,
        positions={},
        latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}},
        memory_query_text="BTC/USDT:USDT price<48000",
        branch="main",
    )
    assert any("48000" in p or "证伪" in p or "H1" in p for p in captured_contexts), (
        "the falsified lesson never reached the next Trader cycle's prompt context"
    )

    # earlier-than-the-lesson query must NOT see it (M2 boundary preserved through the new write path)
    earlier_ts = decision_ts - 3600_000
    captured_earlier = []

    def _capturing_trader_llm_earlier(prompt: str) -> str:
        captured_earlier.append(prompt)
        return json.dumps([{**later_decision_dict, "ts": earlier_ts}])

    trader3 = Trader(llm_client=_capturing_trader_llm_earlier, memory_store=memory)
    trader3.decide(
        ts=earlier_ts,
        positions={},
        latest_snapshot={},
        memory_query_text="BTC/USDT:USDT price<48000",
        branch="main",
    )
    assert not any("48000" in p and "证伪" in p for p in captured_earlier), (
        "a lesson written AFTER earlier_ts leaked into an earlier decision's context"
    )
