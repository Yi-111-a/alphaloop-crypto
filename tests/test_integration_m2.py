"""
M2 跨模块集成校验:不 mock ColdStartGate/MemoryStore/Trader/Simulator 之间的接口
(只在最外层的 LLM 调用打桩,因为那是真实世界里唯一必须联网的部分),验证
冷启动闸门 -> 记忆检索 -> Trader决策 -> Simulator撮合的真实数据能端到端串起来,
并专门验证一次"Trader通过真实MemoryStore读取记忆"路径下的时间边界不泄漏。
"""
from __future__ import annotations

import json

from LOCKED import log_writer
from LOCKED.cold_start import ColdStartGate
from LOCKED.schemas import Decision, Rejection, Trade
from LOCKED.simulator import Simulator

from ASSET.memory.engine import MemoryStore
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


def _fake_llm_client_factory(decision_dict):
    """Returns a fake llm_client that always answers with one valid decision."""
    def _client(prompt: str) -> str:
        return json.dumps([decision_dict])
    return _client


def test_cold_start_blocks_trader_decision_end_to_end(tmp_path):
    log_root = tmp_path / "LOG"
    memory = MemoryStore(db_path=tmp_path / "memory.db")
    gate = ColdStartGate(
        genesis_path=tmp_path / "genesis.md",  # does not exist yet
        min_hypothesis_count=10,
        log_root=log_root,
        state_path=tmp_path / "state" / "cold_start_state.json",
    )
    sim = Simulator(
        config=CONFIG,
        cold_start_gate=gate.is_cold_start,
        universe_symbols=UNIVERSE,
        db_path=tmp_path / "state" / "portfolio_main.db",
        log_root=log_root,
        branch="main",
    )

    decision_dict = {
        "ts": 1_700_000_000_000,
        "symbol": "BTC/USDT:USDT",
        "action": "open_long",
        "target_notional_pct": 30.0,
        "leverage": 2,
        "thesis": "基于H1:BTC波动率处于历史低位,预期短期突破概率上升",
        "falsifier": "若价格跌破近期支撑位则本假设视为证伪",
        "horizon": "12h",
    }
    trader = Trader(llm_client=_fake_llm_client_factory(decision_dict), memory_store=memory)

    decisions = trader.decide(
        ts=1_700_000_000_000,
        positions={},
        latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}},
        memory_query_text="BTC volatility",
        branch="main",
    )
    assert len(decisions) == 1
    decision = decisions[0]
    log_writer.append_jsonl("decisions.jsonl", decision, root=log_root)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection), "COLD_START must block trading regardless of decision quality"
    assert "cold_start_active" in result.reason


def test_normal_state_trader_decision_fills_via_real_memory_retrieval(tmp_path):
    log_root = tmp_path / "LOG"
    memory = MemoryStore(db_path=tmp_path / "memory.db")

    # Seed >=10 hypotheses into memory (L2), as genesis research would.
    for i in range(1, 11):
        memory.write(
            content=f"H{i}: 假设内容占位符,用于满足冷启动假设数量门槛",
            ts=1_699_000_000_000,
            layer="L2",
        )
    # A relevant L1 memory the Trader's query should actually be able to retrieve.
    memory.write(
        content="BTC近期波动率处于历史低位,4小时线呈现收敛三角形态",
        ts=1_699_900_000_000,
        layer="L1",
    )

    genesis_path = tmp_path / "genesis.md"
    genesis_path.write_text("# Genesis\nH1..H10 hypotheses.\n", encoding="utf-8")

    gate = ColdStartGate(
        genesis_path=genesis_path,
        min_hypothesis_count=10,
        log_root=log_root,
        state_path=tmp_path / "state" / "cold_start_state.json",
    )
    state = gate.check_and_transition(hypothesis_count=10, ts=1_699_900_000_001)
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

    decision_dict = {
        "ts": 1_700_000_000_000,
        "symbol": "BTC/USDT:USDT",
        "action": "open_long",
        "target_notional_pct": 30.0,
        "leverage": 2,
        "thesis": "基于H1:BTC波动率处于历史低位,预期短期突破概率上升",
        "falsifier": "若价格跌破近期支撑位则本假设视为证伪",
        "horizon": "12h",
    }
    trader = Trader(llm_client=_fake_llm_client_factory(decision_dict), memory_store=memory)

    decisions = trader.decide(
        ts=1_700_000_000_000,
        positions={},
        latest_snapshot={"BTC/USDT:USDT": {"last": 50_000.0}},
        memory_query_text="BTC波动率",
        branch="main",
    )
    decision = decisions[0]
    log_writer.append_jsonl("decisions.jsonl", decision, root=log_root)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Trade), f"expected a fill once NORMAL, got: {result}"


def test_trader_never_sees_memory_written_after_its_own_decision_ts(tmp_path):
    """The critical M2 leakage test at the FULL wiring level: seed a memory
    record whose content is a perfect match for the query but timestamped
    AFTER the decision's own ts, and confirm it never reaches the LLM prompt
    Trader builds -- proving the query_ts wiring (Trader -> MemoryStore) is
    leak-free end-to-end, not just at each module's own unit-test boundary."""
    memory = MemoryStore(db_path=tmp_path / "memory.db")

    decision_ts = 1_700_000_000_000
    future_ts = decision_ts + 3600_000  # 1h after the decision

    query_text = "UNIQUE_FUTURE_LEAK_PROBE_STRING"
    memory.write(content=query_text, ts=future_ts, layer="L1")  # written in the "future"
    memory.write(content="正常的历史观察,时间早于决策", ts=decision_ts - 3600_000, layer="L1")

    captured_prompts = []

    def _capturing_llm_client(prompt: str) -> str:
        captured_prompts.append(prompt)
        return json.dumps([{
            "ts": decision_ts,
            "symbol": "BTC/USDT:USDT",
            "action": "hold",
            "target_notional_pct": 0.0,
            "leverage": 1,
            "thesis": "占位thesis用于满足最小长度要求测试时间边界不泄漏问题",
            "falsifier": "占位falsifier用于满足最小长度要求测试时间边界问题",
            "horizon": "4h",
        }])

    trader = Trader(llm_client=_capturing_llm_client, memory_store=memory)
    trader.decide(
        ts=decision_ts,
        positions={},
        latest_snapshot={},
        memory_query_text=query_text,
        branch="main",
    )

    assert len(captured_prompts) == 1
    assert query_text not in captured_prompts[0], (
        "a memory written AFTER the decision's own ts leaked into the Trader's prompt context"
    )
