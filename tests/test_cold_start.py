from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

import pytest
import yaml

from LOCKED import log_writer
from LOCKED.cold_start import ColdStartGate
from LOCKED.schemas import Rejection, Trade
from LOCKED.simulator import Simulator

from tests.test_simulator import UNIVERSE, load_config, md  # reuse established helpers


def make_gate(tmp_path, genesis_exists=False, state_path=None, log_root=None):
    genesis_path = tmp_path / "ASSET" / "research_notes" / "genesis.md"
    if genesis_exists:
        genesis_path.parent.mkdir(parents=True, exist_ok=True)
        genesis_path.write_text("genesis content", encoding="utf-8")
    # Always sandbox the state-transition log under tmp_path unless the
    # caller explicitly supplied one (e.g. the shared `log_root` fixture) --
    # never let a test fall back to LOCKED.log_writer's real project LOG/
    # directory default.
    effective_log_root = log_root if log_root is not None else tmp_path / "LOG"
    gate = ColdStartGate(
        genesis_path=genesis_path,
        state_path=state_path if state_path is not None else tmp_path / "cold_start_state.json",
        log_root=effective_log_root,
    )
    return gate, genesis_path


# ---------------------------------------------------------------------------
# 1. Fresh gate defaults to COLD_START
# ---------------------------------------------------------------------------


def test_fresh_gate_is_cold_start(tmp_path):
    gate, _ = make_gate(tmp_path, genesis_exists=False)
    assert gate.is_cold_start() is True
    assert gate.state == "COLD_START"


# ---------------------------------------------------------------------------
# 2. Real Simulator wiring: COLD_START rejects an otherwise-valid decision
# ---------------------------------------------------------------------------


def test_wired_simulator_rejects_trade_during_cold_start(tmp_path, log_root, make_decision):
    gate, _ = make_gate(tmp_path, genesis_exists=False)
    cfg = load_config()
    sim = Simulator(
        config=cfg,
        cold_start_gate=gate.is_cold_start,
        universe_symbols=UNIVERSE,
        db_path=tmp_path / "portfolio_cold_gate.db",
        branch="cold-gate-test",
        log_root=log_root,
    )
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=10.0)
    log_writer.append_jsonl("decisions.jsonl", decision, root=log_root)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "cold_start_active" in result.reason


# ---------------------------------------------------------------------------
# 3. Below-threshold hypothesis count -> stays COLD_START
# ---------------------------------------------------------------------------


def test_below_threshold_hypothesis_count_stays_cold_start(tmp_path):
    gate, _ = make_gate(tmp_path, genesis_exists=True)
    state = gate.check_and_transition(hypothesis_count=9)
    assert state == "COLD_START"
    assert gate.is_cold_start() is True


# ---------------------------------------------------------------------------
# 4. Exactly-threshold hypothesis count with genesis.md present -> transitions
#    to NORMAL; re-wired Simulator no longer rejects for cold_start_active.
# ---------------------------------------------------------------------------


def test_exact_threshold_transitions_to_normal_and_unblocks_simulator(tmp_path, log_root, make_decision):
    gate, _ = make_gate(tmp_path, genesis_exists=True)
    state = gate.check_and_transition(hypothesis_count=10)
    assert state == "NORMAL"
    assert gate.is_cold_start() is False

    cfg = load_config()
    sim = Simulator(
        config=cfg,
        cold_start_gate=gate.is_cold_start,
        universe_symbols=UNIVERSE,
        db_path=tmp_path / "portfolio_post_cold.db",
        branch="post-cold-test",
        log_root=log_root,
    )
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=10.0)
    log_writer.append_jsonl("decisions.jsonl", decision, root=log_root)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    # Must not be rejected for cold_start_active anymore; here it should
    # actually fill since the decision is otherwise valid.
    if isinstance(result, Rejection):
        assert "cold_start_active" not in result.reason
    else:
        assert isinstance(result, Trade)


# ---------------------------------------------------------------------------
# 5. hypothesis_count sufficient but genesis.md missing -> stays COLD_START
#    (both conditions required, not either/or).
# ---------------------------------------------------------------------------


def test_missing_genesis_stays_cold_start_even_with_enough_hypotheses(tmp_path):
    gate, genesis_path = make_gate(tmp_path, genesis_exists=False)
    assert not genesis_path.exists()
    state = gate.check_and_transition(hypothesis_count=15)
    assert state == "COLD_START"
    assert gate.is_cold_start() is True


# ---------------------------------------------------------------------------
# 6. One-directional: NORMAL never reverts, even if inputs regress.
# ---------------------------------------------------------------------------


def test_normal_does_not_revert(tmp_path):
    gate, _ = make_gate(tmp_path, genesis_exists=True)
    assert gate.check_and_transition(hypothesis_count=10) == "NORMAL"

    # Pretend the hypothesis count somehow dropped back to 0.
    state = gate.check_and_transition(hypothesis_count=0)
    assert state == "NORMAL"
    assert gate.is_cold_start() is False


# ---------------------------------------------------------------------------
# 7. Restart/persistence: a fresh gate pointed at the same state_path recovers
#    NORMAL immediately, without another check_and_transition() call.
# ---------------------------------------------------------------------------


def test_state_persists_across_restart(tmp_path):
    state_path = tmp_path / "cold_start_state.json"
    gate, _ = make_gate(tmp_path, genesis_exists=True, state_path=state_path)
    assert gate.check_and_transition(hypothesis_count=10) == "NORMAL"

    # Fresh instance, same state_path, same (now-irrelevant) genesis_path.
    gate2 = ColdStartGate(genesis_path=tmp_path / "ASSET" / "research_notes" / "genesis.md", state_path=state_path)
    assert gate2.is_cold_start() is False
    assert gate2.state == "NORMAL"


def test_fresh_state_path_defaults_to_cold_start(tmp_path):
    """Sanity check: a state_path that has never been written to yields COLD_START."""
    state_path = tmp_path / "never_written.json"
    gate = ColdStartGate(genesis_path=tmp_path / "genesis.md", state_path=state_path)
    assert gate.is_cold_start() is True


# ---------------------------------------------------------------------------
# 8. No wall-clock calls in the core decision logic.
# ---------------------------------------------------------------------------


def test_no_wall_clock_calls_in_check_and_transition():
    # Inspect only the executable body of the method (not its docstring,
    # which is allowed to mention time.time()/datetime.now() in prose
    # explaining why they're prohibited -- only actual calls are forbidden).
    import ast

    source = inspect.getsource(ColdStartGate.check_and_transition)
    tree = ast.parse(textwrap.dedent(source))
    func_node = tree.body[0]
    body = func_node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # drop the docstring node
    body_source = "\n".join(ast.unparse(stmt) for stmt in body)

    forbidden = ["time.time(", "datetime.now(", "datetime.utcnow("]
    for token in forbidden:
        assert token not in body_source, f"found forbidden wall-clock call: {token}"

    # Also confirm the module doesn't import time/datetime at all -- there is
    # no legitimate reason for this module's core logic to reach for the
    # system clock.
    import LOCKED.cold_start as cold_start_module

    module_source = inspect.getsource(cold_start_module)
    assert "import time" not in module_source
    assert "import datetime" not in module_source
    assert "from datetime" not in module_source


# ---------------------------------------------------------------------------
# Extra: log record written on transition (append-only, via log_writer)
# ---------------------------------------------------------------------------


def test_transition_is_logged_via_log_writer(tmp_path, log_root):
    gate, _ = make_gate(tmp_path, genesis_exists=True, log_root=log_root)
    gate.check_and_transition(hypothesis_count=10, ts=1_700_000_000_000)

    records = log_writer.read_jsonl("cold_start_state.jsonl", root=log_root)
    assert len(records) == 1
    assert records[0]["from_state"] == "COLD_START"
    assert records[0]["to_state"] == "NORMAL"
    assert records[0]["ts"] == 1_700_000_000_000
    assert records[0]["hypothesis_count"] == 10

    # A subsequent no-op transition call must not append another record.
    gate.check_and_transition(hypothesis_count=0)
    records_after = log_writer.read_jsonl("cold_start_state.jsonl", root=log_root)
    assert len(records_after) == 1
