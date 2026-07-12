from __future__ import annotations

import dataclasses

import pytest

from LOCKED.baseline_agents import BTCHoldAgent, RandomAgent
from LOCKED.schemas import Decision

UNIVERSE = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]

VALID_ACTIONS = {"open_long", "open_short", "close", "adjust", "hold"}

BASE_TS = 1_700_000_000_000
CYCLE_MS = 4 * 60 * 60 * 1000  # 4h decision cycle, matches horizon="4h"


def _ts_sequence(n: int, start: int = BASE_TS) -> list[int]:
    return [start + i * CYCLE_MS for i in range(n)]


class TestRandomAgentReproducibility:
    def test_same_seed_same_sequence(self):
        ts_seq = _ts_sequence(200)

        agent_a = RandomAgent(UNIVERSE, seed=42)
        agent_b = RandomAgent(UNIVERSE, seed=42)

        decisions_a = [agent_a.decide(ts) for ts in ts_seq]
        decisions_b = [agent_b.decide(ts) for ts in ts_seq]

        assert len(decisions_a) == len(decisions_b) == 200
        for da, db in zip(decisions_a, decisions_b):
            assert dataclasses.asdict(da) == dataclasses.asdict(db)

    def test_different_seed_diverges(self):
        # Sanity check that the reproducibility test isn't trivially true because
        # the agent ignores the seed entirely.
        ts_seq = _ts_sequence(200)

        agent_a = RandomAgent(UNIVERSE, seed=42)
        agent_c = RandomAgent(UNIVERSE, seed=1234)

        decisions_a = [agent_a.decide(ts) for ts in ts_seq]
        decisions_c = [agent_c.decide(ts) for ts in ts_seq]

        assert [dataclasses.asdict(d) for d in decisions_a] != [
            dataclasses.asdict(d) for d in decisions_c
        ]


class TestRandomAgentHoldTradeSplit:
    def test_roughly_80_20_split_with_fixed_seed(self):
        n = 1000
        agent = RandomAgent(UNIVERSE, seed=42)
        decisions = [agent.decide(ts) for ts in _ts_sequence(n)]

        hold_count = sum(1 for d in decisions if d.action == "hold")
        hold_frac = hold_count / n

        # Deterministic (fixed seed) but we still use a generous statistical band
        # (75%-85%) rather than asserting an exact count, per the task spec.
        assert 0.75 <= hold_frac <= 0.85, f"hold fraction {hold_frac} out of band"

    def test_trade_decisions_use_open_long_or_open_short(self):
        n = 1000
        agent = RandomAgent(UNIVERSE, seed=42)
        decisions = [agent.decide(ts) for ts in _ts_sequence(n)]

        trade_decisions = [d for d in decisions if d.action != "hold"]
        assert len(trade_decisions) > 0
        for d in trade_decisions:
            assert d.action in {"open_long", "open_short"}


class TestRandomAgentTargetPctBounds:
    def test_trade_pct_within_5_to_25(self):
        n = 1000
        agent = RandomAgent(UNIVERSE, seed=42)
        decisions = [agent.decide(ts) for ts in _ts_sequence(n)]

        trade_decisions = [d for d in decisions if d.action != "hold"]
        assert len(trade_decisions) > 0
        for d in trade_decisions:
            assert 5.0 <= d.target_notional_pct <= 25.0


class TestRandomAgentDecisionValidity:
    def test_thesis_and_falsifier_non_empty_and_min_length(self):
        n = 500
        agent = RandomAgent(UNIVERSE, seed=42)
        decisions = [agent.decide(ts) for ts in _ts_sequence(n)]

        for d in decisions:
            assert isinstance(d.thesis, str) and len(d.thesis) >= 20
            assert isinstance(d.falsifier, str) and len(d.falsifier) >= 20

    def test_action_is_valid_literal(self):
        n = 500
        agent = RandomAgent(UNIVERSE, seed=42)
        decisions = [agent.decide(ts) for ts in _ts_sequence(n)]

        for d in decisions:
            assert d.action in VALID_ACTIONS

    def test_decisions_are_decision_instances(self):
        agent = RandomAgent(UNIVERSE, seed=42)
        d = agent.decide(BASE_TS)
        assert isinstance(d, Decision)

    def test_requires_non_empty_universe(self):
        with pytest.raises(ValueError):
            RandomAgent([], seed=42)


class TestBTCHoldAgent:
    """BTCHoldAgent is the 'ruler', not a 'player': it is deliberately exempt
    from the perp margin system (see the human ruling captured in
    baseline_agents.py's module docstring) and computed analytically instead
    of going through Simulator.execute(). These tests exercise the analytic
    nav(price) interface directly, hand-computing the expected value against
    the raw BTC price ratio rather than trusting the implementation."""

    CAPITAL = 100_000.0
    TAKER_PCT = 0.0005
    SLIPPAGE_BPS = 15

    def _agent(self, **overrides):
        kwargs = dict(capital_usdt=self.CAPITAL, taker_pct=self.TAKER_PCT, slippage_bps=self.SLIPPAGE_BPS)
        kwargs.update(overrides)
        return BTCHoldAgent(**kwargs)

    def test_before_entry_nav_is_capital(self):
        agent = self._agent()
        assert agent.entered is False
        assert agent.nav(price=50_000.0) == pytest.approx(self.CAPITAL)

    def test_enter_can_only_be_called_once(self):
        agent = self._agent()
        agent.enter(entry_price=50_000.0)
        assert agent.entered is True
        with pytest.raises(RuntimeError):
            agent.enter(entry_price=51_000.0)

    def test_nav_matches_hand_calculated_price_ratio_with_entry_cost(self):
        """The core exemption-correctness check: nav(t) must track
        capital * (1 - entry_cost_pct) * (price_t / price_entry) EXACTLY,
        with the one-time entry cost charged once and never again -- no
        per-step fee, no funding, no margin floor anywhere in the curve."""
        entry_price = 50_000.0
        agent = self._agent()
        agent.enter(entry_price=entry_price)

        entry_cost_pct = self.TAKER_PCT + self.SLIPPAGE_BPS / 1e4
        effective_capital = self.CAPITAL * (1 - entry_cost_pct)

        price_path = [50_000.0, 55_000.0, 45_000.0, 60_000.0, 50_000.0, 100_000.0, 25_000.0]
        for price in price_path:
            expected_nav = effective_capital * (price / entry_price)
            assert agent.nav(price) == pytest.approx(expected_nav)
            # equivalently, directly against the raw price ratio (what a human
            # hand-checking the benchmark would compute on a calculator):
            hand_calc = self.CAPITAL * (1 - entry_cost_pct) * (price / entry_price)
            assert agent.nav(price) == pytest.approx(hand_calc)

    def test_entry_cost_charged_exactly_once_not_per_call(self):
        agent = self._agent()
        agent.enter(entry_price=50_000.0)
        # calling nav() many times at the entry price must always return the
        # same value -- no repeated fee/slippage deduction hiding in nav().
        navs_at_entry = [agent.nav(50_000.0) for _ in range(50)]
        assert len(set(round(v, 8) for v in navs_at_entry)) == 1
        assert navs_at_entry[0] < self.CAPITAL  # entry cost was deducted exactly once
        assert navs_at_entry[0] == pytest.approx(self.CAPITAL * (1 - (self.TAKER_PCT + self.SLIPPAGE_BPS / 1e4)))

    def test_from_config_reads_capital_and_fees(self):
        config = {"capital_usdt": 100_000, "fees": {"taker_pct": 0.0005, "slippage_bps": 15}}
        agent = BTCHoldAgent.from_config(config)
        agent.enter(entry_price=50_000.0)
        assert agent.nav(50_000.0) == pytest.approx(100_000 * (1 - 0.0005 - 15 / 1e4))

    def test_does_not_produce_decision_objects_or_require_simulator(self):
        """BTCHoldAgent has no .decide() / Decision-producing surface at all --
        it never enters the Simulator's execute() validation chain, which is
        the entire point of the exemption."""
        assert not hasattr(BTCHoldAgent, "decide")
