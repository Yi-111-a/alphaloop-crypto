"""
tests for LOCKED/circuit_breaker.py (§2.4 spec, M1 milestone).

Time is always injected via nav_series timestamps / explicit now_ts — never
via the real clock (the module under test must not call time.time() /
datetime.now() anywhere; these tests would be flaky/non-deterministic if it did).
"""
from __future__ import annotations

import json

import pytest

from LOCKED.circuit_breaker import CircuitBreaker
from LOCKED.log_writer import read_jsonl

CONFIG = {"constraints": {"max_drawdown_pct": 20, "daily_loss_freeze_pct": 8}}

DAY_MS = 24 * 60 * 60 * 1000
# 2024-01-01T00:00:00Z in UTC ms, a clean day boundary to build fixtures on.
T0 = 1_704_067_200_000


def make_breaker(log_root, config=None):
    return CircuitBreaker(config or CONFIG, log_root=log_root)


def test_total_drawdown_over_threshold_triggers_frozen_full(log_root):
    breaker = make_breaker(log_root)
    nav_series = [
        (T0, 100_000.0),
        (T0 + 1 * DAY_MS, 105_000.0),  # new peak
        (T0 + 2 * DAY_MS, 90_000.0),
        (T0 + 3 * DAY_MS, 80_000.0),  # drawdown from peak 105000 -> 80000 = 23.8% > 20%
    ]

    state = breaker.check(nav_series)

    assert state == "FROZEN_FULL"
    assert breaker.is_frozen() is True

    # A transition record must have been persisted to LOG/circuit_breaker_state.jsonl.
    records = read_jsonl("circuit_breaker_state.jsonl", root=log_root)
    assert len(records) == 1
    assert records[0]["new_state"] == "FROZEN_FULL"
    assert records[0]["previous_state"] == "NORMAL"


def test_single_day_decline_over_threshold_triggers_frozen_24h(log_root):
    breaker = make_breaker(log_root)
    day2_start = T0 + 1 * DAY_MS
    nav_series = [
        (T0, 100_000.0),
        (T0 + 12 * 60 * 60 * 1000, 101_000.0),  # mild move within day 1
        (day2_start, 100_000.0),  # day 2 opens at 100000 (day_start_nav)
        (day2_start + 3 * 60 * 60 * 1000, 91_500.0),  # intraday low: 8.5% decline
    ]

    state = breaker.check(nav_series)

    assert state == "FROZEN_24H"
    assert breaker.is_frozen() is True

    # total drawdown from the running peak (101000) to 91500 is < 20%, so this
    # must NOT be a FROZEN_FULL — only the daily-decline rule fired.
    records = read_jsonl("circuit_breaker_state.jsonl", root=log_root)
    assert len(records) == 1
    assert records[0]["new_state"] == "FROZEN_24H"


def test_frozen_24h_auto_expires_after_24h_with_no_further_breach(log_root):
    breaker = make_breaker(log_root)
    day2_start = T0 + 1 * DAY_MS
    trigger_series = [
        (T0, 100_000.0),
        (day2_start, 100_000.0),
        (day2_start + 3 * 60 * 60 * 1000, 91_500.0),  # 8.5% single-day decline -> FROZEN_24H
    ]
    state = breaker.check(trigger_series)
    assert state == "FROZEN_24H"
    assert breaker.is_frozen() is True
    trigger_ts = day2_start + 3 * 60 * 60 * 1000

    # Advance time 24h+ past the trigger, NAV recovered/flat, no new breach.
    later_ts = trigger_ts + DAY_MS + 1
    later_series = trigger_series + [(later_ts, 92_000.0)]

    state_after = breaker.check(later_series, now_ts=later_ts)

    assert state_after == "NORMAL"
    assert breaker.is_frozen() is False

    records = read_jsonl("circuit_breaker_state.jsonl", root=log_root)
    assert [r["new_state"] for r in records] == ["FROZEN_24H", "NORMAL"]


def test_frozen_full_does_not_auto_expire_ever(log_root):
    breaker = make_breaker(log_root)
    nav_series = [
        (T0, 100_000.0),
        (T0 + 1 * DAY_MS, 70_000.0),  # 30% drawdown -> FROZEN_FULL
    ]
    state = breaker.check(nav_series)
    assert state == "FROZEN_FULL"

    # Even a wildly-far-future now_ts / NAV recovery must NOT clear it.
    far_future_ts = T0 + 3650 * DAY_MS  # ~10 years later
    recovered_series = nav_series + [(far_future_ts, 500_000.0)]

    state_after = breaker.check(recovered_series, now_ts=far_future_ts)

    assert state_after == "FROZEN_FULL"
    assert breaker.is_frozen() is True

    # Only the original trigger should have been logged; sticky re-checks don't log.
    records = read_jsonl("circuit_breaker_state.jsonl", root=log_root)
    assert len(records) == 1


def test_mild_nav_series_stays_normal(log_root):
    breaker = make_breaker(log_root)
    nav_series = [
        (T0, 100_000.0),
        (T0 + 1 * DAY_MS, 100_800.0),
        (T0 + 2 * DAY_MS, 99_500.0),
        (T0 + 3 * DAY_MS, 101_200.0),
        (T0 + 4 * DAY_MS, 100_100.0),
    ]

    state = breaker.check(nav_series)

    assert state == "NORMAL"
    assert breaker.is_frozen() is False
    # No transition ever happened (started NORMAL, stayed NORMAL) -> nothing logged.
    records = read_jsonl("circuit_breaker_state.jsonl", root=log_root)
    assert records == []


def test_manual_unfreeze_clears_frozen_full(log_root):
    breaker = make_breaker(log_root)
    nav_series = [
        (T0, 100_000.0),
        (T0 + 1 * DAY_MS, 70_000.0),  # 30% drawdown -> FROZEN_FULL
    ]
    state = breaker.check(nav_series)
    assert state == "FROZEN_FULL"
    assert breaker.is_frozen() is True

    breaker.manual_unfreeze(now_ts=T0 + 2 * DAY_MS, note="human reviewed and approved reset")

    assert breaker.state == "NORMAL"
    assert breaker.is_frozen() is False

    # Subsequent check() with the same still-breaching-drawdown history should be
    # free to re-evaluate from NORMAL again (not stuck referencing old sticky state).
    records = read_jsonl("circuit_breaker_state.jsonl", root=log_root)
    assert [r["event"] for r in records] == ["state_transition", "manual_unfreeze"]
    assert records[-1]["new_state"] == "NORMAL"


def test_is_frozen_requires_no_arguments_and_reflects_last_check(log_root):
    """simulator.py's contract: is_frozen() must be a zero-arg, no-nav_series call."""
    breaker = make_breaker(log_root)
    import inspect

    sig = inspect.signature(breaker.is_frozen)
    assert list(sig.parameters) == []

    assert breaker.is_frozen() is False  # default fresh state is NORMAL
