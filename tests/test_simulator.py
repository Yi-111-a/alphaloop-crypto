from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from LOCKED import log_writer
from LOCKED.schemas import Decision, Rejection, Trade
from LOCKED.simulator import Simulator

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


UNIVERSE = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
]

# NOTE: the shared tests/conftest.py `make_decision` fixture's default
# falsifier text is only 18 Python characters long (len()), one short of the
# >= 20 character minimum mandated by schemas.Decision / spec §2.2 check 3.
# We don't modify the shared conftest.py fixture (other agents' modules may
# depend on its exact shape); instead this local wrapper pads the falsifier
# with a definitely-long-enough default whenever the caller doesn't already
# override it, so tests that aren't specifically exercising the
# thesis/falsifier-length rejection path get a decision that clears check 3.
_LONG_FALSIFIER = "若4小时K线收盘价跌破前期关键低点支撑位,则本次交易假设视为证伪并应止损离场"


def md(make_decision, **overrides):
    overrides.setdefault("falsifier", _LONG_FALSIFIER)
    return make_decision(**overrides)


def make_sim(tmp_path, log_root, branch="main", universe=None, config=None, circuit_breaker=None):
    cfg = config if config is not None else load_config()
    return Simulator(
        config=cfg,
        circuit_breaker=circuit_breaker,
        universe_symbols=UNIVERSE if universe is None else universe,
        db_path=tmp_path / f"portfolio_{branch.replace('/', '_')}.db",
        branch=branch,
        log_root=log_root,
    )


def log_and_get(log_root, decision: Decision) -> Decision:
    """Pre-register a decision to decisions.jsonl, mimicking the real production
    flow where the framework appends the decision before calling execute()."""
    log_writer.append_jsonl("decisions.jsonl", decision, root=log_root)
    return decision


# ---------------------------------------------------------------------------
# 1. Manual fee / slippage arithmetic
# ---------------------------------------------------------------------------


def test_fee_and_slippage_arithmetic(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)

    decision = md(make_decision, 
        ts=1_700_000_000_000,
        symbol="BTC/USDT:USDT",
        action="open_long",
        target_notional_pct=50.0,
        leverage=3,
    )
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    # no volume_24h_usdt provided -> base slippage assumed (documented assumption)

    trade = sim.execute(decision, next_bar)

    assert isinstance(trade, Trade)

    # hand computation
    nav_pre = 100_000.0  # config.capital_usdt, no prior positions
    expected_notional = 0.50 * nav_pre  # 50000
    expected_slippage_bps = 15  # base, since no volume given
    expected_price = 50_000.0 * (1 + expected_slippage_bps / 1e4 * 1)  # buy => sign +1
    expected_fee = expected_notional * 0.0005  # taker_pct

    assert trade.notional == pytest.approx(expected_notional)
    assert trade.price == pytest.approx(expected_price)
    assert trade.fee == pytest.approx(expected_fee)
    assert trade.slippage_bps == expected_slippage_bps
    assert trade.side == "long"
    assert trade.leverage == 3

    # trade was persisted to LOG/trades.jsonl
    records = log_writer.read_jsonl("trades.jsonl", root=log_root)
    assert len(records) == 1
    assert records[0]["price"] == pytest.approx(expected_price)


def test_slippage_doubles_for_low_volume_symbol(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=10.0, leverage=2)
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 1_000.0, "volume_24h_usdt": 100e6}
    trade = sim.execute(decision, next_bar)

    assert isinstance(trade, Trade)
    assert trade.slippage_bps == 30  # doubled from base 15
    expected_price = 1_000.0 * (1 + 30 / 1e4 * 1)
    assert trade.price == pytest.approx(expected_price)


def test_next_bar_accepts_object_with_attributes(tmp_path, log_root, make_decision):
    """next_bar can be a dict-or-object; verify attribute access fallback works."""
    sim = make_sim(tmp_path, log_root)
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=10.0, leverage=2)
    log_and_get(log_root, decision)

    next_bar = SimpleNamespace(open_time=1_700_000_100_000, open=1_000.0)
    trade = sim.execute(decision, next_bar)
    assert isinstance(trade, Trade)


# ---------------------------------------------------------------------------
# 2. Funding settlement
# ---------------------------------------------------------------------------


def _open_position(sim, log_root, make_decision, symbol, action, ts, open_price, leverage=3, pct=50.0):
    decision = md(make_decision, 
        ts=ts, symbol=symbol, action=action, target_notional_pct=pct, leverage=leverage
    )
    log_and_get(log_root, decision)
    next_bar = {"open_time": ts + 1_000, "open": open_price}
    trade = sim.execute(decision, next_bar)
    assert isinstance(trade, Trade), f"expected Trade, got Rejection: {trade}"
    return trade


@pytest.mark.parametrize(
    "action,rate",
    [
        ("open_long", 0.0004),
        ("open_long", -0.0004),
        ("open_short", 0.0004),
        ("open_short", -0.0004),
    ],
)
def test_funding_settlement_sign_and_amount(tmp_path, log_root, make_decision, action, rate):
    sim = make_sim(tmp_path, log_root, branch=f"br-{action}-{rate}")
    trade = _open_position(sim, log_root, make_decision, "BTC/USDT:USDT", action, 1_700_000_000_000, 50_000.0)

    notional = trade.notional
    ts_utc = 1_700_028_800_000  # some funding settlement instant

    def lookup(symbol, ts):
        assert symbol == "BTC/USDT:USDT"
        assert ts == ts_utc
        return rate

    settlements = sim.settle_funding(ts_utc, lookup)
    assert len(settlements) == 1
    s = settlements[0]

    if action == "open_long":
        expected_amount = notional * rate
    else:
        expected_amount = -notional * rate

    assert s.amount == pytest.approx(expected_amount)
    assert s.funding_rate == rate
    assert s.notional == pytest.approx(notional)

    # wallet_balance should be reduced by exactly the (positive-signed) amount
    portfolio = sim.get_portfolio()
    # wallet started at 100000, minus fee already paid, minus funding amount
    fee_paid = notional * 0.0005
    expected_wallet = 100_000.0 - fee_paid - expected_amount
    assert portfolio["wallet_balance"] == pytest.approx(expected_wallet)

    records = log_writer.read_jsonl("funding.jsonl", root=log_root)
    assert len(records) == 1
    assert records[0]["amount"] == pytest.approx(expected_amount)


def test_funding_settlement_crash_after_log_write_before_state_persist_recovers_exactly_once(
    tmp_path, log_root, make_decision
):
    """M5 crash-recovery acceptance test: kill the process AFTER funding.jsonl
    is written but BEFORE the sqlite state (wallet_balance deduction +
    applied_funding_settlements marker) is persisted, then restart. The
    settlement must be applied EXACTLY ONCE -- not zero times (money silently
    lost) and not twice (double-deducted). Recovery must also NOT re-fetch a
    fresh funding rate; it must reuse the rate/amount already committed to the
    log, since a live rate lookup at a different time is not guaranteed to
    match what was already promised on disk."""
    db_path = tmp_path / "crash_test.db"
    cfg = load_config()
    sim = Simulator(config=cfg, universe_symbols=UNIVERSE, db_path=db_path, branch="crash-test", log_root=log_root)
    trade = _open_position(sim, log_root, make_decision, "BTC/USDT:USDT", "open_long", 1_700_000_000_000, 50_000.0)
    wallet_before_settlement = sim.wallet_balance
    notional = trade.notional
    ts_utc = 1_700_028_800_000
    rate = 0.0004
    expected_amount = notional * rate  # long, positive rate -> account is debited

    # --- Simulate the crash: funding.jsonl gets the settlement record written,
    #     but the process dies before wallet_balance/sqlite ever see it. We
    #     reach into log_writer directly (bypassing settle_funding()) to
    #     construct exactly that half-committed state.
    from LOCKED.schemas import FundingSettlement

    crashed_settlement = FundingSettlement(
        ts=ts_utc, symbol="BTC/USDT:USDT", branch="crash-test",
        side="long", notional=notional, funding_rate=rate, amount=expected_amount,
    )
    log_writer.append_jsonl("funding.jsonl", crashed_settlement, root=log_root)
    # wallet_balance was NEVER deducted and NEVER persisted -- this is the crash.
    assert sim.wallet_balance == pytest.approx(wallet_before_settlement)

    # --- "Restart": a fresh Simulator instance resumes from the same db.
    sim_restarted = Simulator(
        config=cfg, universe_symbols=UNIVERSE, db_path=db_path, branch="crash-test",
        log_root=log_root, resume=True,
    )
    assert sim_restarted.wallet_balance == pytest.approx(wallet_before_settlement), (
        "sanity: the crash really did leave wallet_balance unsettled in sqlite"
    )

    def lookup_must_not_be_called(symbol, ts):
        raise AssertionError(
            "funding_rate_lookup was called for a settlement that was already "
            "logged pre-crash -- recovery must reuse the logged rate, not refetch"
        )

    settlements = sim_restarted.settle_funding(ts_utc, lookup_must_not_be_called)
    assert len(settlements) == 1
    assert settlements[0].amount == pytest.approx(expected_amount)
    assert sim_restarted.wallet_balance == pytest.approx(wallet_before_settlement - expected_amount)

    # No duplicate audit record was written for the crash-recovered settlement.
    records = log_writer.read_jsonl("funding.jsonl", root=log_root)
    assert len(records) == 1

    # --- Calling settle_funding AGAIN (e.g. the scheduler retries, or a SECOND
    #     restart happens) must be a pure no-op now that it's fully committed.
    def lookup_must_still_not_be_called(symbol, ts):
        raise AssertionError("settlement already fully applied -- must not be reprocessed at all")

    second_call_settlements = sim_restarted.settle_funding(ts_utc, lookup_must_still_not_be_called)
    assert second_call_settlements == []
    assert sim_restarted.wallet_balance == pytest.approx(wallet_before_settlement - expected_amount)
    records_after_second_call = log_writer.read_jsonl("funding.jsonl", root=log_root)
    assert len(records_after_second_call) == 1, "no duplicate log entry from the idempotent no-op call"

    # --- And a THIRD, completely fresh process restart must also see it as done.
    sim_restarted_again = Simulator(
        config=cfg, universe_symbols=UNIVERSE, db_path=db_path, branch="crash-test",
        log_root=log_root, resume=True,
    )
    assert sim_restarted_again.wallet_balance == pytest.approx(wallet_before_settlement - expected_amount)


# ---------------------------------------------------------------------------
# 3. Liquidation by wick (high/low, not close)
# ---------------------------------------------------------------------------


def test_liquidation_triggered_by_wick_low_even_if_close_recovers(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root, branch="liq-test")
    entry_price = 50_000.0
    trade = _open_position(
        sim, log_root, make_decision, "BTC/USDT:USDT", "open_long", 1_700_000_000_000,
        entry_price, leverage=10, pct=50.0,
    )
    assert trade.leverage == 10

    pos = sim.positions["BTC/USDT:USDT"]
    mmr = 0.005  # BTC
    margin_over_notional = pos.margin / pos.notional  # ~ 1/leverage (approx, entry==fill price here)
    p_liq = pos.entry_price * (1 + mmr - margin_over_notional)

    # Bar wicks below p_liq intrabar but closes back above it.
    low_price = p_liq - 50.0
    assert low_price < p_liq
    close_price = p_liq + 500.0
    assert close_price > p_liq

    mark_prices = {
        "BTC/USDT:USDT": {"high": entry_price + 100, "low": low_price, "close": close_price}
    }

    events = sim.check_liquidation(mark_prices, ts_utc=1_700_010_000_000)

    assert len(events) == 1
    event = events[0]
    assert event.symbol == "BTC/USDT:USDT"
    assert event.margin_lost == pytest.approx(pos.margin)
    assert event.liquidation_price == pytest.approx(p_liq)

    # position removed, margin lost, branch marked dead
    assert "BTC/USDT:USDT" not in sim.positions
    assert sim.branch_dead is True

    portfolio = sim.get_portfolio()
    assert portfolio["branch_dead"] is True
    assert len(portfolio["positions"]) == 0

    records = log_writer.read_jsonl("liquidations.jsonl", root=log_root)
    assert len(records) == 1


def test_liquidation_not_triggered_if_wick_does_not_breach(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root, branch="liq-safe")
    entry_price = 50_000.0
    _open_position(
        sim, log_root, make_decision, "BTC/USDT:USDT", "open_long", 1_700_000_000_000,
        entry_price, leverage=10, pct=50.0,
    )
    pos = sim.positions["BTC/USDT:USDT"]
    mmr = 0.005
    margin_over_notional = pos.margin / pos.notional
    p_liq = pos.entry_price * (1 + mmr - margin_over_notional)

    mark_prices = {
        "BTC/USDT:USDT": {"high": entry_price + 100, "low": p_liq + 10.0, "close": entry_price}
    }
    events = sim.check_liquidation(mark_prices, ts_utc=1_700_010_000_000)
    assert events == []
    assert sim.branch_dead is False
    assert "BTC/USDT:USDT" in sim.positions


# ---------------------------------------------------------------------------
# 4. leverage > 10 -> Rejection
# ---------------------------------------------------------------------------


def test_leverage_over_max_is_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)
    decision = md(make_decision, ts=1_700_000_000_000, leverage=11, target_notional_pct=10.0)
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "leverage" in result.reason.lower()

    records = log_writer.read_jsonl("rejections.jsonl", root=log_root)
    assert len(records) == 1


# ---------------------------------------------------------------------------
# 5. total notional over 300% of NAV -> Rejection
# ---------------------------------------------------------------------------


def test_total_notional_over_limit_is_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root, branch="total-notional-test")
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    base_ts = 1_700_000_000_000

    # Open 3 positions at high leverage (small margin footprint) so that the
    # free-margin check does not bind before the total-notional check does.
    for i, sym in enumerate(symbols):
        ts = base_ts + i * 10_000
        decision = md(make_decision, 
            ts=ts, symbol=sym, action="open_long", target_notional_pct=90.0, leverage=10
        )
        log_and_get(log_root, decision)
        next_bar = {"open_time": ts + 1_000, "open": 1_000.0}
        trade = sim.execute(decision, next_bar)
        assert isinstance(trade, Trade), f"expected Trade for {sym}, got {trade}"

    # 4th position on a new symbol should push total notional over 300% of NAV.
    ts4 = base_ts + 40_000
    decision4 = md(make_decision, 
        ts=ts4, symbol="XRP/USDT:USDT", action="open_long", target_notional_pct=90.0, leverage=10
    )
    log_and_get(log_root, decision4)
    next_bar4 = {"open_time": ts4 + 1_000, "open": 1.0}
    result = sim.execute(decision4, next_bar4)

    assert isinstance(result, Rejection)
    assert "total_notional" in result.reason


# ---------------------------------------------------------------------------
# 6. decision.ts >= next_bar.open_time -> Rejection (future peeking)
# ---------------------------------------------------------------------------


def test_future_peeking_is_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)
    decision = md(make_decision, ts=1_700_000_200_000, target_notional_pct=10.0)
    log_and_get(log_root, decision)

    # next_bar.open_time is BEFORE decision.ts
    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "future_peeking" in result.reason


def test_ts_equal_to_open_time_is_also_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)
    decision = md(make_decision, ts=1_700_000_100_000, target_notional_pct=10.0)
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}  # equal, not strictly earlier
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "future_peeking" in result.reason


# ---------------------------------------------------------------------------
# 7. decision not pre-logged -> Rejection
# ---------------------------------------------------------------------------


def test_unlogged_decision_is_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=10.0)
    # NOTE: deliberately NOT calling log_and_get() / append_jsonl here.

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "decision_not_logged" in result.reason


# ---------------------------------------------------------------------------
# 8. thesis / falsifier missing or too short -> Rejection
# ---------------------------------------------------------------------------


def test_short_thesis_is_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=10.0, thesis="too short")
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "thesis_or_falsifier_invalid" in result.reason


def test_empty_falsifier_is_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=10.0, falsifier="")
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "thesis_or_falsifier_invalid" in result.reason


# ---------------------------------------------------------------------------
# 9. symbol not in universe_symbols -> Rejection
# ---------------------------------------------------------------------------


def test_symbol_not_in_universe_is_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root, universe=["BTC/USDT:USDT"])
    decision = md(make_decision, 
        ts=1_700_000_000_000, symbol="DOGE/USDT:USDT", target_notional_pct=10.0
    )
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 0.1}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "symbol_not_in_universe" in result.reason


# ---------------------------------------------------------------------------
# Extra: per-symbol notional cap, free-margin cap, circuit breaker, resume
# ---------------------------------------------------------------------------


def test_position_notional_over_limit_is_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)
    # max_position_notional_pct = 100% of NAV; ask for 150%.
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=150.0, leverage=10)
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "position_notional_exceeds_limit" in result.reason


def test_exact_100pct_notional_decision_fills_and_satisfies_all_constraints(tmp_path, log_root, make_decision):
    """Regression test for the pre/post-fee NAV-basis bug: a decision sized at
    EXACTLY max_position_notional_pct (100% of NAV) must still fill, not be
    rejected by its own trading fee. Position sizing (target_notional_pct% of
    pre-trade NAV) and the position/total-notional cap check must share the
    same NAV basis (pre-trade), or any decision that hugs the cap will forever
    be rejected by an amount equal to its own fee. leverage=2 keeps margin
    (50% of NAV) comfortably clear of the separate min_free_margin_pct(15%)
    floor, isolating this test to the notional-cap-vs-fee-basis bug only."""
    sim = make_sim(tmp_path, log_root, branch="exact-cap-test")
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=100.0, leverage=2)
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Trade), f"expected a fill at exactly the notional cap, got: {result}"

    # Notional/total caps are entry-time sizing gates: sizing and the cap check
    # both key off the SAME pre-trade NAV basis (nav_pre = capital_usdt here,
    # no prior positions). They are not a continuously-enforced invariant
    # against the post-fee marked NAV -- a fee nibbling free collateral after
    # entry doesn't retroactively make an already-sized position "too big";
    # that's what the separate maintenance-margin/liquidation check is for.
    nav_pre = 100_000.0  # config.capital_usdt, no prior positions
    pos = sim.positions["BTC/USDT:USDT"]
    assert abs(pos.notional) <= 1.00 * nav_pre + 1e-6, "position notional must not exceed 100% of pre-trade NAV"
    total_notional = sum(abs(p.notional) for p in sim.positions.values())
    assert total_notional <= 3.00 * nav_pre + 1e-6, "total notional must not exceed 300% of pre-trade NAV"

    # Free-margin-ratio (constraint 8) legitimately IS checked against the
    # post-fee marked NAV -- confirm it's still satisfied, since leverage=2
    # was deliberately chosen to keep this test isolated to the notional-cap
    # bug rather than tripping the separate free-margin floor.
    portfolio = sim.get_portfolio({"BTC/USDT:USDT": result.price})
    nav_post = portfolio["nav"]
    total_margin = sum(p.margin for p in sim.positions.values())
    free_margin_ratio = (nav_post - total_margin) / nav_post
    assert free_margin_ratio >= 0.15 - 1e-9, "free margin ratio must stay >= min_free_margin_pct"
    assert sim.branch_dead is False


def test_free_margin_below_minimum_is_rejected(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root)
    # leverage=1 => margin == notional. target 90% notional at 1x leverage means
    # 90% of NAV locked as margin, leaving only 10% free < min_free_margin_pct(15%).
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=90.0, leverage=1)
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "free_margin_below_minimum" in result.reason


def test_cold_start_gate_rejects_trade(tmp_path, log_root, make_decision):
    """M2 contract: cold_start_gate is checked as step 0, before anything else."""
    cfg = load_config()
    sim = Simulator(
        config=cfg,
        cold_start_gate=lambda: True,
        universe_symbols=UNIVERSE,
        db_path=tmp_path / "portfolio_cold.db",
        branch="cold-start-test",
        log_root=log_root,
    )
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=10.0)
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "cold_start_active" in result.reason


def test_circuit_breaker_frozen_rejects_trade(tmp_path, log_root, make_decision):
    frozen_breaker = SimpleNamespace(is_frozen=lambda: True)
    sim = make_sim(tmp_path, log_root, circuit_breaker=frozen_breaker)
    decision = md(make_decision, ts=1_700_000_000_000, target_notional_pct=10.0)
    log_and_get(log_root, decision)

    next_bar = {"open_time": 1_700_000_100_000, "open": 50_000.0}
    result = sim.execute(decision, next_bar)

    assert isinstance(result, Rejection)
    assert "circuit_breaker_frozen" in result.reason


def test_close_position_realizes_pnl(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root, branch="close-test")
    _open_position(
        sim, log_root, make_decision, "BTC/USDT:USDT", "open_long", 1_700_000_000_000,
        50_000.0, leverage=5, pct=50.0,
    )
    wallet_after_open = sim.wallet_balance

    close_decision = md(make_decision, 
        ts=1_700_000_200_000, symbol="BTC/USDT:USDT", action="close",
        target_notional_pct=0.0, leverage=5,
    )
    log_and_get(log_root, close_decision)
    next_bar = {"open_time": 1_700_000_300_000, "open": 55_000.0}  # price went up
    trade = sim.execute(close_decision, next_bar)

    assert isinstance(trade, Trade)
    assert "BTC/USDT:USDT" not in sim.positions
    # profitable close -> wallet balance increases (net of the closing fee)
    assert sim.wallet_balance > wallet_after_open


def test_mark_to_market_and_get_portfolio(tmp_path, log_root, make_decision):
    sim = make_sim(tmp_path, log_root, branch="mtm-test")
    trade = _open_position(
        sim, log_root, make_decision, "BTC/USDT:USDT", "open_long", 1_700_000_000_000,
        50_000.0, leverage=2, pct=20.0,
    )
    nav_at_entry = sim.mark_to_market({"BTC/USDT:USDT": trade.price})
    assert nav_at_entry == pytest.approx(sim.wallet_balance)

    nav_up = sim.mark_to_market({"BTC/USDT:USDT": trade.price * 1.1})
    assert nav_up > nav_at_entry

    portfolio = sim.get_portfolio({"BTC/USDT:USDT": trade.price * 1.1})
    assert portfolio["nav"] == pytest.approx(nav_up)
    assert portfolio["branch"] == "mtm-test"
    assert portfolio["branch_dead"] is False
    assert len(portfolio["positions"]) == 1


def test_sqlite_state_survives_resume(tmp_path, log_root, make_decision):
    db_path = tmp_path / "resume_test.db"
    cfg = load_config()
    sim = Simulator(
        config=cfg, universe_symbols=UNIVERSE, db_path=db_path, branch="resume-test", log_root=log_root
    )
    trade = _open_position(
        sim, log_root, make_decision, "BTC/USDT:USDT", "open_long", 1_700_000_000_000,
        50_000.0, leverage=4, pct=30.0,
    )
    assert isinstance(trade, Trade)
    wallet_before = sim.wallet_balance
    positions_before = dict(sim.positions)

    # Fresh instance pointed at the same db file, without resume=True -> starts clean.
    sim_fresh = Simulator(
        config=cfg, universe_symbols=UNIVERSE, db_path=db_path, branch="resume-test", log_root=log_root
    )
    assert sim_fresh.wallet_balance == pytest.approx(float(cfg["capital_usdt"]))
    assert sim_fresh.positions == {}

    # Resuming instance recovers prior wallet balance and open positions.
    sim_resumed = Simulator(
        config=cfg, universe_symbols=UNIVERSE, db_path=db_path, branch="resume-test",
        log_root=log_root, resume=True,
    )
    assert sim_resumed.wallet_balance == pytest.approx(wallet_before)
    assert set(sim_resumed.positions.keys()) == set(positions_before.keys())
    resumed_pos = sim_resumed.positions["BTC/USDT:USDT"]
    original_pos = positions_before["BTC/USDT:USDT"]
    assert resumed_pos.notional == pytest.approx(original_pos.notional)
    assert resumed_pos.entry_price == pytest.approx(original_pos.entry_price)
    assert resumed_pos.margin == pytest.approx(original_pos.margin)
