from __future__ import annotations

import pytest

from LOCKED.position_risk_monitor import check_all_positions, check_position_drawdown
from LOCKED.schemas import PerpPosition


def _long_pos(symbol="BTC/USDT:USDT", entry=50_000.0, notional=30_000.0, leverage=3):
    return PerpPosition(
        symbol=symbol, side="long", notional=notional, entry_price=entry,
        margin=notional / leverage, leverage=leverage,
    )


def _short_pos(symbol="BTC/USDT:USDT", entry=50_000.0, notional=30_000.0, leverage=3):
    return PerpPosition(
        symbol=symbol, side="short", notional=notional, entry_price=entry,
        margin=notional / leverage, leverage=leverage,
    )


def test_long_position_triggers_on_drawdown_from_recent_peak():
    pos = _long_pos()
    # peak 51000 -> current 48000: (51000-48000)/51000*100 = 5.88%
    prices = [(1, 50_000.0), (2, 51_000.0), (3, 49_000.0), (4, 48_000.0)]
    result = check_position_drawdown(pos, prices, threshold_pct=5.0)
    assert result.triggered is True
    assert result.drawdown_pct == pytest.approx(5.882, abs=0.01)


def test_long_position_does_not_trigger_below_threshold():
    pos = _long_pos()
    prices = [(1, 50_000.0), (2, 51_000.0), (3, 50_500.0)]  # small dip, ~1%
    result = check_position_drawdown(pos, prices, threshold_pct=5.0)
    assert result.triggered is False
    assert result.drawdown_pct < 5.0


def test_short_position_triggers_on_rise_from_recent_trough():
    pos = _short_pos()
    # trough 48000 -> current 51000: (51000-48000)/48000*100 = 6.25%
    prices = [(1, 50_000.0), (2, 48_000.0), (3, 49_500.0), (4, 51_000.0)]
    result = check_position_drawdown(pos, prices, threshold_pct=5.0)
    assert result.triggered is True
    assert result.drawdown_pct == pytest.approx(6.25, abs=0.01)


def test_short_position_does_not_trigger_when_price_falls():
    pos = _short_pos()
    prices = [(1, 50_000.0), (2, 45_000.0)]  # price falling is GOOD for a short
    result = check_position_drawdown(pos, prices, threshold_pct=5.0)
    assert result.triggered is False


def test_exactly_at_threshold_triggers_inclusive():
    pos = _long_pos()
    prices = [(1, 100.0), (2, 95.0)]  # exactly 5% drawdown
    result = check_position_drawdown(pos, prices, threshold_pct=5.0)
    assert result.triggered is True


def test_insufficient_data_does_not_trigger():
    pos = _long_pos()
    assert check_position_drawdown(pos, [], threshold_pct=5.0).triggered is False
    assert check_position_drawdown(pos, [(1, 50_000.0)], threshold_pct=5.0).triggered is False


def test_check_all_positions_missing_symbol_data_does_not_trigger_or_raise():
    positions = {"BTC/USDT:USDT": _long_pos(), "ETH/USDT:USDT": _long_pos(symbol="ETH/USDT:USDT")}
    recent_prices_by_symbol = {
        "BTC/USDT:USDT": [(1, 50_000.0), (2, 40_000.0)],  # big drawdown, should trigger
        # ETH/USDT:USDT deliberately missing
    }
    results = check_all_positions(positions, recent_prices_by_symbol, threshold_pct=5.0)
    by_symbol = {r.symbol: r for r in results}
    assert by_symbol["BTC/USDT:USDT"].triggered is True
    assert by_symbol["ETH/USDT:USDT"].triggered is False
    assert by_symbol["ETH/USDT:USDT"].reason == "insufficient_recent_price_data"


def test_check_all_positions_empty_positions_returns_empty():
    assert check_all_positions({}, {}, threshold_pct=5.0) == []
