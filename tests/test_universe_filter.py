from __future__ import annotations

import json
import time

import pytest

from LOCKED.universe_filter import UniverseFilter, UNKNOWN_LISTING_DAYS


def make_config(**overrides):
    rule = dict(
        source="binance_usdt_perp",
        min_24h_volume_usdt=100e6,
        min_listing_days=90,
        refresh="weekly",
        blacklist=[],
    )
    rule.update(overrides)
    return {"universe_rule": rule}


@pytest.fixture
def uf():
    # exchange=object() 只是占位,apply_filters 是纯函数测试不会触发网络调用。
    return UniverseFilter(config=make_config(), exchange=object())


# ----------------------------------------------------------------------
# apply_filters 纯函数单测(无需网络/mock)
# ----------------------------------------------------------------------


def test_new_coin_excluded_even_with_huge_volume(uf):
    """验收标准:上市30天的新币,即使成交额巨大也必须被排除。"""
    candidates = [
        {"symbol": "NEWCOIN/USDT:USDT", "volume_24h_usdt": 5_000_000_000.0, "listing_days": 30},
    ]
    result = uf.apply_filters(candidates)
    assert "NEWCOIN/USDT:USDT" not in result
    assert result == []


def test_low_volume_excluded(uf):
    candidates = [
        {"symbol": "LOWVOL/USDT:USDT", "volume_24h_usdt": 50e6, "listing_days": 200},
    ]
    result = uf.apply_filters(candidates)
    assert "LOWVOL/USDT:USDT" not in result


def test_blacklisted_symbol_excluded_even_if_qualifying(uf):
    candidates = [
        {"symbol": "BADCOIN/USDT:USDT", "volume_24h_usdt": 500e6, "listing_days": 300},
    ]
    result = uf.apply_filters(candidates, blacklist=["BADCOIN/USDT:USDT"])
    assert "BADCOIN/USDT:USDT" not in result


def test_leveraged_token_excluded(uf):
    candidates = [
        {"symbol": "BTCUP/USDT:USDT", "volume_24h_usdt": 500e6, "listing_days": 300},
        {"symbol": "ETHBULL/USDT:USDT", "volume_24h_usdt": 500e6, "listing_days": 300},
        {"symbol": "ETHBEAR/USDT:USDT", "volume_24h_usdt": 500e6, "listing_days": 300},
        {"symbol": "BTCDOWN/USDT:USDT", "volume_24h_usdt": 500e6, "listing_days": 300},
    ]
    result = uf.apply_filters(candidates)
    assert result == []


def test_normal_qualifying_coin_included(uf):
    candidates = [
        {"symbol": "BTC/USDT:USDT", "volume_24h_usdt": 500e6, "listing_days": 200},
    ]
    result = uf.apply_filters(candidates)
    assert result == ["BTC/USDT:USDT"]


def test_unknown_listing_date_treated_as_too_new(uf):
    """上市时间未知 -> 保守处理为太新,必须排除(见模块 docstring 的 fallback 策略)。"""
    candidates = [
        {"symbol": "UNKNOWN/USDT:USDT", "volume_24h_usdt": 500e6, "listing_days": UNKNOWN_LISTING_DAYS},
    ]
    result = uf.apply_filters(candidates)
    assert result == []


def test_mixed_candidates_only_qualifying_survive(uf):
    candidates = [
        {"symbol": "BTC/USDT:USDT", "volume_24h_usdt": 500e6, "listing_days": 400},
        {"symbol": "ETH/USDT:USDT", "volume_24h_usdt": 300e6, "listing_days": 400},
        {"symbol": "NEWCOIN/USDT:USDT", "volume_24h_usdt": 5_000_000_000.0, "listing_days": 30},
        {"symbol": "LOWVOL/USDT:USDT", "volume_24h_usdt": 1e6, "listing_days": 400},
        {"symbol": "SOLUP/USDT:USDT", "volume_24h_usdt": 500e6, "listing_days": 400},
    ]
    result = uf.apply_filters(candidates)
    assert result == ["BTC/USDT:USDT", "ETH/USDT:USDT"]


# ----------------------------------------------------------------------
# refresh() 端到端测试,使用注入的 fake exchange(无网络)
# ----------------------------------------------------------------------


class FakeExchange:
    """最小可用的 ccxt 交易所替身,只实现 refresh() 需要的公开方法。"""

    def __init__(self):
        now_ms = int(time.time() * 1000)
        day_ms = 24 * 60 * 60 * 1000

        self._markets = {
            # 合格:swap + USDT + linear,上市 400 天
            "BTC/USDT:USDT": {
                "swap": True,
                "quote": "USDT",
                "linear": True,
                "created": now_ms - 400 * day_ms,
                "info": {},
            },
            # 成交额不足
            "LOWVOL/USDT:USDT": {
                "swap": True,
                "quote": "USDT",
                "linear": True,
                "created": now_ms - 400 * day_ms,
                "info": {},
            },
            # 新币(30天)
            "NEWCOIN/USDT:USDT": {
                "swap": True,
                "quote": "USDT",
                "linear": True,
                "created": now_ms - 30 * day_ms,
                "info": {},
            },
            # 杠杆代币
            "ETHBULL/USDT:USDT": {
                "swap": True,
                "quote": "USDT",
                "linear": True,
                "created": now_ms - 400 * day_ms,
                "info": {},
            },
            # 非 swap(现货),应被跳过
            "SOL/USDT": {
                "swap": False,
                "quote": "USDT",
                "linear": False,
                "created": now_ms - 400 * day_ms,
                "info": {},
            },
            # onboardDate fallback 路径(无 created,靠 info.onboardDate)
            "XRP/USDT:USDT": {
                "swap": True,
                "quote": "USDT",
                "linear": True,
                "info": {"onboardDate": now_ms - 200 * day_ms},
            },
        }

        self._tickers = {
            "BTC/USDT:USDT": {"quoteVolume": 500e6},
            "LOWVOL/USDT:USDT": {"quoteVolume": 10e6},
            "NEWCOIN/USDT:USDT": {"quoteVolume": 5_000e6},
            "ETHBULL/USDT:USDT": {"quoteVolume": 500e6},
            "SOL/USDT": {"quoteVolume": 500e6},
            "XRP/USDT:USDT": {"quoteVolume": 200e6},
        }

    def load_markets(self):
        return self._markets

    def fetch_tickers(self):
        return self._tickers


def test_refresh_writes_expected_json_shape(tmp_path):
    config = make_config()
    fake_exchange = FakeExchange()
    uf = UniverseFilter(config=config, exchange=fake_exchange)

    output_path = tmp_path / "universe_active.json"
    result = uf.refresh(output_path=output_path)

    # 返回值形状
    assert set(result.keys()) == {"generated_at", "symbols"}
    assert isinstance(result["generated_at"], int)
    assert result["symbols"] == ["BTC/USDT:USDT", "XRP/USDT:USDT"]

    # 文件确实写到了 tmp_path,而不是项目根目录的真实文件
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as f:
        on_disk = json.load(f)

    assert on_disk == result
    assert set(on_disk.keys()) == {"generated_at", "symbols"}
    assert on_disk["symbols"] == ["BTC/USDT:USDT", "XRP/USDT:USDT"]


def test_refresh_respects_blacklist_from_config(tmp_path):
    config = make_config(blacklist=["BTC/USDT:USDT"])
    fake_exchange = FakeExchange()
    uf = UniverseFilter(config=config, exchange=fake_exchange)

    output_path = tmp_path / "universe_active.json"
    result = uf.refresh(output_path=output_path)

    assert "BTC/USDT:USDT" not in result["symbols"]
    assert result["symbols"] == ["XRP/USDT:USDT"]
