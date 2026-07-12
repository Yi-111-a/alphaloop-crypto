from __future__ import annotations

import pandas as pd
import pytest

from LOCKED.data_pipeline import DataPipeline


# ----------------------------------------------------------------------
# 假交易所(离线、确定性,不发起任何网络请求)
# ----------------------------------------------------------------------
class FakeExchange:
    """模拟 ccxt 交易所对象的公开数据端点。"""

    def __init__(self):
        self.ohlcv_calls = 0
        self.ticker_calls = 0
        self.funding_rate_calls = 0
        self.funding_history_calls = 0

    def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000):
        self.ohlcv_calls += 1
        # 生成 5 根确定性的 4h K 线,从 since 开始(或从 0 开始)
        start = since if since is not None else 0
        step = 4 * 60 * 60 * 1000
        rows = []
        for i in range(5):
            ts = start + i * step
            rows.append([ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0 + i])
        return rows

    def fetch_ticker(self, symbol):
        self.ticker_calls += 1
        return {"last": 100.0, "bid": 99.9, "ask": 100.1, "quoteVolume": 123456789.0}

    def fetch_funding_rate(self, symbol):
        self.funding_rate_calls += 1
        return {"fundingRate": 0.0001, "symbol": symbol}

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        self.funding_history_calls += 1
        start = since if since is not None else 0
        step = 8 * 60 * 60 * 1000
        return [
            {"timestamp": start + i * step, "fundingRate": 0.0001 * (i + 1)}
            for i in range(3)
        ]


class FlakyExchange(FakeExchange):
    """前 N 次调用抛异常,之后恢复正常 —— 用于测试重试逻辑。"""

    def __init__(self, fail_times: int):
        super().__init__()
        self.fail_times = fail_times
        self.attempts = 0

    def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ConnectionError("simulated network failure")
        return super().fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)


@pytest.fixture
def pipeline(tmp_path):
    fake = FakeExchange()
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)
    return dp, fake


# ----------------------------------------------------------------------
# fetch_ohlcv
# ----------------------------------------------------------------------
def test_fetch_ohlcv_column_shape(pipeline):
    dp, fake = pipeline
    df = dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=5)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 5
    assert fake.ohlcv_calls == 1


def test_fetch_ohlcv_cache_avoids_second_network_call(pipeline):
    dp, fake = pipeline
    df1 = dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=5)
    assert fake.ohlcv_calls == 1

    # Same (symbol, timeframe, since, limit) range -> should be served from
    # the on-disk parquet cache without a second call into the exchange.
    df2 = dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=5)
    assert fake.ohlcv_calls == 1  # unchanged
    pd.testing.assert_frame_equal(df1, df2)


def test_fetch_ohlcv_cache_persists_to_parquet_file(pipeline, tmp_path):
    dp, fake = pipeline
    dp.fetch_ohlcv("ETH/USDT:USDT", timeframe="4h", since=0, limit=5)
    cache_path = dp._symbol_cache_path("ETH/USDT:USDT", "4h")
    assert cache_path.exists()
    on_disk = pd.read_parquet(cache_path)
    assert len(on_disk) == 5


def test_fetch_ohlcv_new_pipeline_instance_reuses_cache(tmp_path):
    fake1 = FakeExchange()
    dp1 = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake1, backoff_base_seconds=0.0)
    dp1.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=5)
    assert fake1.ohlcv_calls == 1

    # Simulate a process restart: brand new DataPipeline pointed at the same
    # cache_dir, with a fake exchange that would raise if actually called.
    class ExplodingExchange:
        def fetch_ohlcv(self, *a, **kw):
            raise AssertionError("network should not be hit after restart with warm cache")

    dp2 = DataPipeline(cache_dir=tmp_path / "cache", exchange=ExplodingExchange(), backoff_base_seconds=0.0)
    df = dp2.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=5)
    assert len(df) == 5


def test_fetch_ohlcv_merges_and_dedupes_new_range(pipeline):
    dp, fake = pipeline
    dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=5)
    assert fake.ohlcv_calls == 1

    # Ask for a later range not covered by cache -> must trigger a new call,
    # and cache should merge/dedupe by timestamp.
    later_since = 100 * 4 * 60 * 60 * 1000
    dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=later_since, limit=5)
    assert fake.ohlcv_calls == 2

    cache_path = dp._symbol_cache_path("BTC/USDT:USDT", "4h")
    on_disk = pd.read_parquet(cache_path)
    assert on_disk["timestamp"].is_monotonic_increasing
    assert on_disk["timestamp"].duplicated().sum() == 0


# ----------------------------------------------------------------------
# retry / backoff behavior
# ----------------------------------------------------------------------
def test_retry_then_succeed(tmp_path):
    fake = FlakyExchange(fail_times=2)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)
    df = dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=5)
    assert len(df) == 5
    assert fake.attempts == 3  # failed twice, succeeded on 3rd attempt


def test_retry_exhausted_raises(tmp_path):
    fake = FlakyExchange(fail_times=3)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)
    with pytest.raises(ConnectionError):
        dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=5)
    assert fake.attempts == 3  # exactly max_retries attempts, then raise


def test_retry_respects_custom_max_retries(tmp_path):
    fake = FlakyExchange(fail_times=1)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0, max_retries=1)
    with pytest.raises(ConnectionError):
        dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=5)
    assert fake.attempts == 1


# ----------------------------------------------------------------------
# fetch_latest_snapshot
# ----------------------------------------------------------------------
def test_fetch_latest_snapshot_shape(pipeline):
    dp, fake = pipeline
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    snapshot = dp.fetch_latest_snapshot(symbols)
    assert set(snapshot.keys()) == set(symbols)
    for sym in symbols:
        entry = snapshot[sym]
        assert set(entry.keys()) == {"last", "bid", "ask", "quote_volume_24h"}
        assert entry["last"] == 100.0
    assert fake.ticker_calls == 2


# ----------------------------------------------------------------------
# fetch_history_bundle
# ----------------------------------------------------------------------
def test_fetch_history_bundle_shape(pipeline):
    dp, fake = pipeline
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    bundle = dp.fetch_history_bundle(symbols, timeframe="4h", history_days=1, limit=5)
    assert set(bundle.keys()) == set(symbols)
    for sym in symbols:
        df = bundle[sym]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(df) == 5


# ----------------------------------------------------------------------
# fetch_funding_rate
# ----------------------------------------------------------------------
def test_fetch_funding_rate_returns_float(pipeline):
    dp, fake = pipeline
    rate = dp.fetch_funding_rate("BTC/USDT:USDT")
    assert isinstance(rate, float)
    assert rate == pytest.approx(0.0001)
    assert fake.funding_rate_calls == 1


# ----------------------------------------------------------------------
# fetch_funding_rate_history
# ----------------------------------------------------------------------
def test_fetch_funding_rate_history_shape(pipeline):
    dp, fake = pipeline
    df = dp.fetch_funding_rate_history("BTC/USDT:USDT", since=0, limit=3)
    assert list(df.columns) == ["timestamp", "funding_rate"]
    assert len(df) == 3
    assert fake.funding_history_calls == 1


def test_fetch_funding_rate_history_cache_avoids_second_call(pipeline):
    dp, fake = pipeline
    dp.fetch_funding_rate_history("BTC/USDT:USDT", since=0, limit=3)
    assert fake.funding_history_calls == 1
    dp.fetch_funding_rate_history("BTC/USDT:USDT", since=0, limit=3)
    assert fake.funding_history_calls == 1  # served from cache


# ----------------------------------------------------------------------
# no private/order-placement surface
# ----------------------------------------------------------------------
def test_pipeline_never_configures_credentials(tmp_path):
    # Constructing a real (non-injected) exchange must not carry apiKey/secret.
    dp = DataPipeline(exchange_id="binance", cache_dir=tmp_path / "cache")
    assert dp.exchange.apiKey in (None, "")
    assert dp.exchange.secret in (None, "")


def test_pipeline_has_no_order_placement_methods_used():
    # Principle check: the module source must not reference private trading
    # endpoints like create_order / cancel_order / fetch_balance.
    import inspect

    from LOCKED import data_pipeline

    source = inspect.getsource(data_pipeline)
    for forbidden in ["create_order", "cancel_order", "fetch_balance", "apiKey", "secret\""]:
        # apiKey appears only as an absence-check word above; ensure it's not
        # being *set* to a literal value in this module.
        if forbidden in ("apiKey",):
            assert "apiKey\":" not in source and "apiKey =" not in source
        else:
            assert forbidden not in source
