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

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000, params=None):
        # params 形参是真实 ccxt 签名的一部分(M6 反向翻页会传 params=
        # {"after": ...}),这里保留但不使用——这个假交易所本来就总共只有
        # 3条确定性数据,不需要真的翻页。
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


class PageCappedExchange(FakeExchange):
    """M5 回归测试用:模拟真实点火时发现的 OKX 行为——单次 fetch_ohlcv 调用
    无视传入的 limit,恒定只返回 per_call_limit 根K线,必须分页才能拿满一个
    更长的区间。"""

    def __init__(self, per_call_limit: int, total_available: int):
        super().__init__()
        self.per_call_limit = per_call_limit
        self.total_available = total_available  # 交易所总共"有"这么多根K线可给
        self.calls_log: list[tuple[int, int]] = []  # (since, limit) 每次实际收到的调用参数

    def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000):
        self.ohlcv_calls += 1
        start = since if since is not None else 0
        self.calls_log.append((start, limit))
        step = 4 * 60 * 60 * 1000
        bar_index_start = start // step
        rows = []
        for i in range(self.per_call_limit):
            bar_index = bar_index_start + i
            if bar_index >= self.total_available:
                break  # 交易所自己也没有更多数据了
            ts = bar_index * step
            rows.append([ts, 100.0, 101.0, 99.0, 100.5, 1000.0])
        return rows


def test_fetch_ohlcv_paginates_past_single_call_cap(tmp_path):
    """回归测试(真实点火时发现的bug):交易所单次调用恒定只返回300根K线,
    请求2年4h数据(~4380根)时,不分页只能拿到最早的300根,离用户要求的
    "COLD_START完整执行2年历史"差得很远。"""
    fake = PageCappedExchange(per_call_limit=300, total_available=2000)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)

    df = dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=1500)

    assert len(df) == 1500
    assert fake.ohlcv_calls == 5  # 300 * 5 = 1500
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].duplicated().sum() == 0
    # 分页游标必须每次都正确前进(下一页的 since 紧接着上一页最后一根K线之后),
    # 不是重复请求同一个区间。
    step = 4 * 60 * 60 * 1000
    for (since_a, _), (since_b, _) in zip(fake.calls_log, fake.calls_log[1:]):
        assert since_b == since_a + fake.per_call_limit * step


def test_fetch_ohlcv_pagination_stops_when_exchange_runs_out_of_data(tmp_path):
    """请求的 limit 超过交易所实际能给的总量时,分页必须正常停在"交易所说
    没有了"这一页,不无限循环、也不假装凑够了 limit 条。"""
    fake = PageCappedExchange(per_call_limit=300, total_available=650)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)

    df = dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=0, limit=100000)

    assert len(df) == 650  # 拿到交易所实际能给的全部数据,不多不少
    assert fake.ohlcv_calls == 3  # 300 + 300 + 50(最后一页不足per_call_limit,分页正确停止)


class ListingDateOhlcvExchange(FakeExchange):
    """M6 缺陷2 回归测试用:模拟"上市日晚于请求 since"的币种在真实OKX上的
    行为(比如实测中的 CL/HYPE/LAB/MU/PUMP/SNDK/XAU/ZEC)——上市日之前
    (bar 下标 < listing_bar_index)完全没有数据;不带游标的"探测"请求永远
    返回紧贴"现在"的最新一页;带 params['until'] 游标时返回"早于该时间戳"
    的一页(最多 per_call_limit 条),翻过上市日之后一律返回空。"""

    def __init__(self, per_call_limit: int, listing_bar_index: int, bars_since_listing: int):
        super().__init__()
        self.per_call_limit = per_call_limit
        self.listing_bar_index = listing_bar_index
        self.total_available = listing_bar_index + bars_since_listing
        self.calls_log: list[dict] = []

    def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000, params=None):
        self.ohlcv_calls += 1
        params = params or {}
        step = 4 * 60 * 60 * 1000
        self.calls_log.append({"since": since, "limit": limit, "params": dict(params)})

        def _bars(start_idx: int, end_idx: int) -> list[list]:
            if start_idx >= end_idx:
                return []
            return [[i * step, 100.0, 101.0, 99.0, 100.5, 1000.0] for i in range(start_idx, end_idx)]

        if "until" in params:
            end_idx = min(params["until"] // step, self.total_available)
            start_idx = max(self.listing_bar_index, end_idx - self.per_call_limit)
            return _bars(start_idx, end_idx)

        if since is not None:
            # 正向请求(旧的、有缺陷的路径):since 早于上市日时,窗口完全
            # 落在"上市前"就如实返回空——这正是缺陷2复现的关键。
            start_idx = max(since // step, self.listing_bar_index)
            end_idx = min(since // step + self.per_call_limit, self.total_available)
            return _bars(start_idx, end_idx)

        # 探测请求(since=None、无 until 游标):永远返回最新一页。
        end_idx = self.total_available
        start_idx = max(self.listing_bar_index, end_idx - self.per_call_limit)
        return _bars(start_idx, end_idx)


def test_fetch_ohlcv_since_before_listing_date_returns_full_history_since_listing(tmp_path):
    """M6 缺陷2 回归测试:since=730天前 请求 CL/HYPE/LAB/MU/PUMP/SNDK/XAU/
    ZEC 这类上市不满730天的币种时,正向分页第一页返回空,原代码把"空"误判
    为"拉完了",最终0根K线。修复后应该自动收敛为"该币种上市以来的全部历史"
    (M6规格书验收标准原话:"或该币种上市以来全部历史,以较短者为准")。"""
    step = 4 * 60 * 60 * 1000
    fake = ListingDateOhlcvExchange(per_call_limit=300, listing_bar_index=2000, bars_since_listing=650)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)

    df = dp.fetch_ohlcv("HYPE/USDT:USDT", timeframe="4h", since=0, limit=100000)

    assert len(df) == 650  # 上市日以来全部历史,不是0根
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].duplicated().sum() == 0
    assert int(df["timestamp"].min()) == 2000 * step  # 起点正好是"上市日"那根K线


def test_fetch_ohlcv_symbol_with_absolutely_no_data_stays_empty_not_error(tmp_path):
    """探测也拿不到任何数据(比如symbol写错/该合约从未有过K线)时,必须
    安静地返回0根,而不是抛异常或者死循环。"""
    fake = ListingDateOhlcvExchange(per_call_limit=300, listing_bar_index=0, bars_since_listing=0)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)

    df = dp.fetch_ohlcv("NODATA/USDT:USDT", timeframe="4h", since=0, limit=100000)

    assert len(df) == 0
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_ohlcv_single_call_unchanged_when_since_is_none(pipeline):
    """since=None(要"最近一批"数据,不是历史区间回放)时,不应该触发分页
    循环——保持原来的单次调用行为。"""
    dp, fake = pipeline
    dp.fetch_ohlcv("BTC/USDT:USDT", timeframe="4h", since=None, limit=5)
    assert fake.ohlcv_calls == 1


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


class CursorPagedFundingExchange(FakeExchange):
    """M6 缺陷1 修复后的回归测试用:模拟真实点火实测确认的 OKX
    fetch_funding_rate_history 行为——

    - 不带 params['after'] 游标时,永远返回"最新一页"(最多 per_call_limit
      条,紧贴"现在"),**忽略 since**——这正是真实 OKX 的行为,ccxt/okx.py
      里 since 只映射到 before(只圈下界,不圈上界),所以服务端永远优先给
      满足下界条件的最新一批,不会因为 since 传得多早就给你更早的数据。
    - 带 params['after']=X 时,返回"时间戳严格早于 X"的一页,同样最多
      per_call_limit 条——这是 OKX 真正的向后翻页游标。

    该币种总共有 total_available 条资金费率历史,时间戳从 idx=0(最早/
    "上市日")到 idx=total_available-1("现在")。
    """

    def __init__(self, per_call_limit: int, total_available: int):
        super().__init__()
        self.per_call_limit = per_call_limit
        self.total_available = total_available
        self.calls_log: list[dict] = []

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000, params=None):
        self.funding_history_calls += 1
        params = params or {}
        step = 8 * 60 * 60 * 1000
        page_limit = min(limit or self.per_call_limit, self.per_call_limit)
        self.calls_log.append({"since": since, "limit": limit, "params": dict(params)})

        if "after" in params:
            end_idx = min(params["after"] // step, self.total_available)
            start_idx = max(0, end_idx - page_limit)
        else:
            end_idx = self.total_available
            start_idx = max(0, end_idx - page_limit)

        if start_idx >= end_idx:
            return []
        return [{"timestamp": i * step, "fundingRate": 0.0001} for i in range(start_idx, end_idx)]


def test_old_forward_since_style_would_only_ever_return_latest_page(tmp_path):
    """回归写照(留档证明旧写法为什么在真实OKX语义下失效,不是测试当前
    实现):旧代码把 since 游标一路向前推、原样传给 exchange.fetch_funding_
    rate_history 的 since= 形参,从不使用 params['after']。在这个模拟真实
    OKX 行为的 mock 下,无论 since 传多大,只要没有触及 after 游标,拿到的
    永远是同一批"最新记录"——这正是生产实测"分页循环形同虚设"的根因。"""
    fake = CursorPagedFundingExchange(per_call_limit=300, total_available=3000)
    step = 8 * 60 * 60 * 1000

    page_at_since_0 = fake.fetch_funding_rate_history("BTC/USDT:USDT", since=0, limit=300)
    page_at_since_far_later = fake.fetch_funding_rate_history(
        "BTC/USDT:USDT", since=1000 * step, limit=300
    )

    assert page_at_since_0 == page_at_since_far_later
    assert len(page_at_since_0) == 300


def test_fetch_funding_rate_history_paginates_backward_using_after_cursor(tmp_path):
    """M6 缺陷1 修复验证:新实现改用 OKX 真实支持的向后翻页游标
    (params['after']),应该能从"现在"往回拼出完整的历史(这里模拟约733天,
    2200条×8h),不再被"since 正向推进对 OKX 无效"这个问题卡在约100天/300条。"""
    fake = CursorPagedFundingExchange(per_call_limit=300, total_available=2200)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)

    df = dp.fetch_funding_rate_history("BTC/USDT:USDT", since=0, limit=100000)

    assert len(df) == 2200  # 完整历史,不再卡在旧上限的~300条
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].duplicated().sum() == 0  # 无重复无空洞
    assert fake.funding_history_calls >= 8  # ceil(2200/300)=8,证明确实分了多页而非一次拿完


def test_fetch_funding_rate_history_stops_when_history_exhausted(tmp_path):
    """请求的区间超过该币种实际存在的历史时(比如 since 早于上市日),反向
    翻页必须正常停在"交易所返回空页"这一页,不无限循环、也不假装凑够了。"""
    fake = CursorPagedFundingExchange(per_call_limit=300, total_available=650)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)

    df = dp.fetch_funding_rate_history("BTC/USDT:USDT", since=0, limit=100000)

    assert len(df) == 650  # 拿到该币种实际存在的全部资金费率历史,不多不少


def test_fetch_funding_rate_history_zero_network_calls_when_fully_cached(tmp_path):
    """请求范围完全在缓存内时,不应该发起任何新的网络调用——现有缓存语义
    (DataPipeline.fetch_funding_rate_history 顶部的 need_fetch 判断)不因为
    分页实现从"前进"换成"后退"而回退。"""
    fake = CursorPagedFundingExchange(per_call_limit=300, total_available=2200)
    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=fake, backoff_base_seconds=0.0)

    df1 = dp.fetch_funding_rate_history("BTC/USDT:USDT", since=0, limit=100000)
    calls_after_first_fetch = fake.funding_history_calls
    assert calls_after_first_fetch > 0
    assert len(df1) == 2200

    df2 = dp.fetch_funding_rate_history("BTC/USDT:USDT", since=0, limit=100000)
    assert fake.funding_history_calls == calls_after_first_fetch  # 零新增网络调用
    assert len(df2) == 2200


def test_fetch_funding_rate_history_single_call_unchanged_when_since_is_none(pipeline):
    """since=None(要"最近一批"数据)时,不应该触发分页循环——保持原来的
    单次调用行为,与 fetch_ohlcv 的同款约定一致。"""
    dp, fake = pipeline
    dp.fetch_funding_rate_history("BTC/USDT:USDT", since=None, limit=3)
    assert fake.funding_history_calls == 1


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


# ----------------------------------------------------------------------
# M5 联网 shakedown 适配:公开历史归档回退路径。全部离线(mock
# urllib.request.urlopen,不碰真实的 data.binance.vision),与本文件
# 一直以来的"测试不接触真实网络"纪律保持一致。
# ----------------------------------------------------------------------
import io
import zipfile

import LOCKED.data_pipeline as data_pipeline_module


def _make_fake_klines_zip(rows: list[list]) -> bytes:
    header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
    lines = [header] + [",".join(str(v) for v in r) for r in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("FAKE-4h-2024-06.csv", "\n".join(lines))
    return buf.getvalue()


def _make_fake_funding_zip(rows: list[list]) -> bytes:
    header = "calc_time,funding_interval_hours,last_funding_rate"
    lines = [header] + [",".join(str(v) for v in r) for r in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("FAKE-fundingRate-2024-06.csv", "\n".join(lines))
    return buf.getvalue()


class _FakeHTTPResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_archive_symbol_and_months_between_are_pure():
    assert DataPipeline._archive_symbol("BTC/USDT:USDT") == "BTCUSDT"
    months = DataPipeline._months_between(
        since_ms=int(pd.Timestamp("2024-06-15", tz="UTC").timestamp() * 1000),
        until_ms=int(pd.Timestamp("2024-08-03", tz="UTC").timestamp() * 1000),
    )
    assert months == [(2024, 6), (2024, 7), (2024, 8)]


def test_fetch_ohlcv_archive_fallback_disabled_by_default_raises_immediately(tmp_path, monkeypatch):
    """安全默认值:enable_public_archive_fallback=False 时,实时API失败必须
    直接抛出,绝不触发任何网络下载(包括归档路径)——这就是修复"离线单测
    意外触发真实网络请求"那次回归所加的护栏。"""
    class AlwaysFailExchange:
        def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000):
            raise ConnectionError("simulated: live API unreachable")

    calls = {"count": 0}

    def _spy_urlopen(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("must not attempt any network call when fallback is disabled")

    monkeypatch.setattr(data_pipeline_module.urllib.request, "urlopen", _spy_urlopen)

    dp = DataPipeline(
        cache_dir=tmp_path / "cache", exchange=AlwaysFailExchange(),
        backoff_base_seconds=0.0, max_retries=1,
        enable_public_archive_fallback=False,
    )
    with pytest.raises(ConnectionError):
        dp.fetch_ohlcv("BTC/USDT:USDT", "4h", since=1_717_200_000_000)
    assert calls["count"] == 0


def test_fetch_ohlcv_falls_back_to_archive_when_enabled_and_live_api_fails(tmp_path, monkeypatch):
    class AlwaysFailExchange:
        def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000):
            raise ConnectionError("simulated: live API unreachable")

    fake_zip = _make_fake_klines_zip([
        [1_717_200_000_000, 67577.9, 67800.7, 67480.0, 67749.9, 11203.855, 0, 0, 0, 0, 0, 0],
        [1_717_214_400_000, 67750.0, 67857.5, 67608.6, 67677.9, 8158.758, 0, 0, 0, 0, 0, 0],
    ])

    def _fake_urlopen(req, timeout=None):
        assert "data.binance.vision" in req.full_url
        assert "BTCUSDT" in req.full_url
        return _FakeHTTPResponse(fake_zip)

    monkeypatch.setattr(data_pipeline_module.urllib.request, "urlopen", _fake_urlopen)

    dp = DataPipeline(
        cache_dir=tmp_path / "cache", exchange=AlwaysFailExchange(),
        backoff_base_seconds=0.0, max_retries=1,
        enable_public_archive_fallback=True,
    )
    result = dp.fetch_ohlcv("BTC/USDT:USDT", "4h", since=1_717_200_000_000, limit=10)
    assert len(result) == 2
    assert result.iloc[0]["open"] == pytest.approx(67577.9)
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_funding_rate_history_falls_back_to_archive_when_enabled(tmp_path, monkeypatch):
    class AlwaysFailExchange:
        def fetch_funding_rate_history(self, symbol, since=None, limit=1000, params=None):
            raise ConnectionError("simulated: live API unreachable")

    fake_zip = _make_fake_funding_zip([
        [1_717_200_000_000, 8, 0.00010000],
        [1_717_228_800_000, 8, -0.00005000],
    ])

    def _fake_urlopen(req, timeout=None):
        assert "fundingRate" in req.full_url
        return _FakeHTTPResponse(fake_zip)

    monkeypatch.setattr(data_pipeline_module.urllib.request, "urlopen", _fake_urlopen)

    dp = DataPipeline(
        cache_dir=tmp_path / "cache", exchange=AlwaysFailExchange(),
        backoff_base_seconds=0.0, max_retries=1,
        enable_public_archive_fallback=True,
    )
    result = dp.fetch_funding_rate_history("BTC/USDT:USDT", since=1_717_200_000_000, limit=10)
    assert len(result) == 2
    assert result.iloc[1]["funding_rate"] == pytest.approx(-0.00005)


def test_fetch_ohlcv_archive_fallback_not_attempted_when_since_is_none(tmp_path, monkeypatch):
    """since=None 代表"要最新数据",归档回退不适用(它只处理纯历史区间),
    即使 enable_public_archive_fallback=True 也必须直接抛出,不下载归档。"""
    class AlwaysFailExchange:
        def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000):
            raise ConnectionError("simulated: live API unreachable")

    calls = {"count": 0}

    def _spy_urlopen(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("archive fallback must not trigger when since=None")

    monkeypatch.setattr(data_pipeline_module.urllib.request, "urlopen", _spy_urlopen)

    dp = DataPipeline(
        cache_dir=tmp_path / "cache", exchange=AlwaysFailExchange(),
        backoff_base_seconds=0.0, max_retries=1,
        enable_public_archive_fallback=True,
    )
    with pytest.raises(ConnectionError):
        dp.fetch_ohlcv("BTC/USDT:USDT", "4h", since=None)
    assert calls["count"] == 0


def test_download_archive_returns_none_on_404_not_treated_as_error(tmp_path, monkeypatch):
    """归档还没被 Binance 发布(常见于"最近一个月")时返回 404,这是预期的
    正常情况(历史缺口),不是需要重试/报错的网络故障。"""
    import urllib.error

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(data_pipeline_module.urllib.request, "urlopen", _fake_urlopen)

    dp = DataPipeline(cache_dir=tmp_path / "cache", exchange=FakeExchange(), backoff_base_seconds=0.0)
    result = dp.fetch_ohlcv_from_public_archive(
        "BTC/USDT:USDT", "4h", since=1_717_200_000_000, until=1_717_300_000_000
    )
    assert result.empty
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
