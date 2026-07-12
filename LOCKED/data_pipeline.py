"""
data_pipeline.py —— 行情数据管道(§2.1)。

职责:通过 ccxt 拉取并缓存行情。只用公开端点。

铁律:
- 全项目没有任何真实交易所下单代码,ccxt 只用公开行情接口。
- 本模块的 ccxt 交易所实例绝不配置 apiKey/secret,绝不调用任何私有
  (下单/余额/持仓)端点。任何新增方法如果需要认证,一律不允许加进来。

设计说明:
- 交易所以 USDT 本位永续合约(swap, defaultType="future")模式构造。
- symbol 使用 ccxt 统一格式,例如 "BTC/USDT:USDT"。
- 所有 OHLCV 数据落盘缓存为 parquet(按 symbol/timeframe 分文件),
  重复请求先查缓存,只对缺失区间发起网络请求,新旧数据按 timestamp
  合并去重后回写缓存,保证进程重启不重复拉取已缓存区间。
- 网络请求失败时指数退避重试 3 次,仍失败则向上抛出异常。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

from LOCKED.clock import Clock, SystemClock

logger = logging.getLogger(__name__)

T = TypeVar("T")

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FUNDING_COLUMNS = ["timestamp", "funding_rate"]

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


class DataPipeline:
    """封装 ccxt 公开行情端点 + 本地 parquet 缓存。

    构造时绝不传入 apiKey/secret —— 仅使用公开市场数据端点
    (fetch_ohlcv / fetch_ticker / fetch_funding_rate / fetch_funding_rate_history)。
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        cache_dir: Path | str | None = None,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        exchange: Any | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.exchange_id = exchange_id
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        # M5: 全项目单一时间源(见 LOCKED/clock.py)。默认 SystemClock 保持既有
        # 生产行为不变;测试/回放注入 FakeClock 即可让 fetch_history_bundle 的
        # "since" 计算完全确定。
        self.clock: Clock = clock if clock is not None else SystemClock()

        if exchange is not None:
            # 测试/依赖注入路径:调用方直接传入一个(通常是 mock 的)交易所对象。
            self.exchange = exchange
        else:
            self.exchange = self._build_exchange(exchange_id)

    # ------------------------------------------------------------------
    # 构造 / 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _build_exchange(exchange_id: str) -> Any:
        import ccxt

        exchange_cls = getattr(ccxt, exchange_id)
        # 仅公开数据模式:不设置 apiKey / secret。
        exchange = exchange_cls(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",  # USDT 本位合约(swap)
                },
            }
        )
        return exchange

    def _with_retry(self, fn: Callable[[], T], description: str) -> T:
        """指数退避重试:最多 self.max_retries 次尝试,全部失败则抛出最后一次异常。"""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - 需要捕获任意 ccxt/网络异常并重试
                last_exc = exc
                logger.warning(
                    "%s failed (attempt %d/%d): %r", description, attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    sleep_s = self.backoff_base_seconds * (2 ** (attempt - 1))
                    time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc

    def _symbol_cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "-").replace(":", "_")
        return self.cache_dir / f"ohlcv_{safe_symbol}_{timeframe}.parquet"

    def _funding_cache_path(self, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "-").replace(":", "_")
        return self.cache_dir / f"funding_{safe_symbol}.parquet"

    @staticmethod
    def _load_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _merge_dedupe(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if existing.empty:
            merged = new
        elif new.empty:
            merged = existing
        else:
            merged = pd.concat([existing, new], ignore_index=True)
        merged = merged.drop_duplicates(subset="timestamp", keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        return merged

    def _save_parquet(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False)

    # ------------------------------------------------------------------
    # 公开数据接口
    # ------------------------------------------------------------------
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        since: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """拉取 OHLCV K 线,自动使用/更新本地 parquet 缓存。

        列: [timestamp, open, high, low, close, volume],timestamp 为 UTC 毫秒。
        若请求区间已完全被缓存覆盖(cache 中存在 >= since 的数据,且最新一条
        缓存时间戳已跟上"现在"允许的粒度),则不发起网络请求,直接从缓存切片返回。
        """
        cache_path = self._symbol_cache_path(symbol, timeframe)
        cached = self._load_parquet(cache_path, OHLCV_COLUMNS)

        need_fetch = True
        if since is not None and not cached.empty:
            cached_min = int(cached["timestamp"].min())
            cached_max = int(cached["timestamp"].max())
            # 缓存已覆盖 since 起点,且缓存条数达到 limit(说明这段区间是"满"的，
            # 不太可能有新数据插进已缓存的历史区间里）—— 直接命中缓存，无需网络请求。
            covers_since = cached_min <= since
            enough_rows = len(cached[cached["timestamp"] >= since]) >= min(limit, len(cached))
            if covers_since and enough_rows and cached_max >= since:
                need_fetch = False

        if need_fetch:
            raw = self._with_retry(
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit),
                f"fetch_ohlcv({symbol}, {timeframe})",
            )
            fetched = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
            merged = self._merge_dedupe(cached, fetched)
            self._save_parquet(cache_path, merged)
            cached = merged

        result = cached
        if since is not None:
            result = result[result["timestamp"] >= since]
        if limit is not None:
            result = result.head(limit)
        return result.reset_index(drop=True)

    def fetch_latest_snapshot(self, symbols: list[str]) -> dict[str, dict]:
        """给定 universe 币种列表,拉取每个币的最新 ticker 快照。

        返回 {symbol: {"last": ..., "bid": ..., "ask": ..., "quote_volume_24h": ...}}
        """
        snapshot: dict[str, dict] = {}
        for symbol in symbols:
            ticker = self._with_retry(
                lambda s=symbol: self.exchange.fetch_ticker(s),
                f"fetch_ticker({symbol})",
            )
            snapshot[symbol] = {
                "last": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "quote_volume_24h": ticker.get("quoteVolume"),
            }
        return snapshot

    def fetch_history_bundle(
        self,
        symbols: list[str],
        timeframe: str = "4h",
        history_days: int = 730,
        limit: int = 1000,
    ) -> dict[str, pd.DataFrame]:
        """冷启动:为 universe 中每个币拉取 history_days 天的历史 K 线。"""
        since = self.clock.now_ms() - history_days * 24 * 60 * 60 * 1000
        bundle: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            bundle[symbol] = self.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        return bundle

    def fetch_funding_rate(self, symbol: str) -> float:
        """当期资金费率(ccxt: fetch_funding_rate,公开接口)。"""
        result = self._with_retry(
            lambda: self.exchange.fetch_funding_rate(symbol),
            f"fetch_funding_rate({symbol})",
        )
        rate = result.get("fundingRate")
        if rate is None:
            raise ValueError(f"fetch_funding_rate({symbol}) returned no fundingRate: {result!r}")
        return float(rate)

    def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """历史资金费率序列,列: [timestamp, funding_rate](UTC 毫秒,decimal 费率）。

        本地缓存到 parquet,按 symbol 分文件，合并去重后返回。
        """
        cache_path = self._funding_cache_path(symbol)
        cached = self._load_parquet(cache_path, FUNDING_COLUMNS)

        need_fetch = True
        if since is not None and not cached.empty:
            cached_min = int(cached["timestamp"].min())
            cached_max = int(cached["timestamp"].max())
            covers_since = cached_min <= since
            enough_rows = len(cached[cached["timestamp"] >= since]) >= min(limit, len(cached))
            if covers_since and enough_rows and cached_max >= since:
                need_fetch = False

        if need_fetch:
            raw = self._with_retry(
                lambda: self.exchange.fetch_funding_rate_history(symbol, since=since, limit=limit),
                f"fetch_funding_rate_history({symbol})",
            )
            rows = [
                {
                    "timestamp": entry.get("timestamp"),
                    "funding_rate": entry.get("fundingRate"),
                }
                for entry in raw
            ]
            fetched = pd.DataFrame(rows, columns=FUNDING_COLUMNS)
            merged = self._merge_dedupe(cached, fetched)
            self._save_parquet(cache_path, merged)
            cached = merged

        result = cached
        if since is not None:
            result = result[result["timestamp"] >= since]
        if limit is not None:
            result = result.head(limit)
        return result.reset_index(drop=True)
