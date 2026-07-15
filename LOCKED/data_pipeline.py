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

M5 联网 shakedown 适配记录(2026年真实联网环境下发现,授权范围内的修改,
只动本文件,LOCKED区其余业务逻辑一行未改):
- 实测发现本沙箱网络环境下 api.binance.com / fapi.binance.com(Binance
  的实时交易API域名)不可达 —— 直接连接超时,ccxt 走这两个域名的请求
  (fetch_ohlcv/fetch_ticker/fetch_funding_rate 等实时接口)全部失败;
  用 urllib 直接探测这两个域名返回 HTTP 451(Unavailable For Legal
  Reasons)—— 这是 Binance 自己基于请求方 IP 的地域限制返回的响应,不是
  本沙箱的防火墙拦截。相反,Binance 官方的公开历史数据归档服务
  data.binance.vision(无需鉴权、CDN托管、不受同一地域限制)可以正常
  访问,且确认能下载真实的 2024 年 USDT本位永续合约 4h K线与资金费率
  历史月度归档(已用一个真实文件核对过 CSV 格式)。
- 因此新增 fetch_ohlcv_from_public_archive() /
  fetch_funding_rate_history_from_public_archive() 两个方法,并把它们接入
  fetch_ohlcv()/fetch_funding_rate_history() 作为"实时API不可达时的历史区间
  回退路径"——只在请求的是纯历史区间(since 存在)且 ccxt 实时调用失败时
  触发,不影响 ccxt 可达环境下的原有行为(原有行为优先,回退路径是新增的
  韧性,不是替换)。
- fetch_latest_snapshot()/fetch_funding_rate()(取"当前"快照/当前费率)
  故意没有加归档回退——月度归档是有延迟的历史数据,不存在"当前"这个概念,
  给它们接一个历史数据当"当前值"用是伪造实时性,不属于本次授权的"适配层"
  修复范畴。这两个方法在实时API不可达的环境下会如实失败,而不是安静地返回
  一个不新鲜的数字。
"""
from __future__ import annotations

import csv
import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

from LOCKED.clock import Clock, SystemClock

logger = logging.getLogger(__name__)

T = TypeVar("T")

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FUNDING_COLUMNS = ["timestamp", "funding_rate"]

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

# Binance 官方公开历史数据归档(无需鉴权,CDN 托管,不经过 api.binance.com/
# fapi.binance.com 这两个可能受地域限制的域名)。只用于"实时API不可达时的
# 历史区间回退",不用于任何"当前值"语义的接口。
_BINANCE_VISION_BASE = "https://data.binance.vision"

_TIMEFRAME_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def _timeframe_to_ms(timeframe: str) -> int:
    """"4h" -> 14400000。支持 m/h/d 三种单位,ccxt 统一 timeframe 字符串的
    子集(本项目只用到 "4h",其它单位一并支持是为了不把这个小工具函数写死
    成只认一种格式)。"""
    unit = timeframe[-1]
    if unit not in _TIMEFRAME_UNIT_MS:
        raise ValueError(f"unsupported timeframe unit in {timeframe!r}: expected one of m/h/d")
    quantity = int(timeframe[:-1])
    return quantity * _TIMEFRAME_UNIT_MS[unit]


class DataPipeline:
    """封装 ccxt 公开行情端点 + 本地 parquet 缓存。

    构造时绝不传入 apiKey/secret —— 仅使用公开市场数据端点
    (fetch_ohlcv / fetch_ticker / fetch_funding_rate / fetch_funding_rate_history)。
    """

    def __init__(
        self,
        exchange_id: str = "okx",
        cache_dir: Path | str | None = None,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        exchange: Any | None = None,
        clock: Clock | None = None,
        enable_public_archive_fallback: bool = False,
    ) -> None:
        self.exchange_id = exchange_id
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        # M5 联网 shakedown 适配:实时API不可达时,是否允许 fetch_ohlcv/
        # fetch_funding_rate_history 回退到 data.binance.vision 的公开历史
        # 归档(见模块 docstring)。**默认关闭**——这是刻意的安全默认值,
        # 不是"先做出来能力,default打开图省事":一旦默认打开,任何用注入
        # 假交易所模拟"实时调用失败"来测试重试耗尽行为的测试(比如
        # test_retry_exhausted_raises),都会在"重试耗尽"之后意外触发一次
        # 真实的网络请求去下载归档——这会让原本完全离线、确定性的单测悄悄
        # 依赖真实网络,违反了本模块一直以来的"测试不碰真实网络"纪律。生产
        # 环境/shakedown 脚本需要这条回退路径时,显式传
        # enable_public_archive_fallback=True。
        self.enable_public_archive_fallback = enable_public_archive_fallback
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

    # M5 shakedown 适配:不同交易所在 ccxt 里给"USDT本位永续合约"用的
    # options.defaultType 术语不统一(币安用 "future",OKX/大多数其它交易所
    # 用 "swap")。exchange_id 默认值也从 "binance" 改为 "okx"——本沙箱网络
    # 环境下 api.binance.com/fapi.binance.com 对该出口IP返回 HTTP 451(见
    # 模块 docstring 开头的联网适配记录),经用户确认后切换到 OKX。
    _DEFAULT_TYPE_BY_EXCHANGE: dict[str, str] = {
        "binance": "future",
        "binanceusdm": "future",
    }

    @classmethod
    def _build_exchange(cls, exchange_id: str) -> Any:
        import ccxt

        exchange_cls = getattr(ccxt, exchange_id)
        default_type = cls._DEFAULT_TYPE_BY_EXCHANGE.get(exchange_id, "swap")
        # 仅公开数据模式:不设置 apiKey / secret。
        exchange = exchange_cls(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": default_type,  # USDT 本位合约(swap)
                },
            }
        )
        # M5 适配:ccxt 的 requests.Session 默认 trust_env=False,不会自动读取
        # HTTP_PROXY/HTTPS_PROXY 环境变量(与标准库 urllib/裸 requests 的默认
        # 行为不同)。本沙箱的联网访问依赖一个已配置好的本地代理环境变量,不
        # 显式打开 trust_env 的话 ccxt 侧的请求会直接尝试(失败的)直连。
        if hasattr(exchange, "session"):
            exchange.session.trust_env = True
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

    # ------------------------------------------------------------------
    # M5 联网适配:OHLCV 分页拉取。真实点火时发现 OKX 单次 fetch_ohlcv 调用
    # 恒定只返回 300 根K线,无视传入的 limit=100000——这是交易所单次请求的
    # 硬上限,不是本项目能通过重试/加大limit绕开的东西。ccxt 自带的
    # params={"paginate": True} 内部机制实测有一个不透明的总条数上限(实测
    # 请求2年4h数据只拿回2000条,远不到期望的~4380条,且没有任何文档说明
    # 这个上限从哪来),不如自己写一个透明、可测试的分页循环可靠。
    # ------------------------------------------------------------------

    def _fetch_ohlcv_live_paginated(
        self, symbol: str, timeframe: str, since: int, limit: int, per_call_limit: int = 300
    ) -> list[list]:
        """从 since 开始,反复调用 self.exchange.fetch_ohlcv 向前翻页,直到
        拿满 limit 条、追上"现在"、或交易所不再返回新数据为止。每一页都走
        _with_retry,单页失败时的重试语义与非分页调用完全一致。"""
        interval_ms = _timeframe_to_ms(timeframe)
        all_rows: list[list] = []
        current_since = since
        now_ms = self.clock.now_ms()

        while len(all_rows) < limit and current_since <= now_ms:
            page = self._with_retry(
                lambda s=current_since: self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=s, limit=per_call_limit
                ),
                f"fetch_ohlcv({symbol}, {timeframe}, since={current_since})",
            )
            if not page:
                break
            all_rows.extend(page)
            last_ts = page[-1][0]
            if last_ts <= current_since:
                # 交易所没有真正向前推进(比如恰好卡在同一个时间戳反复返回),
                # 停止分页,避免死循环——这不是"正常拿满数据"的路径,是防御。
                break
            current_since = last_ts + interval_ms
            if len(page) < per_call_limit:
                # 交易所返回的条数不足单页上限,说明这已经是能拿到的最后一页了。
                break

        return all_rows[:limit]

    # ------------------------------------------------------------------
    # M6 缺陷修复:资金费率历史分页拉取。fetch_ohlcv 早在 M5 就补了
    # _fetch_ohlcv_live_paginated,但 fetch_funding_rate_history 当时漏改,
    # 一直只发一次请求——同样被 OKX 单次请求的硬上限卡住(实测约300条)。
    # 资金费率每8小时结算一次,300条≈100天,而 fetch_ohlcv 能拿到完整
    # history_days=730 天。现有缓存(data_cache/funding_*.parquet)全部
    # 只有约96-99天就是这个缺陷的直接证据。修法与 OHLCV 同构:since 游标
    # 按固定的8小时结算周期推进(资金费率没有"timeframe"概念,间隔是
    # config.yaml funding.settle_hours_utc 隐含的常数,不是参数)。
    # ------------------------------------------------------------------

    _FUNDING_INTERVAL_MS = 8 * 60 * 60 * 1000  # 资金费率结算周期固定8小时

    def _fetch_funding_rate_history_live_paginated(
        self, symbol: str, since: int, limit: int, per_call_limit: int = 300
    ) -> list[dict]:
        """从 since 开始,反复调用 self.exchange.fetch_funding_rate_history 向前
        翻页,直到拿满 limit 条、追上"现在"、或交易所不再返回新数据为止。
        与 _fetch_ohlcv_live_paginated 同构:每一页都走 _with_retry,单页失败时
        的重试语义完全一致;翻页游标用"上一页最后一条记录的 timestamp + 固定
        结算周期"前进,而不是简单地数条数——避免交易所某一页恰好不足
        per_call_limit 但游标计算错误导致的空洞或重复。"""
        all_rows: list[dict] = []
        current_since = since
        now_ms = self.clock.now_ms()

        while len(all_rows) < limit and current_since <= now_ms:
            page = self._with_retry(
                lambda s=current_since: self.exchange.fetch_funding_rate_history(
                    symbol, since=s, limit=per_call_limit
                ),
                f"fetch_funding_rate_history({symbol}, since={current_since})",
            )
            if not page:
                break
            all_rows.extend(page)
            last_ts = page[-1].get("timestamp")
            if last_ts is None or last_ts <= current_since:
                # 交易所没有真正向前推进,停止分页,避免死循环(与 OHLCV 分页
                # 同款防御,理由见 _fetch_ohlcv_live_paginated)。
                break
            current_since = last_ts + self._FUNDING_INTERVAL_MS
            if len(page) < per_call_limit:
                # 交易所返回的条数不足单页上限,说明这已经是能拿到的最后一页。
                break

        return all_rows[:limit]

    # ------------------------------------------------------------------
    # M5 联网 shakedown 适配:公开历史数据归档回退路径(见模块 docstring)
    # ------------------------------------------------------------------

    @staticmethod
    def _archive_symbol(symbol: str) -> str:
        """ccxt 统一 symbol("BTC/USDT:USDT")转 Binance 归档用的扁平 symbol
        ("BTCUSDT")。"""
        base_quote = symbol.split(":")[0]  # "BTC/USDT"
        return base_quote.replace("/", "")

    @staticmethod
    def _months_between(since_ms: int, until_ms: int) -> list[tuple[int, int]]:
        """返回覆盖 [since_ms, until_ms] 区间需要下载的月度归档 (year, month)
        列表,按时间正序,不重复。"""
        if until_ms < since_ms:
            return []
        since_dt = pd.Timestamp(since_ms, unit="ms", tz="UTC")
        until_dt = pd.Timestamp(until_ms, unit="ms", tz="UTC")
        months: list[tuple[int, int]] = []
        year, month = since_dt.year, since_dt.month
        while (year, month) <= (until_dt.year, until_dt.month):
            months.append((year, month))
            month += 1
            if month > 12:
                month = 1
                year += 1
        return months

    def _download_archive_zip_csv_rows(self, url: str) -> list[list[str]] | None:
        """下载一个月度归档 zip,返回解析后的 CSV 行(不含表头)。归档不存在
        (比如请求的月份还没被 Binance 发布出来,常见于"最近一个月")时返回
        None,不当作错误重试——这是"历史缺口"这类适配问题里预期会遇到的
        正常情况,不是网络故障。"""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "alphaloop-data-pipeline"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read()
                zf = zipfile.ZipFile(io.BytesIO(raw))
                inner_name = zf.namelist()[0]
                with zf.open(inner_name) as f:
                    text = f.read().decode("utf-8")
                reader = csv.reader(io.StringIO(text))
                rows = list(reader)
                if not rows:
                    return []
                # 第一行是表头(open_time,open,high,...);数据行的第一列如果
                # 也恰好是 "open_time"/"calc_time" 这种非数字字符串,同样跳过
                # (Binance 部分归档文件的表头格式在不同月份间有过细微调整)。
                data_rows = [r for r in rows[1:] if r and r[0].strip().lstrip("-").isdigit()]
                return data_rows
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    logger.info("archive not found (likely not yet published): %s", url)
                    return None
                last_exc = exc
            except Exception as exc:  # noqa: BLE001 - 归档下载路径的通用重试
                last_exc = exc
            if attempt < self.max_retries:
                time.sleep(self.backoff_base_seconds * (2 ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc

    def fetch_ohlcv_from_public_archive(
        self, symbol: str, timeframe: str, since: int, until: int
    ) -> pd.DataFrame:
        """从 data.binance.vision 的月度归档拉取 [since, until] 区间的 K 线,
        不经过 api.binance.com/fapi.binance.com。返回列与 fetch_ohlcv 一致:
        [timestamp, open, high, low, close, volume]。"""
        archive_symbol = self._archive_symbol(symbol)
        all_rows: list[dict] = []
        for year, month in self._months_between(since, until):
            url = (
                f"{_BINANCE_VISION_BASE}/data/futures/um/monthly/klines/"
                f"{archive_symbol}/{timeframe}/{archive_symbol}-{timeframe}-{year:04d}-{month:02d}.zip"
            )
            rows = self._download_archive_zip_csv_rows(url)
            if rows is None:
                continue
            for r in rows:
                # CSV 列: open_time,open,high,low,close,volume,close_time,quote_volume,
                # count,taker_buy_volume,taker_buy_quote_volume,ignore
                all_rows.append({
                    "timestamp": int(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                })
        df = pd.DataFrame(all_rows, columns=OHLCV_COLUMNS)
        if not df.empty:
            df = df[(df["timestamp"] >= since) & (df["timestamp"] <= until)]
            df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_funding_rate_history_from_public_archive(
        self, symbol: str, since: int, until: int
    ) -> pd.DataFrame:
        """从 data.binance.vision 的月度归档拉取 [since, until] 区间的历史
        资金费率。返回列与 fetch_funding_rate_history 一致:
        [timestamp, funding_rate]。"""
        archive_symbol = self._archive_symbol(symbol)
        all_rows: list[dict] = []
        for year, month in self._months_between(since, until):
            url = (
                f"{_BINANCE_VISION_BASE}/data/futures/um/monthly/fundingRate/"
                f"{archive_symbol}/{archive_symbol}-fundingRate-{year:04d}-{month:02d}.zip"
            )
            rows = self._download_archive_zip_csv_rows(url)
            if rows is None:
                continue
            for r in rows:
                # CSV 列: calc_time,funding_interval_hours,last_funding_rate
                all_rows.append({"timestamp": int(r[0]), "funding_rate": float(r[2])})
        df = pd.DataFrame(all_rows, columns=FUNDING_COLUMNS)
        if not df.empty:
            df = df[(df["timestamp"] >= since) & (df["timestamp"] <= until)]
            df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        return df

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
            try:
                if since is not None:
                    # 分页拉取:交易所单次请求通常有硬上限(OKX实测300根/次,
                    # 无视传入的limit),不分页就只能拿到区间起点附近的一小段
                    # 数据,离"拿满since到现在"差得很远。since=None 时(调用方
                    # 要"最近一批"数据,不关心从哪个起点开始)沿用原来的单次
                    # 调用,不需要分页。
                    raw = self._fetch_ohlcv_live_paginated(symbol, timeframe, since, limit)
                else:
                    raw = self._with_retry(
                        lambda: self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit),
                        f"fetch_ohlcv({symbol}, {timeframe})",
                    )
                fetched = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
            except Exception as exc:  # noqa: BLE001 - 见下方回退路径说明
                # M5 适配:实时API不可达时,对"纯历史区间"请求回退到
                # data.binance.vision 的公开归档(见模块 docstring)。只在
                # since 有值 **且调用方显式开启了 enable_public_archive_fallback**
                # 时才走这条路径——since=None 代表调用方要的是"最新一批"数据,
                # 归档是有发布延迟的历史数据,冒充"最新"会是伪造实时性;而
                # 默认关闭是为了不让任何离线单测意外触发真实网络请求(见
                # __init__ 的详细说明)。
                if since is None or not self.enable_public_archive_fallback:
                    raise
                logger.warning(
                    "fetch_ohlcv(%s, %s) live API unreachable (%r); falling back to "
                    "data.binance.vision public archive for this historical range",
                    symbol, timeframe, exc,
                )
                until = self.clock.now_ms()
                fetched = self.fetch_ohlcv_from_public_archive(symbol, timeframe, since, until)
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
            try:
                if since is not None:
                    # 分页拉取(M6修复):理由同 fetch_ohlcv——since 有值代表
                    # 调用方要"从某个历史起点到现在"的完整区间,不分页只能拿到
                    # 起点附近约300条(≈100天),覆盖不了 history_days=730 天。
                    # since=None(要"最近一批")时沿用原来的单次调用。
                    raw = self._fetch_funding_rate_history_live_paginated(symbol, since, limit)
                else:
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
            except Exception as exc:  # noqa: BLE001 - 见 fetch_ohlcv 同款回退路径说明
                if since is None or not self.enable_public_archive_fallback:
                    raise
                logger.warning(
                    "fetch_funding_rate_history(%s) live API unreachable (%r); falling back to "
                    "data.binance.vision public archive for this historical range",
                    symbol, exc,
                )
                until = self.clock.now_ms()
                fetched = self.fetch_funding_rate_history_from_public_archive(symbol, since, until)
            merged = self._merge_dedupe(cached, fetched)
            self._save_parquet(cache_path, merged)
            cached = merged

        result = cached
        if since is not None:
            result = result[result["timestamp"] >= since]
        if limit is not None:
            result = result.head(limit)
        return result.reset_index(drop=True)
