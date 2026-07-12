"""
universe_filter.py —— 币池筛选(§2.2b)。

职责:每周一 UTC 00:00 运行(调度由外部 scheduler 负责,本模块只实现筛选逻辑本身):
1. 通过 ccxt 拉取币安 USDT 本位永续合约(swap)全部交易对及其元数据
   (24h 成交额、上市时间),只用公开端点。
2. 过滤规则(见 config.yaml: universe_rule):
   - 24h 成交额 >= min_24h_volume_usdt
   - 上市天数 >= min_listing_days
   - 剔除 blacklist 中的交易对
   - 剔除杠杆代币(base asset 以 UP/DOWN/BULL/BEAR 结尾,大小写不敏感)
3. 写出 universe_active.json,simulator.py 只认这个文件,格式:
   {"generated_at": <UTC ms timestamp>, "symbols": ["BTC/USDT:USDT", ...]}

铁律:
- 全项目没有任何真实交易所下单代码,ccxt 只用公开行情接口。
- 本模块的 ccxt 交易所实例绝不配置 apiKey/secret,绝不调用任何私有端点。
- agent 无权修改 universe_rule 与 blacklist(§7),本模块只读 config。

上市时间获取的 fallback 策略(需在此明确记录,因为 ccxt/币安并不总是
提供统一字段):
- 优先使用 ccxt 统一市场结构里的 market["created"](若交易所适配器提供,
  为 UTC 毫秒时间戳)。
- 其次尝试币安期货 exchangeInfo 的交易所专有字段
  market["info"]["onboardDate"](UTC 毫秒时间戳字符串/数字)。
- 若以上两者都拿不到上市时间,则**保守处理为"未知即太新,排除"**——
  也就是把 listing_days 设为 -1(恒小于 min_listing_days,必然被过滤掉)。
  这与"土狗/新上线拉盘币进不来"的设计目标一致:宁可漏掉一个未知上市时间
  的正常币,也不放一个真正的新币进来。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from LOCKED.clock import Clock, SystemClock

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "universe_active.json"

# 杠杆代币后缀(大小写不敏感,匹配 base asset 结尾),如 BTCUP / ETHBULL。
_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

# 未知上市时间时的哨兵值:恒小于任何合理的 min_listing_days,保证被过滤掉。
UNKNOWN_LISTING_DAYS = -1


def _is_leveraged_token(symbol: str) -> bool:
    """判断 symbol 的 base asset 是否为杠杆代币(UP/DOWN/BULL/BEAR 后缀)。"""
    base = symbol.split("/")[0].upper()
    return any(base.endswith(suffix) for suffix in _LEVERAGED_SUFFIXES)


class UniverseFilter:
    """封装 ccxt 公开市场元数据拉取 + 纯函数过滤 + 落盘。"""

    def __init__(self, config: dict, exchange: Any | None = None, clock: Clock | None = None) -> None:
        self.config = config
        self.universe_rule: dict = config.get("universe_rule", {})
        # M5: 全项目单一时间源(见 LOCKED/clock.py)。默认 SystemClock 保持既有
        # 生产行为不变。
        self.clock: Clock = clock if clock is not None else SystemClock()

        if exchange is not None:
            # 测试/依赖注入路径:调用方直接传入一个(通常是 mock 的)交易所对象。
            self.exchange = exchange
        else:
            self.exchange = self._build_exchange()

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @staticmethod
    def _build_exchange() -> Any:
        import ccxt

        # 只走公开数据模式:不设置 apiKey / secret。
        exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",  # USDT 本位永续合约(swap)
                },
            }
        )
        return exchange

    # ------------------------------------------------------------------
    # 上市时间解析
    # ------------------------------------------------------------------
    @staticmethod
    def _get_listing_timestamp_ms(market: dict) -> int | None:
        """从 ccxt market 结构里提取上市时间(UTC 毫秒)。取不到则返回 None。"""
        created = market.get("created")
        if created:
            try:
                return int(created)
            except (TypeError, ValueError):
                pass

        info = market.get("info") or {}
        onboard_date = info.get("onboardDate")
        if onboard_date:
            try:
                return int(onboard_date)
            except (TypeError, ValueError):
                pass

        return None

    # ------------------------------------------------------------------
    # 网络拉取(§2.2b step 1)
    # ------------------------------------------------------------------
    def fetch_candidates(self) -> list[dict]:
        """拉取币安 USDT 本位永续全部交易对的候选元数据。

        返回: [{"symbol": str, "volume_24h_usdt": float, "listing_days": int}, ...]
        listing_days 未知时使用 UNKNOWN_LISTING_DAYS 哨兵值(见模块 docstring)。
        """
        markets = self.exchange.load_markets()
        tickers = self.exchange.fetch_tickers()
        now_ms = self.clock.now_ms()

        candidates: list[dict] = []
        for symbol, market in markets.items():
            if not market:
                continue
            # 只要 USDT 本位永续合约(swap + linear + quote=USDT)。
            if not market.get("swap"):
                continue
            if market.get("quote") != "USDT":
                continue
            if market.get("linear") is False:
                continue

            ticker = tickers.get(symbol) or {}
            volume = ticker.get("quoteVolume")
            volume_24h_usdt = float(volume) if volume is not None else 0.0

            listing_ts = self._get_listing_timestamp_ms(market)
            if listing_ts is None:
                listing_days = UNKNOWN_LISTING_DAYS
            else:
                listing_days = int((now_ms - listing_ts) / (24 * 60 * 60 * 1000))

            candidates.append(
                {
                    "symbol": symbol,
                    "volume_24h_usdt": volume_24h_usdt,
                    "listing_days": listing_days,
                }
            )

        return candidates

    # ------------------------------------------------------------------
    # 纯函数过滤(§2.2b step 2)——不发网络请求,便于单测。
    # ------------------------------------------------------------------
    def apply_filters(
        self,
        candidates: list[dict],
        blacklist: list[str] | None = None,
    ) -> list[str]:
        """对候选列表应用过滤规则,返回合格 symbol 列表(已排序)。"""
        rule = self.universe_rule
        min_volume = rule.get("min_24h_volume_usdt", 0)
        min_listing_days = rule.get("min_listing_days", 0)
        effective_blacklist = set(blacklist if blacklist is not None else rule.get("blacklist", []) or [])

        result: list[str] = []
        for candidate in candidates:
            symbol = candidate["symbol"]

            if symbol in effective_blacklist:
                continue
            if _is_leveraged_token(symbol):
                continue
            if candidate.get("volume_24h_usdt", 0) < min_volume:
                continue
            if candidate.get("listing_days", UNKNOWN_LISTING_DAYS) < min_listing_days:
                continue

            result.append(symbol)

        return sorted(result)

    # ------------------------------------------------------------------
    # 落盘(§2.2b step 3)
    # ------------------------------------------------------------------
    def refresh(self, output_path: str | Path | None = None) -> dict:
        """fetch_candidates -> apply_filters -> 写 universe_active.json,返回写入内容。"""
        candidates = self.fetch_candidates()
        symbols = self.apply_filters(candidates)

        payload = {
            "generated_at": self.clock.now_ms(),
            "symbols": symbols,
        }

        path = Path(output_path) if output_path is not None else _DEFAULT_OUTPUT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("universe_filter.refresh: wrote %d symbols to %s", len(symbols), path)
        return payload
