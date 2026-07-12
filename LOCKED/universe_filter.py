"""
universe_filter.py —— 币池筛选(§2.2b)。

职责:每周一 UTC 00:00 运行(调度由外部 scheduler 负责,本模块只实现筛选逻辑本身):
1. 通过 ccxt 拉取 USDT 本位永续合约(swap)全部交易对及其元数据
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

M5 shakedown 联网适配记录(只动本文件里的交易所连接细节,过滤规则本身
一处未改):
- 默认交易所由 binance 改为 okx——本沙箱网络环境下 api.binance.com/
  fapi.binance.com 对该出口IP返回 HTTP 451(Binance 自身的地域限制,见
  LOCKED/data_pipeline.py 模块 docstring 的详细记录),经用户确认后切换。
  ccxt options.defaultType 相应从 "future"(币安命名)改为 "swap"(OKX/
  多数交易所的命名)。
- ccxt 的 requests.Session 默认 trust_env=False,不会自动读取本沙箱已配置
  好的 HTTP_PROXY/HTTPS_PROXY 环境变量,新增 `exchange.session.trust_env
  = True`。
- ccxt 统一字段 ticker["quoteVolume"] 在 OKX 的 USDT 永续 ticker 上恒为
  None(只有 baseVolume,单位是合约张数)。新增
  `_extract_quote_volume_usdt()`:优先用 quoteVolume,取不到时退化为
  交易所原始字段 info.volCcy24h(标的币计价的24h成交量)× 最新价,近似
  换算成 USDT 成交额——数量级已核对(BTC约2.6十亿美元/日,合理)。这只是
  换一种方式取到同一个"USDT计价24h成交额"数字,min_24h_volume_usdt 这条
  筛选规则的语义没有变。

上市时间获取的 fallback 策略(需在此明确记录,因为不同交易所的 ccxt 适配器
并不总是提供统一字段):
- 优先使用 ccxt 统一市场结构里的 market["created"](若交易所适配器提供,
  为 UTC 毫秒时间戳;OKX 已确认提供)。
- 其次尝试币安期货 exchangeInfo 的交易所专有字段
  market["info"]["onboardDate"](UTC 毫秒时间戳字符串/数字;OKX 用不上
  这个字段,但 market["created"] 已经够用,保留这条 fallback 不影响OKX,
  纯粹是给未来换回币安或接入第三个交易所时的兼容性)。
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

        # M5 shakedown 适配:本沙箱网络环境下 api.binance.com/fapi.binance.com
        # 对该出口IP返回 HTTP 451(Binance 自身的地域限制,见
        # LOCKED/data_pipeline.py 模块 docstring 开头的联网适配记录),经
        # 用户确认后由 binance 切换到 OKX。OKX 的 ccxt options.defaultType
        # 用 "swap"(不是币安的 "future")。
        # 只走公开数据模式:不设置 apiKey / secret。
        exchange = ccxt.okx(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",  # USDT 本位永续合约(swap)
                },
            }
        )
        # 同 data_pipeline.py 的适配:ccxt 的 requests.Session 默认
        # trust_env=False,不会自动读取 HTTP_PROXY/HTTPS_PROXY 环境变量。
        if hasattr(exchange, "session"):
            exchange.session.trust_env = True
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
    @staticmethod
    def _extract_quote_volume_usdt(ticker: dict) -> float:
        """从 ticker 里取 24h 成交额(USDT 计价)。

        M5 shakedown 适配:ccxt 统一字段 quoteVolume 在部分交易所/市场类型
        组合下不会被填充——实测 OKX 的 USDT 本位永续 ticker 就是这样,
        quoteVolume 恒为 None,只有 baseVolume(单位是合约张数,不能直接
        当USDT成交额用)。这里的回退:交易所原始字段 info.volCcy24h 是
        OKX v5 API 文档里"过去24小时成交量,以标的币计价"(即 BTC 张数,
        不是 USDT),乘以最新价格 (ticker["last"]) 近似换算成 USDT 名义
        成交额——数量级校验过(BTC 现在约 6.4 万美元,volCcy24h≈4万BTC,
        换算约26亿美元,是真实、合理的BTC永续24h成交量)。这是"字段名对不上"
        这类适配问题,不改变 min_24h_volume_usdt 这条筛选规则本身的语义
        (仍然是"USDT计价24h成交额 >= 门槛")。
        """
        volume = ticker.get("quoteVolume")
        if volume is not None:
            return float(volume)

        info = ticker.get("info") or {}
        raw_base_volume = info.get("volCcy24h")
        last_price = ticker.get("last")
        if raw_base_volume is not None and last_price:
            try:
                return float(raw_base_volume) * float(last_price)
            except (TypeError, ValueError):
                pass

        return 0.0

    def fetch_candidates(self) -> list[dict]:
        """拉取 USDT 本位永续全部交易对的候选元数据(交易所见 config.yaml
        data.exchange / universe_rule.source)。

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
            volume_24h_usdt = self._extract_quote_volume_usdt(ticker)

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
        """对候选列表应用过滤规则,返回合格 symbol 列表(已排序)。

        M5 shakedown 适配:config.yaml 里 `100e6` 这种没有小数点的科学计数法
        写法,PyYAML 默认解析器按 YAML 1.1 规范会把它读成**字符串** "100e6"
        而不是浮点数(必须写成 "100.0e6" 或 "1e+8" 才会被识别成 float——这是
        PyYAML 一个广为人知的坑)。首次用真实网络数据跑 refresh() 时才会真正
        触发这条比较(离线单测传的是手写的 dict,天然是正确类型),属于"第一次
        接触真实数据/真实config读取路径"才会暴露的适配问题,不是筛选规则本身
        变了——阈值数值原样不动,只是把"字符串还是数字"这件事在比较前防御性
        地摆平,不去改 config.yaml 的写法。
        """
        rule = self.universe_rule
        min_volume = float(rule.get("min_24h_volume_usdt", 0))
        min_listing_days = int(rule.get("min_listing_days", 0))
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
