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

M6 生产实测缺陷修复记录(2026年真实点火/服务器实测发现,授权范围内的
修改,只动本文件,LOCKED区其余业务逻辑一行未改):
- 缺陷1:fetch_funding_rate_history 的分页在真实OKX上无效——实测请求
  730天资金费率,始终只拿到最近92-99天(~277-320条,正好是旧的单次上限)。
  根因:ccxt/okx.py 的 fetch_funding_rate_history 只把 since 映射成 OKX
  的 before 参数(只圈下界,不圈上界),而 fetch_ohlcv 的实现把 since 同时
  映射成 before+after(双边界)。只圈下界的请求,OKX 永远优先返回"满足下界
  条件的最新一批"记录,把 since 游标继续向前推毫无意义。修法:改成反向
  翻页,从"现在"开始用 params={"after": <上一页最早时间戳>} 向历史深处
  推进,直到覆盖 since 或历史耗尽。详见 _fetch_funding_rate_history_live_paginated
  的实现注释。
- 缺陷2:since 早于币种上市日时 OHLCV 拉到0根——CL/HYPE/LAB/MU/PUMP/
  SNDK/XAU/ZEC 这类上市不满 history_days 的币种,正向分页第一页(窗口
  完全落在"上市前")如实返回空,原代码把这种"起点选早了"和"正常拿完了"
  混为一谈直接终止。修法:仅当第一页为空时,用 params={"until": <上一页
  最早时间戳>} 反向翻页探测并补齐"上市日→现在"的全部历史,等效于规格书
  M6 验收标准"或该币种上市以来全部历史,以较短者为准"。详见
  _fetch_ohlcv_backward_from_listing 的实现注释。

M9 感官扩展记录(研究总监备忘录里要的两只新"眼睛",授权范围仅本文件新增
两个拉取方法,不改既有 fetch_ohlcv/fetch_funding_rate_history 一行逻辑):
- fetch_spot_ohlcv(symbol, timeframe, since, limit):给策略提供现货K线原料,
  用于计算"现货溢价"basis = perp_close/spot_close - 1(basis由策略自己算,
  本方法只负责把现货K线拉回来、缓存——与fetch_ohlcv同一个"数据管道只管原料,
  不管指标"的分工原则一致)。symbol 入参沿用永续统一格式("BTC/USDT:USDT"),
  内部按 "://"[0] 之前的 "BASE/QUOTE" 部分转成现货统一格式("BTC/USDT")——
  ccxt统一symbol语法里,":"之后是结算币后缀,永续/现货共用同一个base/quote
  前缀。缓存文件名故意加"spot_"前缀与永续K线区分(ohlcv_spot_{symbol}_
  {timeframe}.parquet vs ohlcv_{symbol}_{timeframe}.parquet),避免两套
  数据混进同一个缓存文件互相污染。复用既有的 _fetch_ohlcv_live_paginated/
  _with_retry/_load_parquet/_merge_dedupe/_save_parquet 几个已经独立测试过
  的私有工具函数,不重新发明一套分页/缓存逻辑;唯一的行为差异是不接
  data.binance.vision 归档回退(那条回退路径的URL模板是币安永续合约归档,
  对现货symbol没有意义,接上去反而可能在归档回退开启时错误地把永续归档
  数据当成现货返回)。该现货交易对没有上市/交易所不认识这个symbol/网络
  失败时一律 log warning 并返回空DataFrame(不抛异常)——这是"增强感官
  输入"的既有约定(与fetch_funding_rate_history的调用方容错处理同源:拿不到
  不该打断整个决策/回测流程)。
- fetch_open_interest_history(symbol, since, limit):持仓量(OI)历史,给
  策略识别"价涨量增/价涨量缩"这类持仓结构信号当原料。实测读本机
  site-packages/ccxt/okx.py(约7445-7507行)确认的真实签名与返回结构:
      def fetch_open_interest_history(self, symbol: str, timeframe='1d',
                                       since: Int = None, limit: Int = None,
                                       params={}):
  OKX没有一个"OI历史"的实时API,走的是"rubik"统计接口
  (publicGetRubikStatContractsOpenInterestVolume,约7491行),响应是
  [timestamp_str, open_interest_usd_str, volume_usd_str] 三元组列表(约
  7493-7504行注释),经 parse_open_interests_history -> safe_open_interest
  (base/exchange.py 约7867-7880行)规整成dict列表,每条至少含 timestamp/
  openInterestValue(约定为USD计价的持仓量,非期权市场下由三元组第2个
  元素填入,见 parse_open_interest 约7544行 openInterestValue = safe_number
  (interest, 1))/openInterestAmount(仅期权市场填充,约7541行)两个字段。
  本方法优先取 openInterestValue,该字段为空时(比如某些非OKX交易所/期权
  市场只填 openInterestAmount)回退取 openInterestAmount——这是本次实现
  自行拍板的字段选择,记录在此供日后review。返回列固定为
  [timestamp, open_interest],parquet缓存到 oi_{symbol}.parquet。
  容错设计(两层):
    1. getattr(self.exchange, "fetch_open_interest_history", None) 探测该
       交易所对象是否真的实现了这个方法——测试里注入的假交易所/未来换成
       不支持OI历史的交易所时,直接优雅降级为空DataFrame + log warning,
       不尝试调用一个根本不存在的方法。
    2. 探测通过后调用仍然失败(交易所返回NotSupported/网络异常等)同样
       log warning 后返回空DataFrame,不向上抛异常。
  OKX的OI历史深度实测/文档均显示只有几十天(远短于fetch_funding_rate_
  history实测的约92天硬上限,更远短于fetch_ohlcv的730天),这是交易所
  自身的数据保留策略限制,不是本侧代码能绕开的分页缺陷(不同于M6资金
  费率分页那种"ccxt/OKX参数映射错误"的真实bug)——因此本方法特意不假设
  "缓存应该覆盖到since"就一定能拿满,cached_max/cached_min的"是否需要
  重新拉取"判断与fetch_funding_rate_history同一套逻辑,短历史只是让
  need_fetch经常判定为True(重新拉取但交易所仍然只给这么多),不是错误。
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
OI_COLUMNS = ["timestamp", "open_interest"]

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

    def _spot_cache_path(self, spot_symbol: str, timeframe: str) -> Path:
        """现货K线缓存文件名故意加"spot_"前缀,与永续K线缓存
        (_symbol_cache_path 产出的 ohlcv_{symbol}_{timeframe}.parquet)区分
        开,避免落到同一个文件里互相覆盖/污染(见 fetch_spot_ohlcv 的模块
        docstring 说明)。"""
        safe_symbol = spot_symbol.replace("/", "-").replace(":", "_")
        return self.cache_dir / f"ohlcv_spot_{safe_symbol}_{timeframe}.parquet"

    def _oi_cache_path(self, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "-").replace(":", "_")
        return self.cache_dir / f"oi_{safe_symbol}.parquet"

    @staticmethod
    def _perp_to_spot_symbol(symbol: str) -> str:
        """永续统一symbol("BTC/USDT:USDT") -> 现货统一symbol("BTC/USDT")。
        ccxt统一symbol语法里":"之后是结算币后缀,永续/现货共用同一个
        base/quote前缀,截掉结算币后缀就是对应的现货交易对写法。"""
        return symbol.split(":")[0]

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
        _with_retry,单页失败时的重试语义与非分页调用完全一致。

        M6 缺陷2 修复(2026年真实点火时发现):CL/HYPE/LAB/MU/PUMP/SNDK/XAU/
        ZEC 这类上市不满 history_days(730天)的币种,since 早于该币种真实
        上市日,第一页请求(窗口完全落在"上市前")如实返回空——原代码把
        "第一页就是空"和"交易所没有更多数据了(正常收尾)"当成同一件事处理,
        直接 break,最终 0 根K线。这两种"空"含义不同:后者是"已经拿到一些
        数据、现在追到头了",前者是"起点选早了,一根都还没拿到"。只有第一页
        (all_rows 还是空的)才需要走下面的"探测上市日"分支;非第一页的空页
        仍然按原逻辑正常结束分页,不受影响。
        """
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
                if not all_rows:
                    # 第一页就是空的:since 很可能早于该币种的上市日(规格书
                    # M6 验收标准:"或该币种上市以来全部历史,以较短者为准")。
                    # 探测该币种是否真的有数据,有的话反向翻页补齐"上市日→
                    # 现在"的完整历史。
                    return self._fetch_ohlcv_backward_from_listing(
                        symbol, timeframe, limit, per_call_limit
                    )[:limit]
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

    def _fetch_ohlcv_backward_from_listing(
        self, symbol: str, timeframe: str, limit: int, per_call_limit: int = 300
    ) -> list[list]:
        """M6 缺陷2 修复:当正向分页第一页就是空的(since 早于上市日)时,
        从"现在"开始反向翻页,补齐"该币种上市以来的全部历史"。

        游标语义取自实测读到的 ccxt/okx.py fetch_ohlcv 源码(约2608-2622行):
            if since is not None:
                ...
                request['before'] = startTime            # since-1,下界
                request['after'] = self.sum(since, durationInMilliseconds * limit)  # 上界
            until = self.safe_integer(params, 'until')
            if until is not None:
                request['after'] = until                  # 显式上界游标
        也就是说 ccxt 把 since 同时映射成 before(下界)+ after(上界)两个
        边界,没有单独暴露"只给上界、不给下界"的高层参数;但官方支持的
        params.until 会直接覆盖 request['after'],语义正是"只拿这个时间戳
        之前的K线"。这里固定传 since=None(不设下界),用 params={"until":
        cursor} 把游标设到"上一页最早一根K线的时间戳",从"现在"向历史深处
        反向翻页,直到交易所返回空页(说明已经翻过上市日,历史耗尽)为止。
        """
        all_rows: list[list] = []
        cursor: int | None = None
        seen_oldest: int | None = None

        while len(all_rows) < limit:
            params = {"until": cursor} if cursor is not None else {}
            page = self._with_retry(
                lambda p=params: self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=None, limit=per_call_limit, params=p
                ),
                f"fetch_ohlcv({symbol}, {timeframe}, until={cursor}) [listing-date probe]",
            )
            if not page:
                # 没有更早的数据了——翻过了上市日(或该币种本来就没有任何
                # 数据),历史耗尽是预期的正常终止条件,不是错误。
                break
            all_rows.extend(page)
            oldest_ts = min(row[0] for row in page)
            if seen_oldest is not None and oldest_ts >= seen_oldest:
                # 游标没有真正向历史深处推进,停止分页,避免死循环(与正向
                # 分页同款防御)。
                break
            seen_oldest = oldest_ts
            cursor = oldest_ts

        if not all_rows:
            return []
        dedup: dict[int, list] = {row[0]: row for row in all_rows}
        return sorted(dedup.values(), key=lambda row: row[0])

    # ------------------------------------------------------------------
    # M6 缺陷1 修复(2026年真实点火实测确认):资金费率历史分页拉取。
    #
    # 原实现(把 since 游标一路向前推、原样传给 ccxt 的 since= 形参)在真实
    # OKX 上线后实测发现分页形同虚设:请求13个币种730天资金费率,返回的
    # 全部仍然只有最近92-99天(~277-320条,正好是旧的单次上限)。
    #
    # 根因(读 ccxt/okx.py fetch_funding_rate_history 源码,约2683-2690行
    # 确认):
    #     request = {'instId': market['id']}
    #     if since is not None:
    #         request['before'] = max(since - 1, 0)
    #     if limit is not None:
    #         request['limit'] = limit
    #     response = self.publicGetPublicFundingRateHistory(self.extend(request, params))
    # ccxt 把 since 只映射成 OKX 的 before 参数——OKX 官方语义里 before 只
    # 圈定"下界"(只返回比 before 更新的记录),不圈定上界。对比同一份源码
    # 里 fetch_ohlcv 的实现(约2617-2619行):since 同时映射到 before(下界)
    # 和 after(上界)两个参数,形成一个双边界窗口,这正是 fetch_ohlcv 分页
    # 能生效、funding_rate_history 分页失效的关键差异。只圈下界不圈上界的
    # 请求,OKX 服务端永远优先返回"满足下界条件的最新一批"记录——所以无论
    # since 传得多早,只要早于"现在往前数 per_call_limit 条"这个窗口,拿到
    # 的都是同一批最新记录;把 since 游标继续向前推、再发一次请求,拿到的
    # 还是同一批"最新"——分页循环因此形同虚设,这与实测现象(始终卡在约
    # 300条/92-99天)完全吻合。
    #
    # 修法:改成反向翻页——从"现在"开始,每页取回后用**本页最早一条记录的
    # 时间戳**作为下一页的游标,向历史深处推进。OKX 没有给 fetch_funding_
    # rate_history 暴露 fetch_ohlcv 那样的 params.until 高层封装,但 params
    # 会被 self.extend(request, params) 直接合并进请求体,可以绕过 since=
    # 形参,直接用 params={"after": <游标>}(OKX 语义:只返回早于该时间戳的
    # 记录)把上界游标透传给交易所。翻页直到某一页最早时间戳已经 <= since
    # (覆盖到请求起点)、或交易所返回空页(历史耗尽)、或游标未能真正推进
    # (防御死循环)为止。
    # ------------------------------------------------------------------

    def _fetch_funding_rate_history_live_paginated(
        self, symbol: str, since: int, limit: int, per_call_limit: int = 300
    ) -> list[dict]:
        """从"现在"开始反向翻页,直到覆盖 since 或交易所耗尽历史为止。
        每一页都走 _with_retry,单页失败时的重试语义与非分页调用完全一致。
        返回的记录不假设全局有序(见下方 dedupe+sort),调用方
        fetch_funding_rate_history 里的 _merge_dedupe 也会再次按 timestamp
        排序去重,双重保险。"""
        all_rows: list[dict] = []
        cursor: int | None = None
        seen_oldest: int | None = None

        while len(all_rows) < limit:
            params = {"after": cursor} if cursor is not None else {}
            page = self._with_retry(
                lambda p=params: self.exchange.fetch_funding_rate_history(
                    symbol, since=None, limit=per_call_limit, params=p
                ),
                f"fetch_funding_rate_history({symbol}, after={cursor})",
            )
            if not page:
                # 没有更早的数据了——可能是真的历史耗尽,也可能是该币种上市
                # 不满 since 要求的天数(M6 缺陷2 同款场景),都是预期的正常
                # 终止条件,不是错误。
                break
            all_rows.extend(page)
            page_timestamps = [row.get("timestamp") for row in page if row.get("timestamp") is not None]
            if not page_timestamps:
                break
            oldest_ts = min(page_timestamps)
            if seen_oldest is not None and oldest_ts >= seen_oldest:
                # 游标没有真正向历史深处推进,停止分页,避免死循环(与
                # OHLCV 分页同款防御)。
                break
            seen_oldest = oldest_ts
            if oldest_ts <= since:
                # 已经翻到覆盖 since 起点的一页,足够了——更早的数据不需要。
                break
            cursor = oldest_ts

        if not all_rows:
            return []
        dedup: dict[int, dict] = {}
        for row in all_rows:
            ts = row.get("timestamp")
            if ts is not None:
                dedup[ts] = row
        return sorted(dedup.values(), key=lambda row: row["timestamp"])[:limit]

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

    # ------------------------------------------------------------------
    # M9 感官扩展:现货K线(见模块 docstring "M9 感官扩展记录"）
    # ------------------------------------------------------------------

    def fetch_spot_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        since: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """现货K线,供策略计算"现货溢价"basis = perp_close/spot_close - 1
        当原料(basis由策略自己算,本方法只负责把现货K线拉回来、缓存)。

        symbol 入参沿用永续统一格式("BTC/USDT:USDT"),内部转换成现货统一
        格式("BTC/USDT")后走与 fetch_ohlcv 完全同源的分页/缓存机制(复用
        _fetch_ohlcv_live_paginated/_with_retry/_load_parquet/_merge_dedupe/
        _save_parquet,不重新发明一套逻辑),但缓存文件名与永续区分
        (见 _spot_cache_path)。

        与 fetch_ohlcv 唯一的行为差异:不接 data.binance.vision 归档回退
        (那条回退路径的URL模板是币安永续合约月度归档,对现货symbol没有
        意义)。该现货交易对没有上市/交易所不认识这个symbol/网络请求失败
        时,一律 log warning 并返回空DataFrame,不向上抛异常——这是"增强
        感官输入"的既有约定,拉不到不该打断整个决策/回测流程。
        """
        spot_symbol = self._perp_to_spot_symbol(symbol)
        cache_path = self._spot_cache_path(spot_symbol, timeframe)
        cached = self._load_parquet(cache_path, OHLCV_COLUMNS)

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
                    raw = self._fetch_ohlcv_live_paginated(spot_symbol, timeframe, since, limit)
                else:
                    raw = self._with_retry(
                        lambda: self.exchange.fetch_ohlcv(spot_symbol, timeframe=timeframe, since=since, limit=limit),
                        f"fetch_ohlcv({spot_symbol}, {timeframe}) [spot]",
                    )
                fetched = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
            except Exception as exc:  # noqa: BLE001 - 现货没上市/未知symbol/网络失败,一律降级为"没有现货数据"
                logger.warning(
                    "fetch_spot_ohlcv(%s -> %s) unavailable, treating as no spot data: %r",
                    symbol, spot_symbol, exc,
                )
                fetched = pd.DataFrame(columns=OHLCV_COLUMNS)
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

    # ------------------------------------------------------------------
    # M9 感官扩展:持仓量(OI)历史(见模块 docstring "M9 感官扩展记录"，
    # 含读 ccxt/okx.py 源码确认的真实方法签名/返回结构引用)。
    # ------------------------------------------------------------------

    def fetch_open_interest_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """持仓量(OI)历史,列: [timestamp, open_interest]。本地缓存到
        parquet(oi_{symbol}.parquet),合并去重后返回,need_fetch判断口径
        与 fetch_funding_rate_history 一致。

        底层调用 ccxt 的 fetch_open_interest_history(真实签名见本文件模块
        docstring"M9 感官扩展记录"一节,已读 site-packages/ccxt/okx.py
        约7445-7507行核实:OKX 走 publicGetRubikStatContractsOpenInterestVolume
        这个"rubik"统计接口,不是普通的公开行情实时端点)。open_interest列
        优先取 ccxt 规整后的 openInterestValue,为空时回退取
        openInterestAmount(自行拍板的字段选择,见模块docstring）。

        两层容错(交易所不支持/拉取失败都不应该抛异常打断调用方):
          1. getattr 探测 self.exchange 是否真的有这个方法——没有的话直接
             log warning + 返回空DataFrame,不尝试调用不存在的方法。
          2. 调用抛异常(NotSupported/网络异常等)同样 log warning + 返回
             空DataFrame。
        OKX的OI历史深度实测/文档均显示明显短于fetch_ohlcv/
        fetch_funding_rate_history(只有几十天量级)——这是交易所自身数据
        保留策略的限制,不是本侧代码缺陷,代码需要容忍这种"重新拉取但仍然
        拿不满since"的短历史,而不是当成错误处理。
        """
        cache_path = self._oi_cache_path(symbol)
        cached = self._load_parquet(cache_path, OI_COLUMNS)

        need_fetch = True
        if since is not None and not cached.empty:
            cached_min = int(cached["timestamp"].min())
            cached_max = int(cached["timestamp"].max())
            covers_since = cached_min <= since
            enough_rows = len(cached[cached["timestamp"] >= since]) >= min(limit, len(cached))
            if covers_since and enough_rows and cached_max >= since:
                need_fetch = False

        if need_fetch:
            fetch_fn = getattr(self.exchange, "fetch_open_interest_history", None)
            if fetch_fn is None:
                logger.warning(
                    "fetch_open_interest_history(%s): exchange %r does not implement "
                    "fetchOpenInterestHistory, returning empty DataFrame",
                    symbol, self.exchange_id,
                )
                fetched = pd.DataFrame(columns=OI_COLUMNS)
            else:
                # OKX真实探测结果(2026-07-16,生产服务器直连验证):该接口只
                # 支持 5m/1h/1d 三种粒度;1h 只允许查最近约30天,1d 可查到约
                # 260天,窗口超限直接报 50030 "Illegal time range" 而不是返回
                # 空——所以按请求窗口自适应选粒度,窗口仍超限时逐级收缩since
                # 重试,拿到多少算多少(短历史是交易所限制,不是错误)。
                now_ms = self.clock.now_ms()
                window_ms = (now_ms - since) if since is not None else 30 * 86_400_000
                oi_timeframe = "1d" if window_ms > 25 * 86_400_000 else "1h"
                attempt_sinces = [since]
                if since is not None:
                    # 逐级收缩:请求窗口 -> 240天 -> 25天(1d粒度约260天上限、
                    # 1h粒度约30天上限,各留余量)
                    for clamp_days in (240, 25):
                        clamp_since = now_ms - clamp_days * 86_400_000
                        if clamp_since > since:
                            attempt_sinces.append(clamp_since)
                fetched = pd.DataFrame(columns=OI_COLUMNS)
                for attempt_since in attempt_sinces:
                    tf = "1d" if (attempt_since is None or now_ms - attempt_since > 25 * 86_400_000) else "1h"
                    try:
                        raw = self._with_retry(
                            lambda s=attempt_since, t=tf: fetch_fn(symbol, timeframe=t, since=s, limit=limit),
                            f"fetch_open_interest_history({symbol},{tf})",
                        )
                        rows = [
                            {
                                "timestamp": entry.get("timestamp"),
                                "open_interest": (
                                    entry.get("openInterestValue")
                                    if entry.get("openInterestValue") is not None
                                    else entry.get("openInterestAmount")
                                ),
                            }
                            for entry in raw
                            if entry.get("timestamp") is not None
                        ]
                        fetched = pd.DataFrame(rows, columns=OI_COLUMNS)
                        break  # 本级窗口成功,不再收缩
                    except Exception as exc:  # noqa: BLE001 - OI是增强感官,拉不到不该打断调用方
                        logger.warning(
                            "fetch_open_interest_history(%s, tf=%s, since=%s) failed (%r); "
                            "尝试收缩窗口重试(OKX: 1h粒度约30天/1d粒度约260天上限)",
                            symbol, tf, attempt_since, exc,
                        )
                        continue
            merged = self._merge_dedupe(cached, fetched)
            self._save_parquet(cache_path, merged)
            cached = merged

        result = cached
        if since is not None:
            result = result[result["timestamp"] >= since]
        if limit is not None:
            result = result.head(limit)
        return result.reset_index(drop=True)
