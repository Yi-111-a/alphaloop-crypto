"""
scripts/backfill_history.py —— M6 一次性历史数据回填脚本。

背景:LOCKED/data_pipeline.py 的 fetch_funding_rate_history 在本次改造前
存在分页缺陷(单次请求被交易所硬上限卡住,只能拿到约100天资金费率历史,
详见该文件模块 docstring 里的 M6 记录),导致 data_cache/funding_*.parquet
一直只有约96-99天数据,而 fetch_ohlcv 早就能拿到完整 730 天。分页缺陷已经
在本次改造里修好,本脚本负责一次性把现有 universe 的历史缓存补齐到完整
history_days(config.yaml data.history_days,当前 730 天)。

跑法(**本机网络时断时续,不要在本机跑这个脚本**——按用户/任务要求,这个
脚本只负责"写出来",实际执行留到网络稳定的服务器上):
    cd alphaloop
    python scripts/backfill_history.py                       # 用 universe_active.json 里的全部标的
    python scripts/backfill_history.py --symbols BTC/USDT:USDT ETH/USDT:USDT
    python scripts/backfill_history.py --history-days 730 --timeframe 4h

可重复运行:DataPipeline.fetch_ohlcv / fetch_funding_rate_history 内部自己
做"缓存已覆盖请求区间则跳过网络请求"的判断(见 data_pipeline.py 模块
docstring),本脚本不需要、也不应该在这之上再叠加一层自己的"已经跑过就跳过"
逻辑——直接调用底层接口,缓存命中天然是空网络调用,重复跑是安全且廉价的。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from LOCKED.data_pipeline import DataPipeline  # noqa: E402

UNIVERSE_PATH = PROJECT_ROOT / "universe_active.json"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

MS_PER_DAY = 86_400_000


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_default_symbols() -> list[str]:
    """默认标的来源优先级:universe_active.json(生产环境的合格名单,
    universe_filter.py 每周刷新一次)> 空(调用方必须显式传 --symbols)。

    不在这里编一个"config.yaml 推出来的默认名单"——config.yaml 的
    universe_rule 只是筛选*规则*(成交额/上市天数门槛),不包含具体
    symbol 名单,凭空编一份默认标的清单没有依据,也可能跟真实 universe
    不一致,悄悄回填错的币种历史反而造成误导。"""
    if UNIVERSE_PATH.exists():
        import json

        with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return list(payload.get("symbols", []))
    return []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config = load_config()
    data_cfg = config.get("data", {}) or {}

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="要回填的symbol列表(ccxt统一格式,如 BTC/USDT:USDT)。不传则读取 universe_active.json。",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=int(data_cfg.get("history_days", 730)),
        help="回填多少天历史(默认取 config.yaml data.history_days)。",
    )
    parser.add_argument(
        "--timeframe",
        default=str(data_cfg.get("timeframe", "4h")),
        help="K线时间框架(默认取 config.yaml data.timeframe)。",
    )
    parser.add_argument(
        "--exchange",
        default=str(data_cfg.get("exchange", "okx")),
        help="ccxt交易所id(默认取 config.yaml data.exchange)。",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="parquet缓存目录(默认 DataPipeline 的内置默认值:项目根目录下 data_cache/)。",
    )
    return parser.parse_args(argv)


def backfill(dp: DataPipeline, symbols: list[str], history_days: int, timeframe: str) -> None:
    since_ms = dp.clock.now_ms() - history_days * MS_PER_DAY

    for symbol in symbols:
        print(f"=== {symbol} ===", flush=True)
        try:
            ohlcv = dp.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=100_000)
            span_days = (
                (int(ohlcv["timestamp"].max()) - int(ohlcv["timestamp"].min())) / MS_PER_DAY
                if not ohlcv.empty
                else 0.0
            )
            print(f"  ohlcv({timeframe}): {len(ohlcv)} 根K线, 跨度约 {span_days:.1f} 天", flush=True)
        except Exception as exc:  # noqa: BLE001 - 单个symbol失败不应该中止整个回填批次
            print(f"  ohlcv({timeframe}) 拉取失败,跳过: {exc!r}", flush=True)

        try:
            funding = dp.fetch_funding_rate_history(symbol, since=since_ms, limit=100_000)
            span_days = (
                (int(funding["timestamp"].max()) - int(funding["timestamp"].min())) / MS_PER_DAY
                if not funding.empty
                else 0.0
            )
            print(f"  funding_rate_history: {len(funding)} 条, 跨度约 {span_days:.1f} 天", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  funding_rate_history 拉取失败,跳过: {exc!r}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config()

    symbols = args.symbols if args.symbols else load_default_symbols()
    if not symbols:
        raise SystemExit(
            "没有可回填的symbol:universe_active.json 不存在/为空,且未显式传 --symbols。"
            "请先跑一次 universe_filter.refresh(),或用 --symbols 显式指定。"
        )

    print(
        f"回填计划: {len(symbols)} 个symbol, history_days={args.history_days}, "
        f"timeframe={args.timeframe}, exchange={args.exchange}",
        flush=True,
    )
    print(f"标的: {symbols}", flush=True)

    dp_kwargs = dict(exchange_id=args.exchange)
    if args.cache_dir:
        dp_kwargs["cache_dir"] = Path(args.cache_dir)
    dp = DataPipeline(**dp_kwargs)

    backfill(dp, symbols, history_days=args.history_days, timeframe=args.timeframe)
    print("回填完成(缓存命中的区间已跳过网络请求,可安全重复运行)。", flush=True)


if __name__ == "__main__":
    main()
