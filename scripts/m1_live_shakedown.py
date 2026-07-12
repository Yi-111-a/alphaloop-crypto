"""
scripts/m1_live_shakedown.py —— M5 阶段二:挂账的 M1 联网验收。

跑法:
    cd alphaloop
    python scripts/m1_live_shakedown.py

用真实 2024 年某个完整月份的 4h K线 + 真实历史资金费率(通过
LOCKED.data_pipeline.DataPipeline 的公开归档回退路径,从
data.binance.vision 下载,详见该模块 docstring 里记录的联网适配说明),
驱动 RandomAgent 跑完整月,再抽 3 笔真实成交,逐项手工核对:
  - 成交价 = 该K线真实 open × (1 ± slippage_bps/1e4)
  - 手续费 = 名义额 × taker_pct
  - 跨结算点持仓的资金费用 = 名义价值 × 币安真实历史费率,方向正确
    (正费率多仓扣钱)

本脚本不修改 LOCKED 区任何业务逻辑,只是把已经独立测试过的 LOCKED 模块
(DataPipeline/Simulator/RandomAgent)接起来跑一遍真实数据,输出核对表。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from LOCKED import log_writer  # noqa: E402
from LOCKED.baseline_agents import RandomAgent  # noqa: E402
from LOCKED.data_pipeline import DataPipeline  # noqa: E402
from LOCKED.schemas import Trade  # noqa: E402
from LOCKED.simulator import Simulator  # noqa: E402

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "4h"
YEAR, MONTH = 2024, 6  # 任一完整月份,这里选 2024-06

MS_PER_DAY = 86_400_000


def month_range_ms(year: int, month: int) -> tuple[int, int]:
    import pandas as pd
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.DateOffset(months=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def main() -> None:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    since_ms, until_ms = month_range_ms(YEAR, MONTH)

    run_root = PROJECT_ROOT / "scratch" / "m1_shakedown"
    run_root.mkdir(parents=True, exist_ok=True)
    log_root = run_root / "LOG"
    log_root.mkdir(parents=True, exist_ok=True)
    cache_dir = run_root / "data_cache"

    print(f"=== M1 联网 shakedown: {SYMBOL} {YEAR}-{MONTH:02d} (真实历史数据) ===")

    dp = DataPipeline(cache_dir=cache_dir, enable_public_archive_fallback=True)

    print("拉取真实 4h K线...")
    ohlcv = dp.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since_ms, limit=100000)
    ohlcv = ohlcv[ohlcv["timestamp"] < until_ms].reset_index(drop=True)
    print(f"  取得 {len(ohlcv)} 根K线, {ohlcv.iloc[0]['timestamp']} ~ {ohlcv.iloc[-1]['timestamp']}")
    if len(ohlcv) < 100:
        raise SystemExit(f"K线数量异常偏少({len(ohlcv)}),疑似归档拉取不完整,中止")

    print("拉取真实历史资金费率...")
    funding = dp.fetch_funding_rate_history(SYMBOL, since=since_ms, limit=100000)
    funding = funding[funding["timestamp"] < until_ms].reset_index(drop=True)
    print(f"  取得 {len(funding)} 条资金费率记录")
    funding_by_ts = dict(zip(funding["timestamp"], funding["funding_rate"]))

    def funding_rate_lookup(symbol: str, ts: int) -> float:
        if ts in funding_by_ts:
            return funding_by_ts[ts]
        # 归档里没有恰好对齐的时间戳时,取最近的一条(真实数据边界效应)
        nearest = min(funding_by_ts.keys(), key=lambda t: abs(t - ts))
        return funding_by_ts[nearest]

    sim = Simulator(
        config=config,
        universe_symbols=[SYMBOL],
        db_path=run_root / "portfolio_shakedown.db",
        log_root=log_root,
        branch="shakedown",
    )
    agent = RandomAgent(universe_symbols=[SYMBOL], seed=42, branch="shakedown")

    taker_pct = config["fees"]["taker_pct"]
    slippage_bps = config["fees"]["slippage_bps"]

    settle_hours = set(config["funding"]["settle_hours_utc"])
    trades: list[Trade] = []
    rejections = 0

    for i in range(1, len(ohlcv)):
        prev_bar = ohlcv.iloc[i - 1]
        cur_bar = ohlcv.iloc[i]
        decision_ts = int(prev_bar["timestamp"]) + 1  # 上一根收盘后一毫秒做决策
        next_bar = {
            "open_time": int(cur_bar["timestamp"]),
            "open": float(cur_bar["open"]),
            "volume_24h_usdt": 1e9,  # 真实值本脚本未拉取,给一个明显高于门槛的值避免滑点加倍分支影响核对
        }

        decision = agent.decide(ts=decision_ts)
        sim.log_decision(decision)
        result = sim.execute(decision, next_bar)
        if isinstance(result, Trade):
            trades.append(result)
        else:
            rejections += 1

        # 资金费率结算:K线的 open_time 恰好落在 UTC 0/8/16 点时结算一次
        bar_dt_hour = (int(cur_bar["timestamp"]) // 3_600_000) % 24
        if bar_dt_hour in settle_hours:
            sim.settle_funding(int(cur_bar["timestamp"]), funding_rate_lookup)

        # 每根K线都做一次插针检查(用真实 high/low)
        sim.check_liquidation(
            {SYMBOL: {"high": float(cur_bar["high"]), "low": float(cur_bar["low"]), "close": float(cur_bar["close"])}},
            ts_utc=int(cur_bar["timestamp"]),
        )
        if sim.branch_dead:
            print(f"  分支在第{i}根K线爆仓,提前结束回放(这是真实数据下的合理结果,不是bug)")
            break

    print(f"\n回放完成: {len(trades)} 笔成交, {rejections} 笔拒绝")
    portfolio = sim.get_portfolio({SYMBOL: float(ohlcv.iloc[min(i, len(ohlcv) - 1)]["close"])})
    print(f"最终账户: wallet_balance={portfolio['wallet_balance']:.2f} nav={portfolio['nav']:.2f} "
          f"branch_dead={portfolio['branch_dead']}")

    # ---- 抽 3 笔成交,手工核对表 ----
    # 只挑"真实成交"(action != hold)做核对——hold本身也会产生一条 Trade
    # 记录(notional=0/fee=0/price=bar的open价,因为没有真的买卖任何东西),
    # RandomAgent 80%概率hold,直接取trades[:3]大概率抽到没有代表性的hold
    # "成交",核对表里滑点/手续费自然对不上,不是业务逻辑的问题。
    real_fills = [t for t in trades if t.action != "hold"]
    sample = real_fills[:3] if len(real_fills) >= 3 else real_fills
    print(f"\n=== 抽样核对表({len(sample)}笔) ===\n")
    # open_time -> 真实K线open价,Trade.ts 记录的正是撮合时刻(next_bar.open_time)
    open_by_ts = {int(ts): float(o) for ts, o in zip(ohlcv["timestamp"], ohlcv["open"])}

    header = f"{'ts':<15}{'方向':<8}{'真实open':<14}{'成交价':<14}{'预期成交价':<14}{'名义额':<14}{'手续费':<12}{'预期手续费':<12}{'price_match':<12}{'fee_match'}"
    print(header)
    print("-" * len(header))
    for t in sample:
        real_open = open_by_ts.get(int(t.ts), None)
        sign = 1 if t.side == "long" else -1
        expected_price = real_open * (1 + sign * slippage_bps / 1e4) if real_open is not None else None
        expected_fee = t.notional * taker_pct
        price_match = abs(t.price - expected_price) < 1e-6 if expected_price is not None else "N/A"
        fee_match = abs(t.fee - expected_fee) < 1e-6
        print(
            f"{t.ts:<15}{t.side:<8}{(real_open or 0):<14.2f}{t.price:<14.2f}"
            f"{(expected_price or 0):<14.2f}{t.notional:<14.2f}{t.fee:<12.4f}{expected_fee:<12.4f}"
            f"{str(price_match):<12}{fee_match}"
        )

    # ---- 资金费率核对(找一笔跨结算点的持仓) ----
    funding_records = log_writer.read_jsonl("funding.jsonl", root=log_root)
    print(f"\n=== 资金费率结算核对({len(funding_records)}笔实际结算)===\n")
    for r in funding_records[:3]:
        real_rate = funding_by_ts.get(r["ts"])
        # Simulator.settle_funding 自己文档写死的符号约定: rate>0 时多仓付款
        # (amount=+notional*rate)、空仓收款(amount=-notional*rate);rate<0 时
        # 反之。核对表必须用同一个公式独立算一遍,而不是直接抄它的输出。
        if real_rate is not None:
            expected_amount = r["notional"] * real_rate if r["side"] == "long" else -r["notional"] * real_rate
        else:
            expected_amount = None
        direction_ok = None
        if real_rate is not None:
            if r["side"] == "long":
                direction_ok = (real_rate > 0 and r["amount"] > 0) or (real_rate < 0 and r["amount"] < 0) or real_rate == 0
            else:
                direction_ok = (real_rate > 0 and r["amount"] < 0) or (real_rate < 0 and r["amount"] > 0) or real_rate == 0
        amount_match = abs(r["amount"] - expected_amount) < 1e-6 if expected_amount is not None else "N/A"
        print(
            f"ts={r['ts']} side={r['side']} notional={r['notional']:.2f} "
            f"real_rate={real_rate} amount={r['amount']:.4f} expected={expected_amount} "
            f"amount_match={amount_match} direction_ok={direction_ok}"
        )

    print(f"\n数据保存在: {run_root}")


if __name__ == "__main__":
    main()
