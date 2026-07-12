"""
position_risk_monitor.py —— 每小时紧急风控检查(用户明确要求,不在原
spec v1.2 里,是 M5 阶段三点火过程中补充的一条护栏)。

铁律(用户原话确认过的设计裁决):这个检查**必须是确定性代码判断,不走
LLM**——"紧急操作"意味着不能等 Trader/agent 签入推理(agent 可能几十分钟
才签入一次,见 scripts/llm_bridge.py),必须像本项目一贯的纪律一样
(M1 的爆仓判定、M2 的时间边界都是代码硬判定,不是"问LLM觉得要不要紧急平
仓")。本文件因此是纯函数,不接受任何 llm_client、不 import 任何 LLM 相关
的东西。

职责边界:本模块只做"给定最近的分钟级价格 + 当前持仓,判断是否触发紧急
平仓阈值"这一件事,是一个纯函数,不触碰 Simulator/账本——触发后真正提交
close 决策、调用 Simulator.execute() 是调度层(main.py 的
AlphaLoopScheduler.run_risk_check_cycle)的职责,不在这里做,保持"判定"
与"执行"分离(与 circuit_breaker.py 的 check()/调用方决定怎么处理是同一种
分工)。

判定口径("从最近高点回撤",用户原话):
  - 多头:在传入的 recent_prices 窗口内找到滚动最高价 peak,
    drawdown_pct = (peak - current) / peak * 100。
  - 空头:方向相反,找滚动最低价 trough,
    drawdown_pct = (current - trough) / trough * 100。
  - "最近"窗口的长度由调用方决定(传入 recent_prices 本身有多长就是多长)——
    本模块不知道、也不关心这是分钟线还是别的什么颗粒度的数据,只要求
    recent_prices 按时间升序排列。这与"仓位从开仓以来的最大回撤"是两个
    不同的概念,本模块刻意只做前者(窗口内局部回撤),因为它是为"分钟级
    行情突然砸出一个尖刺"这类场景设计的哨兵,不是替代 circuit_breaker.py
    的组合级总回撤熔断。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from LOCKED.schemas import PerpPosition


@dataclass
class RiskCheckResult:
    symbol: str
    triggered: bool
    drawdown_pct: float
    reason: str


def check_position_drawdown(
    position: PerpPosition,
    recent_prices: list[tuple[int, float]],
    threshold_pct: float,
) -> RiskCheckResult:
    """纯函数,无副作用。recent_prices 为空或只有一个点时视为数据不足,
    不触发(不能在没有真实窗口数据的情况下"确定性地"判断出一个回撤——
    宁可这次不触发,也不要用不完整的数据编出一个数字)。"""
    if len(recent_prices) < 2:
        return RiskCheckResult(
            symbol=position.symbol, triggered=False, drawdown_pct=0.0,
            reason="insufficient_recent_price_data",
        )

    current_price = recent_prices[-1][1]

    if position.side == "long":
        peak = max(p for _, p in recent_prices)
        if peak <= 0:
            return RiskCheckResult(
                symbol=position.symbol, triggered=False, drawdown_pct=0.0, reason="invalid_peak_price"
            )
        drawdown_pct = (peak - current_price) / peak * 100.0
    else:  # short
        trough = min(p for _, p in recent_prices)
        if trough <= 0:
            return RiskCheckResult(
                symbol=position.symbol, triggered=False, drawdown_pct=0.0, reason="invalid_trough_price"
            )
        drawdown_pct = (current_price - trough) / trough * 100.0

    triggered = drawdown_pct >= threshold_pct
    reason = (
        f"drawdown {drawdown_pct:.2f}% >= threshold {threshold_pct:.2f}%"
        if triggered
        else f"drawdown {drawdown_pct:.2f}% < threshold {threshold_pct:.2f}%"
    )
    return RiskCheckResult(symbol=position.symbol, triggered=triggered, drawdown_pct=drawdown_pct, reason=reason)


def check_all_positions(
    positions: dict[str, PerpPosition],
    recent_prices_by_symbol: dict[str, list[tuple[int, float]]],
    threshold_pct: float,
) -> list[RiskCheckResult]:
    """对一批持仓逐个跑 check_position_drawdown。某个symbol如果在
    recent_prices_by_symbol 里没有数据,视为数据不足(不触发),不抛异常——
    单个symbol的行情拉取失败不应该让整轮风控检查全部失败。"""
    results = []
    for symbol, position in positions.items():
        recent_prices = recent_prices_by_symbol.get(symbol, [])
        results.append(check_position_drawdown(position, recent_prices, threshold_pct))
    return results
