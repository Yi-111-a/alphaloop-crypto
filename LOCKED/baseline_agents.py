"""
照妖镜(baseline_agents.py) —— 对照组agent,用于证明/证伪主策略agent是否真的跑赢了
"什么都不做"或"随便交易"(§2.5)。

包含两个对照组,但两者的角色和实现方式截然不同(人类裁决,见下):

- RandomAgent 是"选手":存在的意义是回答"同样的规则下,不用脑子能赚多少"。因此它
  必须完整走一遍 Simulator 的九步校验链 —— 杠杆、仓位、总敞口、可用保证金、熔断
  —— 一条都不能豁免,否则就不是在同一套规则下比较。本模块只负责为它产出
  schemas.Decision 对象,不执行撮合;与 simulator 的接线是 main.py 的职责。

- BTCHoldAgent 是"尺子":代表的是"用户根本不用这套系统,直接现货买入BTC并持有"的
  机会成本。这把尺子如果被迫套用永续合约的风控地板(比如 min_free_margin_pct 这种
  为保护杠杆仓位不被强平而设的最低可用保证金比例),就只能实际部署 100% -
  min_free_margin_pct 的资金(默认参数下是 85%),尺子被人为改短了——这会系统性
  压低基准,让主策略agent的"超额收益"显得比真实情况更好看。往对自己(被评估的
  主策略agent)有利的方向弯尺子,是这个项目 LOCKED 区纪律要防止的头号问题,绝对
  不允许。因此 BTCHoldAgent **不通过 Simulator 的永续保证金体系**,而是解析计算:
      nav_benchmark(t) = capital_usdt × (1 - entry_cost_pct) × (price_t / price_entry)
  其中 entry_cost_pct = taker_pct + slippage_bps/1e4,只在期初买入那一刻扣一次
  (对应真实现货买入的一次性成本),此后不再产生任何手续费、资金费率或强平相关的
  扣减——因为真实的现货持有者本来就不暴露在这些风险里。BTCHoldAgent 因此不产出
  Decision 对象,也不经过 execute() / 九步校验链,它的账本就是这条解析曲线本身。
"""
from __future__ import annotations

import random

from LOCKED.schemas import Decision

# 对照组固定使用的 thesis/falsifier 样板文案(均 >=20 字符,满足 simulator 的校验要求)。
# 对照组没有主观判断,因此使用统一的、说明其"对照组"身份的样板文案,而不是编造理由。
_RANDOM_HOLD_THESIS = "随机对照组:无主观判断,仅用于统计基线参照"
_RANDOM_HOLD_FALSIFIER = "随机对照组:不设证伪条件,仅作统计基线参照"
_RANDOM_TRADE_THESIS = "随机对照组:随机抽样标的与方向,用于统计基线,无主观判断依据"
_RANDOM_TRADE_FALSIFIER = "随机对照组:不设证伪条件,仓位到期后按周期随机重新抽样"
_HOLD_HORIZON = "4h"  # 一个决策周期


class RandomAgent:
    """随机对照组。

    每个决策周期:80%概率 hold;20%概率从universe中随机抽取一个symbol,
    随机 open_long 或 open_short,target_notional_pct 从 [5, 25] 均匀抽样。

    种子固定(默认seed=42),使用独立的 random.Random 实例(不污染全局随机状态),
    因此同一seed + 同一调用序列 一定产出完全相同的Decision序列(可复现,§2.5要求)。

    leverage 固定为1(无杠杆):对照组的目的是衡量"随机择时/择币"本身是否有alpha,
    引入杠杆会把爆仓风险和杠杆放大效应混进对比里,污染"随机 vs 主策略"这个单一变量的
    比较,因此这里选择用最简单、风险最低的杠杆设置(spec未明确规定随机组杠杆,此为工程选择)。
    """

    HOLD_PROBABILITY = 0.8
    TRADE_PCT_MIN = 5.0
    TRADE_PCT_MAX = 25.0
    LEVERAGE = 1

    def __init__(self, universe_symbols: list[str], seed: int = 42, branch: str = "random"):
        if not universe_symbols:
            raise ValueError("universe_symbols must not be empty")
        self.universe_symbols = list(universe_symbols)
        self.seed = seed
        self.branch = branch
        self._rng = random.Random(seed)

    def decide(self, ts: int) -> Decision:
        roll = self._rng.random()  # [0.0, 1.0)
        if roll >= self.HOLD_PROBABILITY:
            symbol = self._rng.choice(self.universe_symbols)
            action = self._rng.choice(["open_long", "open_short"])
            pct = self._rng.uniform(self.TRADE_PCT_MIN, self.TRADE_PCT_MAX)
            return Decision(
                ts=ts,
                symbol=symbol,
                action=action,
                target_notional_pct=pct,
                leverage=self.LEVERAGE,
                thesis=_RANDOM_TRADE_THESIS,
                falsifier=_RANDOM_TRADE_FALSIFIER,
                horizon=_HOLD_HORIZON,
                branch=self.branch,
            )

        # 80% 分支:hold,不指向任何具体持仓变化,symbol 用第一个universe标的占位。
        return Decision(
            ts=ts,
            symbol=self.universe_symbols[0],
            action="hold",
            target_notional_pct=0.0,
            leverage=self.LEVERAGE,
            thesis=_RANDOM_HOLD_THESIS,
            falsifier=_RANDOM_HOLD_FALSIFIER,
            horizon=_HOLD_HORIZON,
            branch=self.branch,
        )


class BTCHoldAgent:
    """BTC_HOLD 基准("尺子",config.yaml 中 benchmark: BTC_HOLD 的本体实现)。

    刻意豁免于 Simulator 的永续保证金体系(理由见模块顶部docstring的人类裁决)——
    不产出 Decision,不调用 execute(),没有杠杆/仓位/可用保证金/熔断这些为管理
    杠杆合约风险而存在的校验,因为它代表的是零风险的现货买入持有,这些校验对它
    本就不适用。

    用法:
        agent = BTCHoldAgent.from_config(config)
        agent.enter(entry_price=<期初BTC开盘价>)          # 只调用一次
        nav_t = agent.nav(price=<t时刻BTC价格>)            # 此后任意时刻调用

    nav(t) = capital_usdt × (1 - entry_cost_pct) × (price_t / price_entry)
    entry_cost_pct = taker_pct + slippage_bps/1e4,只在 enter() 时结算一次,
    此后 nav() 是价格的纯函数——不再产生任何手续费、资金费率或强平相关扣减。
    """

    def __init__(
        self,
        capital_usdt: float,
        taker_pct: float,
        slippage_bps: float,
        symbol: str = "BTC/USDT:USDT",
        branch: str = "btc_hold",
    ):
        self.capital_usdt = float(capital_usdt)
        self.entry_cost_pct = float(taker_pct) + float(slippage_bps) / 1e4
        self.symbol = symbol
        self.branch = branch
        self.entry_price: float | None = None
        self.units: float | None = None

    @classmethod
    def from_config(cls, config: dict, symbol: str = "BTC/USDT:USDT", branch: str = "btc_hold") -> "BTCHoldAgent":
        fees = config.get("fees", {})
        return cls(
            capital_usdt=config["capital_usdt"],
            taker_pct=fees.get("taker_pct", 0.0),
            slippage_bps=fees.get("slippage_bps", 0.0),
            symbol=symbol,
            branch=branch,
        )

    @property
    def entered(self) -> bool:
        return self.entry_price is not None

    def enter(self, entry_price: float) -> None:
        """期初买入,只允许调用一次。扣除一次性 taker 费 + 滑点成本后按 entry_price
        换算成持有的 BTC 数量(units),此后 nav() 不再产生任何额外扣减。"""
        if self.entered:
            raise RuntimeError("BTCHoldAgent.enter() must only be called once")
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        self.entry_price = float(entry_price)
        effective_capital = self.capital_usdt * (1.0 - self.entry_cost_pct)
        self.units = effective_capital / self.entry_price

    def nav(self, price: float) -> float:
        """t 时刻的基准净值。enter() 之前返回期初资金(尚未建仓)。"""
        if not self.entered:
            return self.capital_usdt
        return self.units * float(price)
