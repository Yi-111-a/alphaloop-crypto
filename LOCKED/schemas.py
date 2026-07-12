"""
共享数据结构(LOCKED区与调用方共用的契约)。

铁律:本文件定义的是接口契约本身,不包含业务逻辑。
LOCKED 区各模块(simulator/scorer/circuit_breaker/universe_filter/baseline_agents)
以及 ASSET 区都通过这里定义的结构互相通信,禁止绕过这些结构直接传 dict。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

ActionType = Literal["open_long", "open_short", "close", "adjust", "hold"]
SideType = Literal["long", "short"]
CircuitState = Literal["NORMAL", "FROZEN_FULL", "FROZEN_24H"]
VerdictDecision = Literal["PROMOTE", "ARCHIVE", "FAIL"]
ThesisStatus = Literal["应验", "证伪", "未决"]
MemoryLayer = Literal["L1", "L2", "L3"]
ColdStartState = Literal["COLD_START", "NORMAL"]

# §3.4 分层记忆时间常数(τ,单位:天)。术语勘误(spec v1.3):公式是
# exp(-Δt/τ),这是"e-折时间"(时间常数),不是严格物理意义上的"半衰期"——
# 后者要求 Δt=τ 时衰减到 0.5,而 exp(-1)≈0.368。这里的具体取值(3天/30天)
# 本来就是拍的经验参数,0.368 与 0.5 的差异落在参数任意性之内,不改公式,
# 只改名字消除误导。L3(永久层)τ 为 None,检索时不做时间衰减。
MEMORY_TIME_CONSTANT_DAYS: dict[MemoryLayer, Optional[float]] = {
    "L1": 3.0,
    "L2": 30.0,
    "L3": None,
}


@dataclass
class Decision:
    """决策的标准结构,ASSET区必须按此格式产出(§2.2)。"""

    ts: int  # 决策产生时间(UTC ms),必须早于目标K线open时间
    symbol: str
    action: ActionType
    target_notional_pct: float  # 目标名义仓位占净值百分比(如200 = 2倍净值的名义敞口)
    leverage: int  # 1-10
    thesis: str  # 必填,非空,>=20字符
    falsifier: str  # 必填,非空,>=20字符(自然语言,给人看的)
    horizon: str  # 预期持有周期,如 "12h" / "3d"
    branch: str = "main"  # 决策所属策略分支(main 或 evo/YYYYMMDD-简述)
    falsifier_condition: Optional[str] = None  # 机器可读版本的falsifier,见 parse_falsifier_condition


# ---------------------------------------------------------------------------
# falsifier_condition —— M3 裁决:falsifier 的判定不能全靠 LLM 自评(已知的自我
# 评估偏差:LLM 倾向于把自己的失败判成"未决")。Trader 必须在自然语言 falsifier
# 之外,额外产出这个机器可读子句,格式:"price<48000" / "price>=52000.5" —— 固定
# 关键字 "price"(指 decision.symbol 在 decision.horizon 窗口内的价格) + 比较符
# + 数字。Reflector 用下面这对纯函数确定性判定,不依赖 LLM 自评。
# ---------------------------------------------------------------------------

FalsifierOp = Literal["<", "<=", ">", ">="]

_FALSIFIER_CONDITION_RE = re.compile(r"^price\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)$")


@dataclass
class FalsifierCondition:
    op: FalsifierOp
    price: float


def parse_falsifier_condition(raw: Optional[str]) -> Optional[FalsifierCondition]:
    """将 "price<48000" 这样的机器可读子句解析成 FalsifierCondition。
    解析失败(None、非字符串、不匹配格式)一律返回 None,不抛异常——调用方
    (Reflector)据此判定"这条决策的证伪条件不可机读"，走"未决"分支，而不是
    崩溃或误判。"""
    if not raw or not isinstance(raw, str):
        return None
    m = _FALSIFIER_CONDITION_RE.match(raw.strip())
    if not m:
        return None
    op, num = m.group(1), m.group(2)
    return FalsifierCondition(op=op, price=float(num))


def evaluate_falsifier_condition(condition: FalsifierCondition, price: float) -> bool:
    """在给定观测价格下,该证伪条件是否被触发(True = 证伪已发生)。"""
    if condition.op == "<":
        return price < condition.price
    if condition.op == "<=":
        return price <= condition.price
    if condition.op == ">":
        return price > condition.price
    if condition.op == ">=":
        return price >= condition.price
    raise ValueError(f"unknown falsifier condition op: {condition.op!r}")


@dataclass
class PerpPosition:
    symbol: str
    side: SideType
    notional: float  # 名义价值(USDT)
    entry_price: float
    margin: float  # 占用保证金 = notional / leverage
    leverage: int


@dataclass
class Trade:
    ts: int
    symbol: str
    side: SideType
    action: ActionType
    notional: float
    price: float  # 成交价
    fee: float
    slippage_bps: float
    leverage: int
    branch: str = "main"


@dataclass
class Rejection:
    ts: int
    symbol: str
    reason: str
    decision: Decision


@dataclass
class LiquidationEvent:
    ts: int
    symbol: str
    branch: str
    side: SideType
    notional: float
    entry_price: float
    liquidation_price: float
    margin_lost: float


@dataclass
class FundingSettlement:
    ts: int
    symbol: str
    branch: str
    side: SideType
    notional: float
    funding_rate: float
    amount: float  # 正数=账户被扣款,负数=账户收款


@dataclass
class Verdict:
    """scorer.ratchet_score 的返回结构(§2.3)。"""

    branch: str
    decision: VerdictDecision
    score: float  # 窗口内扣费收益率 - 同期benchmark收益率
    max_drawdown_pct: float
    reason: str = ""
    edge_vs_main_pct: float = 0.0  # M4:候选分支收益率 - 主线收益率(同一对齐窗口),
                                    # PROMOTE 门槛直接判定的就是这个数(见 min_promote_edge_pct)


@dataclass
class PromotionRecord:
    """M4:一次 PROMOTE 事件的记录,供 scorer.monthly_report 的"晋升前后表现对比"
    栏使用——晋升本身不是本文件的职责(那是 LOCKED/evolution_orchestrator.py 的
    职责),这里只是记录事件本身的最小结构。"""

    branch: str
    created_date: str  # ISO日期,该分支开始被计入棘轮评分的起点(=分支创建时刻)
    promoted_date: str  # ISO日期,PROMOTE 判定发生的日期


@dataclass
class MemoryRecord:
    """分层记忆的单条记录(§3.4)。ts 是记忆写入时间(UTC ms),检索时的时间衰减
    以调用方显式传入的 query_ts 为基准计算 Δt = query_ts - ts,绝不使用墙钟
    时间——任何检索函数都不得隐式调用 time.time()/datetime.now(),否则同一次
    历史回放在不同时间跑会产生不同结果,也为"未来记忆污染当下决策"这类信息
    泄漏留了后门。"""

    id: str
    ts: int
    layer: MemoryLayer
    content: str
    importance: float = 1.0  # 重要性权重,检索得分的乘子之一
    embedding: Optional[list[float]] = None


@dataclass
class ThesisMark:
    """反思模块对单条历史决策的标注(§3.3)。"""

    decision_ts: int
    symbol: str
    thesis_status: ThesisStatus
    note: str = ""
