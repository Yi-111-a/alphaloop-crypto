"""
ASSET/strategy/policies/flat_v1.py —— 量化研究员德比(第五代架构)的种子策略。

第五代架构决定(废除大模型直接交易,大模型转岗"量化研究员",见
scripts/ignite.py 模块docstring里 quant_derby 模式的完整说明):每个大脑
研究员从这份最朴素的空仓策略起步——decide() 恒返回空列表,永远不建仓、
不平仓、不调整任何持仓,是一张真正意义上的白纸。

这不是一个"能用"的战术,是一条基线和一记警钟:
  - 基线:任何研究员提交的新策略代码,只要跑赢"永远空仓、零收益、零回撤"
    这条基线,就已经证明它至少做对了一件事——空仓本身在锦标赛净值对比里
    永远是edge=0,不可能触发晋升,也不可能触发回撤淘汰,是整个德比里
    "最安全但注定被淘汰"的那个策略。
  - 警钟:量化德比的末位斩杀规则(见 scripts/ignite.py 里
    _TOURNAMENT_STAKES_SUFFIX / evaluate_cull 的docstring)不看对错,只看
    排名——长期挂着这份种子策略空仓,窗口内收益率恒为0,一旦其他分支中
    有任何一个哪怕小幅盈利,这个分支的排名必然垫底,大概率就是下一个被
    末位斩杀出局的对象。这份策略存在的意义就是逼着背后的大脑研究员尽快
    提交一份真正能交易、且已经通过 ASSET/strategy 内环回测关卡
    (config.yaml backtest 段)验证过的策略代码来接管这个分支,而不是
    允许"研究员迟迟不交作业、账户永远空仓"这种状态无限期存在下去。

REQUIRED_HISTORY_BARS=1 只是为了满足 load_policy() 的契约校验(必须是
int),decide() 本身完全不读 ctx.recent_bars,不对历史长度有任何真实要求。
"""
from __future__ import annotations

from ASSET.strategy.policies import StrategyContext
from LOCKED.schemas import Decision

REQUIRED_HISTORY_BARS = 1

DESCRIPTION = (
    "量化德比种子策略:恒空仓。每个大脑研究员从这里起步,必须尽快提交"
    "能通过回测关卡的真策略,否则空仓零收益撑不过末位斩杀。"
)


def decide(ctx: StrategyContext) -> list[Decision]:
    """恒返回空列表——不建仓、不平仓、不调整,本周期没有任何操作。"""
    return []
