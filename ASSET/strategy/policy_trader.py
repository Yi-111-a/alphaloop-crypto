"""
ASSET/strategy/policy_trader.py -- M8 前向决策分发层(改造规格书M8)。

背景(规格书§0.1诊断2,复述其结论):M7(ASSET/strategy/policies/)引入了
确定性代码策略,但在M8之前,从来没有任何前向调度路径真的调用过它们。
main.py.AlphaLoopScheduler 只认识 trader.decide() 这一个签名,
scripts/ignite.py 里不管是main分支还是任何evo候选分支,调用的永远是同一个
ASSET/strategy/trader.py::Trader(LLM路径)。锦标赛"晋升"一个分支时,晋升的
只是一段战术文字(scripts/ignite.py::save_main_tactics),被塞进下一次LLM
prompt里当"提示",从来没有一条真正跳过LLM、纯代码执行的前向决策路径——
GitMergeExecutor/EvolutionOrchestrator这条生产链路因此从未在"代码化策略
分支"上被真实调用过。

DispatchingTrader 补上这条路径:duck-type 完全兼容
ASSET.strategy.trader.Trader.decide() 的签名,main.py 的
AlphaLoopScheduler._call_trader_with_timeout() 不需要知道、也不需要关心
self.trader 到底是原始的 LLM Trader,还是这层分发器——按 main.py 既定的
依赖注入原则("trader: Any,duck-typed: .decide(...)"),DispatchingTrader
本身就是这个注入点合法的一种实现,main.py 不需要,也没有,任何改动。

分发规则:
  1. policy_resolver(branch) 返回非None的policy_id -> 纯代码路径:
     load_policy(policy_id) + 拼装 StrategyContext + policy.decide(ctx)。
     全程不触碰 llm_trader,不消耗任何LLM调用预算(见
     tests/test_m8_promotion_gate.py 的"零LLM断言")。
  2. policy_resolver(branch) 返回 None -> 原样委托 llm_trader.decide(),
     全部参数透传,不改变任何既有LLM路径行为(凉兮等纯提示词分支不受影响)。

positions 参数的形状:main.py.run_decision_cycle 传入的是
sim.get_portfolio()["positions"](list[PerpPosition],不是dict——见
LOCKED/simulator.py::get_portfolio() 的真实实现:`list(self.positions.values())`)。
StrategyContext.positions 要求 dict[str, PerpPosition],这里按 symbol 做一次
keying 转换;调用方如果已经传入 dict(比如未来某条测试/调用路径),这里
直接透传,不强制要求是 main.py 的具体传参形状。

空决策语义(设计决策,规格书未强制二选一时的取舍):policy.decide() 在
"无信号/数据不足"时按 M7 约定返回空列表(见
ASSET/strategy/policies/__init__.py 模块docstring)。这里刻意不把空列表
转换成一条合成的hold Decision——读 main.py::run_decision_cycle 确认过:
decisions 为空列表时,撮合循环(`for decision in decisions`)天然是0次
迭代,advice_path 保持 None,函数正常返回 status="decided",不抛异常、
不要求非空。main.py 并不需要"至少一条决策"这个前提,因此 M7 既定的
"空列表=本周期无操作"语义在这里被完整保留,不需要在分发层强行合成一条
其实没有真实信息量的 hold 决策(合成的 hold 决策还需要编一句真实但空洞的
thesis/falsifier 才能通过 schema 校验,纯粹是为了凑数,没有必要)。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

from LOCKED.schemas import Decision, PerpPosition

from ASSET.strategy.policies import StrategyContext, load_policy

# 除 REQUIRED_HISTORY_BARS 之外额外多拉的K线缓冲根数。5个种子策略里最大的
# 单项指标窗口是14(ATR_WINDOW),这里给一点富余,降低"历史刚好够
# REQUIRED_HISTORY_BARS、但因为交易所K线收盘时间/缓存边界差1根导致本该有
# 信号却被跳过"的概率——不是正确性必需的(即使不加这个缓冲,policy 自己
# `len(df) < REQUIRED_HISTORY_BARS` 的防御性检查也不会崩,只会保守地跳过
# 本周期),纯粹是工程余量。
_HISTORY_BUFFER_BARS = 5


class DispatchingTrader:
    """决定"这个分支这个周期该不该经过LLM"的分发层(§M8)。

    构造参数:
      llm_trader      ASSET.strategy.trader.Trader 实例(或任何duck-type
                       兼容的对象)——policy_resolver 对某个分支返回 None
                       时,原样委托给它,不做任何改写。
      policy_resolver  Callable[[str], Optional[str]]:branch -> policy_id
                       (或 None)。调用方(scripts/ignite.py)负责这个
                       映射从哪里来(名册文件/main_policy.json 等),本类
                       只按它的返回值二选一分发,不关心它的实现细节。
      data_pipeline    LOCKED.data_pipeline.DataPipeline(或任何暴露
                       fetch_ohlcv(symbol, timeframe, limit=...) 的对象)。
                       缓存命中时零网络请求(见 DataPipeline.fetch_ohlcv
                       模块docstring)。
      memory_store     可选。存在时按分支做只读检索,喂给
                       StrategyContext.memory_context;不存在时该字段
                       固定为空列表。
      timeframe        拉取K线用的周期字符串,默认 "4h"(与
                       config.yaml.data.timeframe 的默认值一致;调用方
                       如果用了不同周期,自行传入覆盖)。
    """

    def __init__(
        self,
        llm_trader: Any,
        policy_resolver: Callable[[str], Optional[str]],
        data_pipeline: Any,
        memory_store: Optional[Any] = None,
        timeframe: str = "4h",
    ) -> None:
        self.llm_trader = llm_trader
        self.policy_resolver = policy_resolver
        self.data_pipeline = data_pipeline
        self.memory_store = memory_store
        self.timeframe = timeframe

    # ------------------------------------------------------------------
    # positions: list[PerpPosition] -> dict[symbol, PerpPosition]
    # ------------------------------------------------------------------

    @staticmethod
    def _positions_to_dict(positions: Any) -> dict[str, PerpPosition]:
        if isinstance(positions, dict):
            return dict(positions)
        return {p.symbol: p for p in positions}

    # ------------------------------------------------------------------
    # memory_store 检索结果的防御性解包,与 ASSET/strategy/trader.py::
    # Trader._extract_content 同一套逻辑(MemoryStore.retrieve 返回
    # (record, score) 元组)。独立保留一份而不是 import Trader 的内部方法,
    # 避免两个模块之间产生非公开实现细节层面的耦合。
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(item: Any) -> str:
        content = getattr(item, "content", None)
        if content is not None:
            return content
        if isinstance(item, dict) and "content" in item:
            return item["content"]
        if isinstance(item, (tuple, list)) and len(item) > 0:
            first = item[0]
            content = getattr(first, "content", None)
            if content is not None:
                return content
            if isinstance(first, dict) and "content" in first:
                return first["content"]
        return str(item)

    def _retrieve_memory_context(self, query_text: str, ts: int, top_k: int, branch: str) -> list[str]:
        if self.memory_store is None:
            return []
        try:
            raw = self.memory_store.retrieve(query_text, query_ts=ts, top_k=top_k, branch=branch)
        except TypeError:
            # 旧接口/测试Fake可能没有 branch 参数,duck-type降级兼容(与
            # ASSET/strategy/trader.py::Trader.build_context 同一处理方式)。
            raw = self.memory_store.retrieve(query_text, query_ts=ts, top_k=top_k)
        if not raw:
            return []
        return [self._extract_content(item) for item in raw]

    # ------------------------------------------------------------------
    # 唯一入口:与 Trader.decide() 逐参数同签名。
    # ------------------------------------------------------------------

    def decide(
        self,
        ts: int,
        positions: Any,
        latest_snapshot: dict,
        last_reflection_summary: Optional[str] = None,
        program_tactics: Optional[str] = None,
        memory_query_text: str = "",
        top_k: int = 5,
        branch: str = "main",
    ) -> list[Decision]:
        policy_id = self.policy_resolver(branch)
        if policy_id is None:
            # 回退路径:原样委托LLM Trader,全部参数透传,不改变任何既有
            # 行为(program_tactics/last_reflection_summary 等纯提示词
            # 机制继续对这类分支生效)。
            return self.llm_trader.decide(
                ts=ts,
                positions=positions,
                latest_snapshot=latest_snapshot,
                last_reflection_summary=last_reflection_summary,
                program_tactics=program_tactics,
                memory_query_text=memory_query_text,
                top_k=top_k,
                branch=branch,
            )

        # 纯代码路径:全程零LLM调用。
        policy = load_policy(policy_id)

        limit = int(policy.REQUIRED_HISTORY_BARS) + _HISTORY_BUFFER_BARS
        recent_bars = {
            symbol: self.data_pipeline.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            for symbol in latest_snapshot
        }

        ctx = StrategyContext(
            ts=ts,
            positions=self._positions_to_dict(positions),
            snapshot=latest_snapshot,
            recent_bars=recent_bars,
            memory_context=self._retrieve_memory_context(memory_query_text, ts, top_k, branch),
        )

        decisions = policy.decide(ctx)

        # policy代码本身不知道、也不应该知道自己此刻正跑在哪个分支下(见
        # StrategyContext docstring:只暴露纯只读快照,不暴露"我是谁"这类
        # 身份信息),branch归属由分发层在这里统一盖章,而不是要求每个
        # policy自己在Decision里正确填branch。
        return [d if d.branch == branch else replace(d, branch=branch) for d in decisions]


__all__ = ["DispatchingTrader"]
