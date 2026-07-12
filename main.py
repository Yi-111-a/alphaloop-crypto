"""
main.py -- AlphaLoop-Crypto 调度入口(M5,§4 冷启动与调度 / §5 建议输出)。

复审对 M5 的定性(逐字引用,写进代码而非注释里的一句话总结):"main.py 看起来是
胶水代码,其实是第五关,敌人又换了——状态与恢复。前四关的模块都是纯逻辑,
main.py 是唯一要和'进程会死、网络会断、LLM 会超时、时钟会跨结算点'的现实打交道
的模块。" 下面六件事分别是这句话在代码里的落点:

  1. 决策周期幂等性 -- kill -9 重启后不得对同一个4小时周期重复调用 Trader/
     重复落盘决策(见 run_decision_cycle / _decision_already_exists_for_cycle)。
  2. 停机补偿的非对称性 -- 决策绝不补(错过的周期就是错过了,永远只对"现在"
     决策一次),资金费率结算必须补齐每一个错过的结算时点(见
     run_settlement_catchup / compute_missed_settlement_instants)。
  3. 唯一墙钟来源 -- 本文件全文件零 time.time()/datetime.now()/
     datetime.utcnow() 调用,"现在"只有一个来源:注入的 Clock 实例。见本文件
     底部的模块自测(tests/test_main_scheduler.py 里的 AST 扫描测试),风格与
     tests/test_clock.py::test_no_wallclock_calls_anywhere_in_locked_or_asset
     完全一致。main.py 故意放在项目根目录、不放进 LOCKED/ 或 ASSET/,所以不会
     被那个项目级扫描测试自动覆盖到,但本文件对自己的要求与那条纪律逐字相同,
     只是用一份独立的、同风格的测试守住。
  4. git merge 闸门 -- PROMOTE 判定不等于真正的晋升;GitMergeExecutor 拒绝
     (测试红/分支非法)时,是一种比普通 ARCHIVE 更严重的失败模式("评分赢了
     但代码会炸"),必须区别记录,且调度层自己维护的 effective_main_branch
     绝不能在这种情况下静默前进(见 run_ratchet_judgment)。
  5. LLM 故障不挂起主循环 -- Trader 调用外面再包一层墙钟超时(线程+
     future.result(timeout=...),不用信号,信号在子线程/跨平台下不可靠);
     Researcher/Reflector 的任意异常被捕获、记 LOG、跳过本轮,下一个调度节拍
     自然就是重试,不在本次调用里做任何重试循环(见 run_reflection_cycle /
     run_daily_research / run_cold_start_research / _call_trader_with_timeout)。
  6. latest_advice.md -- 唯一面向人类用户的界面(§5)。固定免责声明必须逐字
     出现在最上方;thesis/falsifier 必须原文展示,不做任何摘要改写(见
     generate_latest_advice)。

依赖注入原则:与全项目已建立的纪律一致 -- AlphaLoopScheduler 的每一个外部
依赖(每个分支的 Simulator、Trader、Reflector、Researcher、MemoryStore、
Scorer、EvolutionOrchestrator、CircuitBreaker、ColdStartGate、Clock、
GitMergeExecutor,以及行情/资金费率的取数回调)都通过构造函数注入,不在内部
new 出任何一个真实实现,保证离线用 Fake 完全可测。生产环境唯一的真实墙钟来源
是 LOCKED.clock.SystemClock,由启动脚本注入,本文件自己不 import
LOCKED.clock.SystemClock 也不实例化它(那会在 AST 扫描里显得可疑,即使
SystemClock 本身是"合法调用 time.time() 的唯一类"—— 更干净的做法是让启动脚本
在 main.py 之外完成这一次实例化,main.py 只认 Clock 这个抽象)。
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any, Callable, Optional

from LOCKED import log_writer
from LOCKED.schemas import Decision

_PROJECT_ROOT = Path(__file__).resolve().parent

_MS_PER_HOUR = 3_600_000
_MS_PER_DAY = 86_400_000

# ---------------------------------------------------------------------------
# §5 建议输出:固定免责声明,逐字来自 spec。
#
# 为什么这不是可有可无的样板文字(复审原话的转述,不是装饰):这套系统的输出
# 天然带着一种"分析得头头是道"的自信语气 -- 它是不是真的有效,只有等
# agent/BTC_HOLD/随机对照组三条净值线真实分化了几个月之后才能被知道。在那一
# 天到来之前,这行免责声明是给正在阅读 latest_advice.md 的人的必要安全护栏,
# 不是可以被摘要、被精简、被优化掉的样板文字 -- 因此它是一个模块级常量字符串,
# 而不是拼在别的文案里的一部分,并且有专门的测试断言它逐字出现在每次生成的
# 文件最上方(见 tests/test_main_scheduler.py)。
# ---------------------------------------------------------------------------
DISCLAIMER = "本内容为AI模拟实验输出,非投资建议,不构成任何交易依据"

# 调度器级别的超时兜底决策文案 -- 刻意与 ASSET/strategy/trader.py 自己的
# _HOLD_FALLBACK_THESIS 用不同措辞,方便在日志/测试里明确区分"Trader 内部重试
# 3 次后自己兜底"和"调度层等不到 Trader 返回,直接放弃等待"这两种性质不同的
# 失败模式(见模块 docstring 第5条)。
_SCHEDULER_TIMEOUT_FALLBACK_THESIS = (
    "调度器级超时兜底:Trader.decide()未能在配置的超时时间内返回(底层llm_client"
    "调用疑似挂起),不等待Trader自身的重试收敛,直接按安全默认值执行hold"
)
_SCHEDULER_TIMEOUT_FALLBACK_FALSIFIER = (
    "调度器级超时兜底:本次无有效决策依据,不设证伪条件,等待下一个调度周期"
    "自然重试,不在本次调用内做任何额外重试"
)
_SCHEDULER_TIMEOUT_FALLBACK_HORIZON = "4h"
_SCHEDULER_TIMEOUT_FALLBACK_SYMBOL = "BTC/USDT:USDT"


# ---------------------------------------------------------------------------
# 纯函数:决策周期窗口 / 停机补偿的缺口计算。两者都不接触任何状态,只依赖显式
# 传入的时间戳,方便独立单测(见 tests/test_main_scheduler.py)。
# ---------------------------------------------------------------------------


def decision_cycle_window(now_ms: int, interval_hours: int) -> tuple[int, int]:
    """给定"现在"和决策周期时长,返回该时刻所属周期的 [cycle_start, cycle_end)。

    用整数除法对齐到 UTC 纪元起点的整周期边界(与 circuit_breaker.py 用
    `ts_ms // _DAY_MS` 对齐 UTC 自然日同一手法,不使用 datetime 时区API)。
    """
    interval_ms = int(interval_hours) * _MS_PER_HOUR
    if interval_ms <= 0:
        raise ValueError(f"interval_hours must be positive, got {interval_hours!r}")
    start = (now_ms // interval_ms) * interval_ms
    return start, start + interval_ms


def compute_missed_settlement_instants(
    last_settlement_ts: Optional[int], now_ts: int, settle_hours_utc: list[int]
) -> list[int]:
    """硬要求2 的核心:算出 (last_settlement_ts, now_ts] 区间内所有错过的资金费率
    结算时点(每天 settle_hours_utc 里列出的那几个 UTC 小时)。

    - last_settlement_ts 为 None:代表"从未记录过任何一次结算",不存在"错过"
      这个概念(调用方按约定应在首次调用时先做一次 bootstrap,把 last_settlement_ts
      初始化为当前时刻,而不是回溯到系统诞生那天去补一堆历史结算)。这里直接
      返回空列表,不猜测。
    - now_ts <= last_settlement_ts:没有新的时间流逝,自然无缺口,返回空列表。
    - 否则:逐天扫描 [last_settlement_ts 所在UTC日, now_ts 所在UTC日],对每天的
      每个 settle hour 计算出具体的毫秒时间戳,只保留严格晚于 last_settlement_ts
      且不晚于 now_ts 的那些,按时间正序返回。
    """
    if last_settlement_ts is None or now_ts <= last_settlement_ts:
        return []

    hours = sorted({int(h) for h in settle_hours_utc})
    start_day = last_settlement_ts // _MS_PER_DAY
    end_day = now_ts // _MS_PER_DAY

    instants: list[int] = []
    for day in range(start_day, end_day + 1):
        day_start = day * _MS_PER_DAY
        for h in hours:
            ts = day_start + h * _MS_PER_HOUR
            if last_settlement_ts < ts <= now_ts:
                instants.append(ts)
    instants.sort()
    return instants


# ---------------------------------------------------------------------------
# AlphaLoopScheduler
# ---------------------------------------------------------------------------


class AlphaLoopScheduler:
    """§4 调度入口。每个外部依赖都通过构造函数注入(见模块 docstring),生产
    环境由一个薄的启动脚本(不在本文件职责内)组装真实实现并注入 SystemClock；
    测试永远注入 FakeClock + 各种 Fake 依赖,离线、确定性地跑。
    """

    def __init__(
        self,
        config: dict,
        clock: Any,  # LOCKED.clock.Clock -- 保持 Any 类型注解是本文件"不 import
                     # SystemClock" 这条自我纪律的自然延伸,duck-type 只要求
                     # now_ms() -> int。
        simulators: dict[str, Any],  # branch -> LOCKED.simulator.Simulator,至少含 "main"
        trader: Any,  # ASSET.strategy.trader.Trader,duck-typed: .decide(...)
        reflector: Optional[Any] = None,
        researcher: Optional[Any] = None,
        memory_store: Optional[Any] = None,
        scorer: Optional[Any] = None,
        evolution_orchestrator: Optional[Any] = None,
        circuit_breaker: Optional[Any] = None,
        cold_start_gate: Optional[Any] = None,
        git_merge_executor: Optional[Any] = None,
        next_bar_provider: Optional[Callable[[str, int], Any]] = None,
        snapshot_provider: Optional[Callable[[int], dict]] = None,
        mark_price_provider: Optional[Callable[[int], dict]] = None,
        funding_rate_lookup: Optional[Callable[[str, int], float]] = None,
        price_lookup: Optional[Callable[[str, int], float]] = None,
        benchmark_nav_provider: Optional[Callable[[int], float]] = None,
        random_nav_provider: Optional[Callable[[int], float]] = None,
        decisions_log_path: str = "decisions.jsonl",
        log_root: Optional[str | Path] = None,
        state_path: Optional[str | Path] = None,
        errors_log_path: str = "scheduler_errors.jsonl",
        advice_path: str = "latest_advice.md",
        trader_timeout_seconds: float = 30.0,
        memory_query_text: str = "",
        top_k: int = 5,
    ) -> None:
        self.config = config
        self.clock = clock
        self.simulators = simulators
        self.trader = trader
        self.reflector = reflector
        self.researcher = researcher
        self.memory_store = memory_store
        self.scorer = scorer
        self.evolution_orchestrator = evolution_orchestrator
        self.circuit_breaker = circuit_breaker
        self.cold_start_gate = cold_start_gate
        self.git_merge_executor = git_merge_executor

        self.next_bar_provider = next_bar_provider
        self.snapshot_provider = snapshot_provider
        self.mark_price_provider = mark_price_provider
        self.funding_rate_lookup = funding_rate_lookup
        self.price_lookup = price_lookup
        self.benchmark_nav_provider = benchmark_nav_provider
        self.random_nav_provider = random_nav_provider

        self.decisions_log_path = decisions_log_path
        self.log_root: Optional[Path] = Path(log_root) if log_root is not None else None
        self.errors_log_path = errors_log_path
        self.advice_path = advice_path
        self.trader_timeout_seconds = float(trader_timeout_seconds)
        self.memory_query_text = memory_query_text
        self.top_k = top_k

        self.last_reflection_summary: Optional[str] = None
        self.program_tactics: Optional[str] = None

        self._decision_interval_hours = int(
            (config.get("cycle", {}) or {}).get("decision_interval_hours", 4)
        )
        self._settle_hours_utc = list((config.get("funding", {}) or {}).get("settle_hours_utc", [0, 8, 16]))

        if state_path is not None:
            self.state_path = Path(state_path)
        elif self.log_root is not None:
            self.state_path = self.log_root.parent / "state" / "scheduler_state.json"
        else:
            self.state_path = _PROJECT_ROOT / "state" / "scheduler_state.json"

        self._state: dict = self._load_state()

        # 硬要求4:git merge 被 GitMergeExecutor 拒绝时,调度层自己维护的
        # "真正生效的主分支" 绝不能跟着 EvolutionOrchestrator.current_main_branch
        # 一起静默前进 -- 后者是纯粹的棘轮判定结果,不知道、也不该知道 git 层面
        # 是否真的合并成功。effective_main_branch 只有在 attempt_merge() 返回
        # merged=True 时才会前进,见 run_ratchet_judgment。
        self.effective_main_branch: str = self._state.get("effective_main_branch", "main")

    # ------------------------------------------------------------------
    # 调度器自身的最小可变状态(仅用于崩溃恢复所需的两个游标,不是业务日志)。
    # 与 LOCKED/cold_start.py 的 state_path 用同一手法:一个小 JSON 文件,不走
    # append_only_writer -- 因为它就不是一份历史日志,而是"当前指针"本身，
    # 与本文件"latest_advice.md 允许覆盖写"是同一类需求,同一类解决方式。
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # 通用日志(scheduler_errors.jsonl):真正的错误(超时/异常)和需要被大声
    # 记录的审计事件(promotion_veto/promotion_merged)共用同一个 append-only
    # 文件 -- 都是"调度层观察到的、值得回看的事件",用 event 字段区分类别。
    # ------------------------------------------------------------------

    def _log_scheduler_event(self, event: str, **fields: Any) -> None:
        record = {"ts": self.clock.now_ms(), "event": event, **fields}
        log_writer.append_jsonl(self.errors_log_path, record, root=self.log_root)

    # ==================================================================
    # 硬要求1 + 5(超时) + 6(建议输出触发点): 决策周期
    # ==================================================================

    def _decision_already_exists_for_cycle(self, branch: str, cycle_start: int, cycle_end: int) -> bool:
        """硬要求1:决策先落盘 decisions.jsonl 才会被 simulator 接受(§0 铁律3),
        因此"这个周期是否已经决策过"这件事本身就可以、也应该直接读回
        decisions.jsonl 来判定 -- 不需要调度器自己另外维护一份"已决策周期"的
        状态,decisions.jsonl 本身就是权威事实来源(与
        EvolutionOrchestrator 从两份 append-only 日志重放状态是同一种哲学)。
        """
        records = log_writer.read_jsonl(self.decisions_log_path, root=self.log_root)
        for r in records:
            if r.get("branch", "main") != branch:
                continue
            ts = r.get("ts")
            if ts is not None and cycle_start <= ts < cycle_end:
                return True
        return False

    @staticmethod
    def _scheduler_timeout_fallback_decision(ts: int, branch: str) -> Decision:
        return Decision(
            ts=ts,
            symbol=_SCHEDULER_TIMEOUT_FALLBACK_SYMBOL,
            action="hold",
            target_notional_pct=0.0,
            leverage=1,
            thesis=_SCHEDULER_TIMEOUT_FALLBACK_THESIS,
            falsifier=_SCHEDULER_TIMEOUT_FALLBACK_FALSIFIER,
            horizon=_SCHEDULER_TIMEOUT_FALLBACK_HORIZON,
            branch=branch,
        )

    def _call_trader_with_timeout(
        self, ts: int, positions: Any, latest_snapshot: dict, branch: str
    ) -> tuple[list[Decision], bool]:
        """硬要求5(Trader部分):Trader 自己的 retry-then-hold 兜底(§3.2)只
        防得住"LLM 返回了格式不对的输出" -- 一次真正挂起（网络卡死、对端不
        返回)的 llm_client 调用永远不会走到 Trader 的校验逻辑，因为它压根不
        返回。这里在调度层面再包一层墙钟超时。

        用线程 + future.result(timeout=...),不用 signal 模块的超时机制 --
        signal 在非主线程/Windows 上不可靠(APScheduler 的 job 常常不在主线程
        跑),而 concurrent.futures 是标准库里唯一跨平台都能用的同步调用超时
        方案。

        超时后不等待 Trader 内部重试收敛 -- 如果 llm_client 本身在挂起，
        Trader 的三次重试大概率会各自挂起同样长的时间，再等下去只是把超时
        乘以3；这里直接放弃这次 future(线程仍在后台跑，但调度器不再等它)，
        产出一条调度器级别的 hold 兜底决策，让主循环继续往前走。
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self.trader.decide,
            ts=ts,
            positions=positions,
            latest_snapshot=latest_snapshot,
            last_reflection_summary=self.last_reflection_summary,
            program_tactics=self.program_tactics,
            memory_query_text=self.memory_query_text,
            top_k=self.top_k,
            branch=branch,
        )
        try:
            decisions = future.result(timeout=self.trader_timeout_seconds)
            executor.shutdown(wait=False)
            return list(decisions), False
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False)  # 不等待挂起的线程收尾,避免调度器自己被拖死
            self._log_scheduler_event(
                "trader_timeout",
                branch=branch,
                decision_ts=ts,
                timeout_seconds=self.trader_timeout_seconds,
                detail=(
                    "Trader.decide() did not return within the scheduler-level timeout; this is "
                    "distinct from Trader's own internal retry-then-hold fallback (which only "
                    "triggers on malformed LLM output, never on a hung call that simply never "
                    "returns). Falling back to a scheduler-level hold decision and moving on "
                    "without waiting for Trader's internal retries."
                ),
            )
            return [self._scheduler_timeout_fallback_decision(ts, branch)], True

    def run_decision_cycle(self, branch: str = "main") -> dict:
        """§4.1 "每4小时整点" 那一行的落地实现,单个分支一次。

        编排顺序:冷启动闸门 -> 周期幂等性检查(硬要求1)-> 组装Trader输入 ->
        超时保护下调用Trader(硬要求5)-> 决策落盘 -> simulator撮合 ->
        生成 latest_advice.md(硬要求6)。
        """
        now = self.clock.now_ms()
        sim = self.simulators[branch]

        if self.cold_start_gate is not None and self.cold_start_gate.is_cold_start():
            # 冷启动期间 simulator.execute() 本来就会 Rejection(§4.0),这里提前
            # 短路只是为了不浪费一次 LLM 调用 -- 不是額外的校验逻辑,真正的闸门
            # 仍然是 simulator 自己的 cold_start_gate 检查。
            return {"branch": branch, "status": "skipped_cold_start", "ts": now}

        cycle_start, cycle_end = decision_cycle_window(now, self._decision_interval_hours)

        if self._decision_already_exists_for_cycle(branch, cycle_start, cycle_end):
            # 硬要求1:同一周期不重复调用 Trader、不重复落盘决策。
            return {
                "branch": branch,
                "status": "skipped_already_decided",
                "ts": now,
                "cycle_start": cycle_start,
                "cycle_end": cycle_end,
            }

        portfolio = sim.get_portfolio()
        positions = portfolio["positions"]
        snapshot = self.snapshot_provider(now) if self.snapshot_provider is not None else {}

        decisions, timed_out = self._call_trader_with_timeout(
            ts=now, positions=positions, latest_snapshot=snapshot, branch=branch
        )

        executed = []
        for decision in decisions:
            # §0 铁律3:决策必须先落盘,simulator才接受。
            sim.log_decision(decision)
            if self.next_bar_provider is not None:
                next_bar = self.next_bar_provider(decision.symbol, now)
            else:
                next_bar = {"open_time": now + 1, "open": 0.0}
            executed.append(sim.execute(decision, next_bar))

        advice_path = None
        if decisions:
            nav_agent = sim.get_portfolio()["nav"]
            nav_benchmark = self.benchmark_nav_provider(now) if self.benchmark_nav_provider else None
            nav_random = self.random_nav_provider(now) if self.random_nav_provider else None
            advice_path = self.generate_latest_advice(
                branch=branch,
                decision=decisions[-1],
                nav_agent=nav_agent,
                nav_benchmark=nav_benchmark,
                nav_random=nav_random,
            )

        return {
            "branch": branch,
            "status": "decided",
            "ts": now,
            "cycle_start": cycle_start,
            "cycle_end": cycle_end,
            "decisions": decisions,
            "executed": executed,
            "timed_out": timed_out,
            "advice_path": advice_path,
        }

    # ==================================================================
    # 硬要求2: 停机补偿 -- 决策不补、结算必须补齐
    # ==================================================================

    def run_settlement_catchup(self) -> dict:
        """崩溃/停机恢复入口。只处理资金费率结算的追赶(硬要求2)。决策周期
        故意没有对应的"追赶"方法 -- run_decision_cycle() 每次只认"现在"属于
        哪个周期,重启后调用它天然只会为当前周期产出至多一条决策,不会,也没有
        代码路径可以,去为中间跳过的周期补决策(§0 铁律"禁止在过时数据上偷看
        未来"的调度层体现:哪怕真尝试补,next_bar_provider 拿到的也已经是当下
        的价格,而 simulator.execute() 自己的"decision.ts < next_bar.open_time"
        检查也会拒绝任何试图用旧时间戳配新行情的决策 -- 但调度层本就不应该
        尝试,这里不需要、也没有实现"回填决策"的路径)。
        """
        if self.funding_rate_lookup is None:
            raise ValueError("funding_rate_lookup must be injected to run settlement catchup")

        now = self.clock.now_ms()
        last_ts = self._state.get("last_settlement_ts_ms")

        if last_ts is None:
            # 全新调度器,没有"上一次结算"可言 -- 不回溯到系统诞生那天去补历史
            # 结算,只是把游标设到当前时刻,从此刻开始追踪。
            self._state["last_settlement_ts_ms"] = now
            self._persist_state()
            return {"missed_instants": [], "settlements_by_instant": {}}

        missed = compute_missed_settlement_instants(last_ts, now, self._settle_hours_utc)

        settlements_by_instant: dict[int, list] = {}
        for ts in missed:
            instant_settlements = []
            for sim in self.simulators.values():
                # Simulator.settle_funding 本身对 (branch, symbol, ts) 幂等
                # (见 LOCKED/simulator.py 模块 docstring),这里的追赶逻辑只负责
                # "算出该补哪些时点、按顺序补",不需要、也不应该自己再实现一层
                # 去重。
                instant_settlements.extend(sim.settle_funding(ts, self.funding_rate_lookup))
            settlements_by_instant[ts] = instant_settlements
            self._state["last_settlement_ts_ms"] = ts
            self._persist_state()

        return {"missed_instants": missed, "settlements_by_instant": settlements_by_instant}

    # ==================================================================
    # 硬要求5(Researcher/Reflector部分): 失败跳过本次、记LOG、下周期自然重试
    # ==================================================================

    def run_reflection_cycle(self, branch: str = "main", window: int = 20) -> Optional[list]:
        if self.reflector is None:
            return None
        now = self.clock.now_ms()
        try:
            marks = self.reflector.reflect(
                now_ts=now, branch=branch, window=window, price_lookup=self.price_lookup
            )
        except Exception as exc:  # noqa: BLE001 -- 有意捕获任意异常,见模块docstring第5条
            self._log_scheduler_event(
                "reflector_failure",
                branch=branch,
                cycle_ts=now,
                error=repr(exc),
                detail=(
                    "Reflector.reflect() raised; skipping this cycle's reflection entirely. "
                    "No retry-within-call -- the next scheduled reflection tick IS the retry."
                ),
            )
            return None
        return marks

    def run_daily_research(self, date_str: str, queries: Optional[list[str]] = None) -> Optional[Path]:
        if self.researcher is None:
            return None
        now = self.clock.now_ms()
        try:
            return self.researcher.daily_research(ts=now, date_str=date_str, queries=queries)
        except Exception as exc:  # noqa: BLE001
            self._log_scheduler_event(
                "researcher_daily_failure",
                cycle_ts=now,
                date_str=date_str,
                error=repr(exc),
                detail=(
                    "Researcher.daily_research() raised; skipping this cycle's research entirely. "
                    "No retry-within-call -- tomorrow's scheduled research tick IS the retry."
                ),
            )
            return None

    def run_cold_start_research(
        self,
        universe_symbols: list[str],
        price_history: dict,
        funding_rate_history: Optional[dict] = None,
        min_hypotheses: int = 10,
    ) -> Optional[dict]:
        if self.researcher is None:
            return None
        now = self.clock.now_ms()
        try:
            return self.researcher.run_cold_start_research(
                ts=now,
                universe_symbols=universe_symbols,
                price_history=price_history,
                funding_rate_history=funding_rate_history,
                min_hypotheses=min_hypotheses,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_scheduler_event(
                "researcher_cold_start_failure",
                cycle_ts=now,
                error=repr(exc),
                detail="Researcher.run_cold_start_research() raised; COLD_START cannot complete this "
                "attempt. Skipping -- the next time the scheduler drives cold-start research IS the retry.",
            )
            return None

    # ==================================================================
    # 硬要求4: git merge 闸门
    # ==================================================================

    def run_ratchet_judgment(
        self,
        now_date: str,
        branch_navs: dict[str, list[tuple[str, float]]],
        benchmark_navs: list[tuple[str, float]],
        branch_dead_flags: Optional[dict[str, bool]] = None,
    ) -> dict:
        """§4.1 "每3日 00:30" 那一行。EvolutionOrchestrator.judge() 是唯一的
        棘轮裁决权威 -- 本方法不重新判分,只在裁决结果是 PROMOTE 时把它交给
        GitMergeExecutor 做真正的代码层晋升,并且把"棘轮判定"和"代码真的合并
        成功了"这两件事在调度层显式分开(effective_main_branch 只在 merged=True
        时前进)。
        """
        if self.evolution_orchestrator is None:
            raise ValueError("evolution_orchestrator must be injected to run ratchet judgment")

        verdicts = self.evolution_orchestrator.judge(
            now_date=now_date,
            branch_navs=branch_navs,
            benchmark_navs=benchmark_navs,
            branch_dead_flags=branch_dead_flags,
        )

        for branch, verdict in verdicts.items():
            if verdict.decision != "PROMOTE":
                continue  # 普通 ARCHIVE/FAIL,evolution_orchestrator 自己已经记过log了

            if self.git_merge_executor is None:
                self._log_scheduler_event(
                    "promotion_veto",
                    branch=branch,
                    now_date=now_date,
                    severity="critical",
                    reason="git_merge_executor not injected -- cannot execute a PROMOTE verdict at "
                    "the code level, treating this promotion as vetoed",
                )
                continue

            merge_result = self.git_merge_executor.attempt_merge(branch)
            if merge_result.merged:
                self.effective_main_branch = branch
                self._state["effective_main_branch"] = branch
                self._persist_state()
                self._log_scheduler_event(
                    "promotion_merged",
                    branch=branch,
                    now_date=now_date,
                    reason=merge_result.reason,
                )
            else:
                # 这是一个比普通 ARCHIVE 更严重的失败模式 -- "赢了分数,代码却会
                # 炸"。severity=critical 把它和普通的 trader_timeout/reflector_
                # failure 区分开来,大声记录,且绝不允许 effective_main_branch
                # 静默前进(它就是停在原地,即使
                # evolution_orchestrator.current_main_branch 已经在纯棘轮判定
                # 意义上指向了这个候选分支)。
                self._log_scheduler_event(
                    "promotion_veto",
                    branch=branch,
                    now_date=now_date,
                    severity="critical",
                    reason=merge_result.reason,
                    test_suite_passed=merge_result.test_suite_passed,
                    detail=(
                        "ratchet verdict was PROMOTE (this branch WON on score) but "
                        "GitMergeExecutor refused the real git merge -- the promotion is vetoed "
                        "at the code level; effective_main_branch stays put even though "
                        "EvolutionOrchestrator.current_main_branch has already flipped to this "
                        "branch in its own pure-judgment bookkeeping"
                    ),
                )

        return verdicts

    # ==================================================================
    # 其余 §4.1 日常调度节拍的薄封装(非六条硬要求本身,但完整调度器该有的
    # 挂载点;都只是直接转发给已经各自测试过的 LOCKED 模块,不额外造轮子)。
    # ==================================================================

    def mark_daily_nav(self, date: str, nav_agent: float, nav_benchmark: float, nav_random: float) -> None:
        if self.scorer is None:
            raise ValueError("scorer must be injected to mark daily NAV")
        self.scorer.daily_mark(nav_agent=nav_agent, nav_benchmark=nav_benchmark, nav_random=nav_random, date=date)

    def run_circuit_breaker_check(self, nav_series, now_ts: Optional[int] = None):
        if self.circuit_breaker is None:
            raise ValueError("circuit_breaker must be injected to run a circuit breaker check")
        effective_now = now_ts if now_ts is not None else self.clock.now_ms()
        return self.circuit_breaker.check(nav_series, now_ts=effective_now)

    def run_monthly_report(self, **kwargs) -> str:
        if self.scorer is None:
            raise ValueError("scorer must be injected to run the monthly report")
        return self.scorer.monthly_report(**kwargs)

    # ==================================================================
    # 硬要求6: latest_advice.md
    # ==================================================================

    def generate_latest_advice(
        self,
        branch: str,
        decision: Decision,
        nav_agent: Optional[float],
        nav_benchmark: Optional[float] = None,
        nav_random: Optional[float] = None,
    ) -> Path:
        """§5 建议输出。每次决策周期后调用,**覆盖写**(不是追加)-- "latest"
        的含义就是只保留最新状态,不是一份不断增长的历史(那是 decisions.jsonl/
        trades.jsonl 的职责,它们才走 append_only_writer)。这正是本方法不通过
        LOCKED.log_writer(它压根没有提供覆盖写接口)、而是直接 Path.write_text
        的原因 -- latest_advice.md 结构性地不属于"LOG 区历史记录"这条铁律管辖
        的对象集合,它是一份始终只反映"现在"的人类可读快照。

        内容顺序严格按任务要求:
          1. 固定免责声明(逐字,最上方)
          2. 三线净值对比(agent/BTC_HOLD/随机)
          3. 本周期建议动作
          4. thesis/falsifier 原文(不摘要、不改写)
        """
        nav_benchmark_display = "N/A" if nav_benchmark is None else nav_benchmark
        nav_random_display = "N/A" if nav_random is None else nav_random
        nav_agent_display = "N/A" if nav_agent is None else nav_agent

        lines = [
            DISCLAIMER,
            "",
            f"_generated ts={self.clock.now_ms()} branch={branch}_",
            "",
            "## NAV Comparison",
            "",
            f"- Agent ({branch}): {nav_agent_display}",
            f"- BTC_HOLD benchmark: {nav_benchmark_display}",
            f"- Random agent: {nav_random_display}",
            "",
            "## This Cycle's Suggested Action",
            "",
            f"- symbol: {decision.symbol}",
            f"- action: {decision.action}",
            f"- target_notional_pct: {decision.target_notional_pct}",
            f"- leverage: {decision.leverage}",
            f"- horizon: {decision.horizon}",
            "",
            "## Thesis (verbatim -- not summarized or paraphrased)",
            "",
            decision.thesis,
            "",
            "## Falsifier (verbatim -- not summarized or paraphrased)",
            "",
            decision.falsifier,
            "",
        ]
        content = "\n".join(lines)

        base = self.log_root if self.log_root is not None else log_writer.LOG_ROOT
        path = base / self.advice_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
