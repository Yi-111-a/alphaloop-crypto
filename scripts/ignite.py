"""
scripts/ignite.py —— M5 阶段三:真实点火入口。

这是全项目唯一允许调用真实墙钟(datetime.now)、真实构造 SystemClock、真实
构造 anthropic 客户端的地方——main.py 自身刻意不做这些事(见 main.py 模块
docstring),这里就是它说的"启动脚本"。

llm_client 说明(用户明确纠正过一次的设计决策,写进这里而不是只留在对话
记录里):Trader/Researcher/Reflector 需要的"AI推理能力"不通过独立的
ANTHROPIC_API_KEY/anthropic SDK 调用——运行本脚本的 Claude Code agent 本身
就是这个推理能力的来源("你是agent,所有ai操作都是由agent开始做")。
llm_client 用 scripts/llm_bridge.py 的 AgentBridgeLLMClient 实现:每次
Trader/Researcher/Reflector 需要一次"LLM调用"时,把 prompt 写到
state/llm_pending_request.json 并阻塞轮询等待响应文件——由签入的 agent
(我)读 pending request、推理、调 llm_bridge.respond_to_pending() 写回
响应。main.py 已有的失败隔离机制(Trader外层超时线程包装、
Researcher/Reflector的try/except)直接复用这条"迟迟没有响应就超时"的路径
作为"agent没有及时签入"场景的兜底,不需要额外实现。

调度节拍说明(点火实盘shakedown过程中发现main.py刻意把这些"该多久跑一次"
的节拍决策全部留给启动脚本 —— run_reflection_cycle/run_daily_research/
run_ratchet_judgment/mark_daily_nav/run_circuit_breaker_check 都是main.py
暴露的"薄封装",本身不含任何cadence gating,main.py注释里明确写了这是
调度层(本脚本)的职责,不是main.py该管的):
    - 决策周期:30分钟(config.cycle.decision_interval_hours=0.5,用户为
      "快速验证"主动要求从4小时收紧,2026-07-14),硬要求1的幂等性本身在
      main.py里,这里只管"到点了就调"。
    - 结算追赶/风控哨兵:每小时(已有实现,未改动)。
    - 反思(Reflector):每天 config.cycle.reflection_per_day 次(现为48次/天,
      等价于每30分钟一次,与决策周期同步),这里按24h/次数均匀分布节拍。
    - 研究(Researcher):每 config.cycle.research_interval_hours 小时一次
      (现为1小时,原先是"每个UTC自然日一次",这次改造前没有独立配置项、
      直接硬编码在本脚本里)。
    - 每日净值记录(nav.tsv 三线)+ 熔断检查:仍是每个UTC自然日一次不变——
      这条不能跟着决策周期一起收紧,因为nav.tsv本身是LOCKED区
      scorer.daily_mark()维护的日粒度权威历史,棘轮评分(ratchet_score)
      是按日历日对齐设计、已经审过的核心逻辑,不在这次改造范围内。
    - 棘轮判定(EvolutionOrchestrator):每 config.cycle.ratchet_interval_hours
      小时一次(现为2小时,原为3天)。用户要求棘轮判定加速到2小时一次时,
      我们特意确认过"方案2"——不改nav.tsv/ratchet_score本身的日粒度语义,
      因为那是已审过的LOCKED核心评分逻辑;这里改的只是"多久调一次
      run_ratchet_judgment"这个调度频率,它读到的nav.tsv数据本身仍然是
      每天才新增一行,所以在还没有任何候选分支被注册(§4.1"分支提案"机制
      本身是后续迭代内容,不在本次点火范围内)之前,不管几小时调一次都是
      "0条候选分支裁决"的空跑,这是正确的空跑,不是bug——真正需要小时级
      粒度的"平行判定机制"(用本脚本已经在维护的nav_intraday.jsonl)要等
      将来真的有候选分支时才有意义去建,现在建了也没有东西可比较。
    - benchmark(BTC_HOLD)通过 LOCKED.baseline_agents.BTCHoldAgent 解析计算
      (不经过Simulator/execute()九步校验链,见该模块docstring的人类裁决);
      random 通过独立的 "random" Simulator 分支,由 RandomAgent 按与主决策
      相同的4h节奏产出决策,但完全不经过Trader/LLM(它的存在意义就是"同样
      规则下不用脑子能赚多少",混入LLM签入延迟毫无意义)。random分支使用
      自己独立的 CircuitBreaker 实例(而不是与main共用),因为熔断器内部
      跟踪的是"迄今见过的NAV峰值"这类分支自身状态,main和random是两条完全
      独立的净值轨迹,共用一个实例会让二者互相污染彼此的冻结状态。

跑法:
    cd alphaloop
    python scripts/ignite.py            # 前台运行,Ctrl+C 停止
    nohup python scripts/ignite.py &     # 后台常驻

本脚本不修改任何 LOCKED 区业务逻辑,只是把已经各自独立测试过的 LOCKED/ASSET
模块按 main.py 的 AlphaLoopScheduler 接口接起来,加一层真实时钟驱动的轮询
循环。
"""
from __future__ import annotations

import datetime
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml  # noqa: E402

from llm_bridge import AgentBridgeLLMClient  # noqa: E402

from LOCKED import log_writer  # noqa: E402
from LOCKED.baseline_agents import BTCHoldAgent, RandomAgent  # noqa: E402
from LOCKED.circuit_breaker import CircuitBreaker  # noqa: E402
from LOCKED.clock import SystemClock  # noqa: E402
from LOCKED.cold_start import ColdStartGate  # noqa: E402
from LOCKED.data_pipeline import DataPipeline  # noqa: E402
from LOCKED.evolution_orchestrator import EvolutionOrchestrator  # noqa: E402
from LOCKED.git_merge_executor import GitMergeExecutor  # noqa: E402
from LOCKED.reflector import Reflector  # noqa: E402
from LOCKED.scorer import Scorer  # noqa: E402
from LOCKED.simulator import Simulator  # noqa: E402
from LOCKED.universe_filter import UniverseFilter  # noqa: E402

from ASSET.memory.engine import MemoryStore  # noqa: E402
from ASSET.strategy.researcher import Researcher  # noqa: E402
from ASSET.strategy.trader import Trader  # noqa: E402

import main as main_module  # noqa: E402

LOG_ROOT = PROJECT_ROOT / "LOG"
STATE_ROOT = PROJECT_ROOT / "state"
GENESIS_PATH = PROJECT_ROOT / "ASSET" / "research_notes" / "genesis.md"
HEARTBEAT_PATH = STATE_ROOT / "ignite_heartbeat.json"
SCHEDULE_STATE_PATH = STATE_ROOT / "ignite_schedule_state.json"
UNIVERSE_PATH = PROJECT_ROOT / "universe_active.json"

POLL_SECONDS = 60  # 主循环轮询间隔:每分钟检查一次是否有到期的周期任务


def build_llm_client():
    """见模块 docstring:llm_client 是"签入的 Claude Code agent"本身,通过
    state/ 下的请求/响应文件握手,不调用任何外部 LLM API。
    timeout_seconds 故意给得比 4h 决策周期短得多——agent 应该在合理时间内
    (设计上以"分钟"为量级)签入响应,超时说明 agent 这一轮没有及时签入,
    应该让这个具体周期走 main.py 已有的失败隔离路径(Trader的调度层超时
    兜底 / Researcher-Reflector的try-except-skip),而不是让整个循环无限期
    卡死等一个可能永远不会来的响应。"""
    return AgentBridgeLLMClient(poll_seconds=2.0, timeout_seconds=1800.0)


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_heartbeat(clock, extra: dict) -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts_ms": clock.now_ms(), **extra}
    HEARTBEAT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _utc_date_str(ts_ms: int) -> str:
    """ts_ms 转 ISO 日期字符串,供 daily_mark/daily_research/ratchet 用作
    "今天是哪一天"的判断依据。这里用的是传入的 ts_ms(来自 clock.now_ms()),
    不是 datetime.now() 本身——本文件是全项目唯一允许读真实墙钟的地方,
    SystemClock 才是那个真正调用 time.time() 的类,这里只是格式化一个已经
    读到的时间戳,没有引入新的墙钟读数来源。"""
    return datetime.datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")


def _utc_research_label(ts_ms: int) -> str:
    """研究改成每小时一次(cycle.research_interval_hours)之后,不能再拿
    _utc_date_str 那种纯日期字符串当 Researcher.daily_research(date_str=...)
    的参数——该方法内部按 f"{date_str}.md" 写文件(见 ASSET/strategy/
    researcher.py::daily_research,open模式是覆盖写不是追加),一天24次
    调用如果都传同一个日期字符串,后23次会依次覆盖掉前面的研究笔记,只留下
    当天最后一小时那一份。这里改成"日期-小时"粒度的标签,让每小时的研究
    笔记各自落到独立文件,不互相覆盖;daily_research本身不派生"今天"是
    哪天,接受调用方传什么标签就用什么标签,所以这是纯调用方(本脚本)的
    改动,不需要碰ASSET/researcher.py。"""
    return datetime.datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d-%H")


def load_schedule_state() -> dict:
    if SCHEDULE_STATE_PATH.exists():
        try:
            return json.loads(SCHEDULE_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "last_reflection_ts": None,
        # last_research_date 字段已废弃(研究改成每小时一次后,cadence 状态
        # 挪到 main() 里的本地变量 last_research_ms,与 risk_check/settlement
        # 用同一种"重启后立即补跑一次"的模式,不再需要跨重启持久化)。
        "last_daily_mark_date": None,
        "last_ratchet_ts": None,
        "btc_hold_entry_price": None,
    }


def save_schedule_state(state: dict) -> None:
    SCHEDULE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _date_str_to_utc_midnight_ms(date_str: str) -> int:
    """circuit_breaker.check() 要的是整数毫秒时间戳(内部用 ts_ms//86_400_000
    算UTC自然日边界),而 nav.tsv/ratchet_score 那一路用的是 ISO 日期字符串
    (按字典序比较等价于按时间比较,见 scorer._slice_from_date docstring)——
    两者是同一份 nav.tsv 历史数据分别喂给两个要求不同格式的LOCKED模块,这里
    只是格式转换,不是引入新的时间语义。"""
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


def _navs_to_ms_series(navs: list[tuple[str, float]]) -> list[tuple[int, float]]:
    return [(_date_str_to_utc_midnight_ms(d), nav) for d, nav in navs]


def read_nav_tsv_series() -> dict[str, list[tuple[str, float]]]:
    """从 LOG/nav.tsv 重建 main/benchmark/random 三条 (date, nav) 序列,供
    run_ratchet_judgment / circuit_breaker.check 使用 —— nav.tsv 本身就是
    scorer.daily_mark() 唯一的真实历史来源,不需要调度层自己另外维护一份
    重复的净值序列存储(与main.py"decisions.jsonl本身就是权威事实来源"的
    重放哲学一致)。"""
    path = LOG_ROOT / "nav.tsv"
    series: dict[str, list[tuple[str, float]]] = {"main": [], "benchmark": [], "random": []}
    if not path.exists():
        return series
    column_by_branch = {"main": "nav_agent", "benchmark": "nav_benchmark", "random": "nav_random"}
    with open(path, "r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            date = row.get("date")
            if not date:
                continue
            for branch, column in column_by_branch.items():
                try:
                    series[branch].append((date, float(row[column])))
                except (KeyError, ValueError):
                    pass
    return series


def read_evo_branch_daily_nav_series() -> dict[str, list[tuple[str, float]]]:
    """方案2(用户2026-07-14确认):不改nav.tsv/scorer.ratchet_score的日粒度
    权威语义,而是把本脚本已经在维护的小时级 LOG/nav_intraday_branches.jsonl
    降采样成"每个evo分支每天最后一次看到的净值"这种官方棘轮评分要求的
    (date_str, nav) 序列格式——LOCKED的评分逻辑本身完全不变,只是ignite.py
    侧多做一步聚合,让候选分支真的能被judge()评估,而不是因为branch_navs
    里缺了对应entry而被(EvolutionOrchestrator.judge()内部本来就有的防御性
    过滤)静默跳过。"""
    records = log_writer.read_jsonl("nav_intraday_branches.jsonl", root=LOG_ROOT)
    by_branch_date: dict[str, dict[str, float]] = {}
    for r in records:
        branch, ts, nav = r.get("branch"), r.get("ts"), r.get("nav")
        if branch is None or ts is None or nav is None:
            continue
        by_branch_date.setdefault(branch, {})[_utc_date_str(ts)] = float(nav)  # 同一天多条 -> 保留最后一次(append-only,后写的在后面)
    return {branch: sorted(date_nav.items()) for branch, date_nav in by_branch_date.items()}


def run_cold_start(scheduler, dp: DataPipeline, uf: UniverseFilter, researcher: Researcher, clock, config: dict) -> None:
    print("=== COLD_START: 拉取历史数据 + universe + genesis 研究 ===", flush=True)

    print("universe_filter.refresh() ...", flush=True)
    universe_result = uf.refresh()
    symbols = universe_result["symbols"]
    print(f"  universe: {len(symbols)} 个交易对: {symbols}", flush=True)
    if not symbols:
        raise SystemExit("universe_active.json 为空,COLD_START 无法继续(流动性筛选后无合格标的)")

    history_days = config["data"]["history_days"]
    print(f"拉取 {history_days} 天历史K线 + 资金费率历史(共 {len(symbols)} 个标的)...", flush=True)
    price_history: dict = {}
    funding_rate_history: dict = {}
    since_ms = clock.now_ms() - history_days * 86_400_000
    for symbol in symbols:
        try:
            ohlcv = dp.fetch_ohlcv(symbol, config["data"]["timeframe"], since=since_ms, limit=100000)
            price_history[symbol] = ohlcv["close"].tolist()
            funding = dp.fetch_funding_rate_history(symbol, since=since_ms, limit=100000)
            funding_rate_history[symbol] = funding.to_dict("records")
            print(f"  {symbol}: {len(ohlcv)} 根K线, {len(funding)} 条资金费率", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol}: 拉取失败,跳过该标的的历史画像 ({exc!r})", flush=True)

    print("Researcher.run_cold_start_research() ...", flush=True)
    result = scheduler.run_cold_start_research(
        universe_symbols=symbols,
        price_history=price_history,
        funding_rate_history=funding_rate_history,
        min_hypotheses=10,
    )
    if result is None:
        raise SystemExit("run_cold_start_research 失败(见 LOG/scheduler_errors.jsonl),COLD_START 无法完成")
    print(f"  genesis.md: {result['hypothesis_count']} 条假设, 写入 {result['genesis_path']}", flush=True)

    state = scheduler.cold_start_gate.check_and_transition(
        hypothesis_count=result["hypothesis_count"], ts=clock.now_ms()
    )
    print(f"冷启动状态机: {state}", flush=True)
    if state != "NORMAL":
        raise SystemExit(f"COLD_START 未能切换到 NORMAL(当前: {state}),不继续点火")


def run_random_branch_cycle(sim_random, random_agent, next_bar_provider, now: int, interval_hours: int) -> bool:
    """随机对照组(RandomAgent)按与主策略相同的4h决策节奏产出+执行决策,
    但完全不经过 Trader/LLM——它的存在意义就是回答"同样规则下不用脑子能
    赚多少"(见 LOCKED/baseline_agents.py 模块docstring),混进LLM签入延迟
    对它没有任何意义,反而会不必要地拖慢这条对照线。

    幂等性检查直接照搬 main.py._decision_already_exists_for_cycle 同一套
    "读 decisions.jsonl 按 branch+ts 窗口判断"的逻辑(读的是同一份权威
    事实来源,不需要另外维护状态),只是这里没有复用那个私有方法,而是在
    本脚本内独立实现一份同样简单的读回校验。返回 True 表示本次真的产出
    并执行了一条新决策,False 表示这个周期已经决策过、本次是空转。
    """
    cycle_start, cycle_end = main_module.decision_cycle_window(now, interval_hours)
    records = log_writer.read_jsonl("decisions.jsonl", root=LOG_ROOT)
    already = any(
        r.get("branch") == "random" and cycle_start <= r.get("ts", -1) < cycle_end
        for r in records
    )
    if already:
        return False

    decision = random_agent.decide(now)
    sim_random.log_decision(decision)
    next_bar = next_bar_provider(decision.symbol, now)
    sim_random.execute(decision, next_bar)
    return True


def main() -> None:
    config = load_config()
    clock = SystemClock()
    llm_client = build_llm_client()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)

    dp = DataPipeline(exchange_id=config["data"]["exchange"], clock=clock)
    uf = UniverseFilter(config, clock=clock)
    memory_store = MemoryStore(db_path=PROJECT_ROOT / "ASSET" / "memory" / "memory.db")
    researcher = Researcher(llm_client=llm_client, memory_store=memory_store, genesis_path=GENESIS_PATH)
    trader = Trader(llm_client=llm_client, memory_store=memory_store)
    reflector = Reflector(llm_client=llm_client, memory_store=memory_store, log_root=LOG_ROOT)
    scorer = Scorer(config, log_root=LOG_ROOT)
    cold_start_gate = ColdStartGate(genesis_path=GENESIS_PATH, min_hypothesis_count=10, log_root=LOG_ROOT)
    circuit_breaker = CircuitBreaker(config, log_root=LOG_ROOT)
    # random 分支用独立的 CircuitBreaker 实例:熔断器内部跟踪的是"迄今见过
    # 的NAV峰值"等分支自身状态,main/random是两条独立净值轨迹,共用一个
    # 实例会让二者的回撤/冻结状态互相污染。
    circuit_breaker_random = CircuitBreaker(config, log_root=LOG_ROOT)
    evolution_orchestrator = EvolutionOrchestrator(config, scorer=scorer, log_root=LOG_ROOT)
    git_merge_executor = GitMergeExecutor(repo_path=PROJECT_ROOT, log_root=LOG_ROOT)

    schedule_state = load_schedule_state()

    # BTC_HOLD 基准(benchmark):解析计算,不经过Simulator(见 baseline_agents.py
    # 模块docstring的人类裁决)。entry_price 只应该在整个点火生命周期里确定
    # 一次,重启时必须复用同一个值(否则每次重启都相当于"重新买入"一次,
    # 基准线会被人为拉高/拉低),因此持久化进 schedule_state。
    btc_hold_agent = BTCHoldAgent.from_config(config)
    if schedule_state.get("btc_hold_entry_price") is not None:
        btc_hold_agent.enter(entry_price=float(schedule_state["btc_hold_entry_price"]))
    else:
        entry_ticker = dp.fetch_latest_snapshot(["BTC/USDT:USDT"])["BTC/USDT:USDT"]
        entry_price = float(entry_ticker["last"])
        btc_hold_agent.enter(entry_price=entry_price)
        schedule_state["btc_hold_entry_price"] = entry_price
        save_schedule_state(schedule_state)
        print(f"BTC_HOLD 基准建仓: entry_price={entry_price}", flush=True)

    # 重启时如果 universe_active.json 已存在(COLD_START 在之前的进程里已经
    # 跑过),这里必须真的把合格名单传给 Simulator——此前的实现只在
    # "本次进程自己跑COLD_START"这一个分支里传真实symbols,重启后永远走的是
    # universe_symbols=None,等价于每次重启都悄悄关掉了universe筛选校验
    # (§0铁律"只允许合格名单内标的"),这是本次点火过程中发现的另一个真实bug。
    symbols = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))["symbols"] if UNIVERSE_PATH.exists() else None

    sim_main = Simulator(
        # Simulator 期望零参 callable(见 LOCKED/cold_start.py 模块docstring
        # "Simulator(..., cold_start_gate=gate.is_cold_start)"的既定契约,与
        # AlphaLoopScheduler 期望整个gate对象、自己调 .is_cold_start() 不同,
        # 这是本次点火过程中发现的一个真实bug:此前三处 Simulator() 构造都
        # 传了裸对象,execute() 里的 self.cold_start_gate() 会直接 TypeError。
        config=config, circuit_breaker=circuit_breaker, cold_start_gate=cold_start_gate.is_cold_start,
        universe_symbols=symbols,
        db_path=STATE_ROOT / "portfolio_main.db", log_root=LOG_ROOT, branch="main", resume=True,
    )

    def next_bar_provider(symbol: str, ts: int) -> dict:
        """实盘场景下"下一根K线"这个历史回放概念本身不成立——决策产生的时刻
        就是实时前沿,还不存在时间戳晚于ts的已收盘K线(config.yaml的4h周期
        要再等最多4小时才会收出下一根)。fetch_ohlcv(since=ts) 因此总是返回
        空结果,这是COLD_START之后第一次真实决策周期就复现的真实bug,不是
        假设性的边界情况。撮合价格改用当前真实ticker最新成交价(与
        snapshot_provider同一路径),open_time 取 ts+1 只是为了满足
        simulator.execute() 第1步"decision.ts 必须严格早于 open_time"的
        防未来偷看校验,不代表真的存在一根时间戳为ts+1的K线。"""
        ticker = dp.fetch_latest_snapshot([symbol])[symbol]
        return {"open_time": ts + 1, "open": float(ticker["last"])}

    def snapshot_provider(ts: int) -> dict:
        symbols_now = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))["symbols"] if UNIVERSE_PATH.exists() else []
        return dp.fetch_latest_snapshot(symbols_now) if symbols_now else {}

    def funding_rate_lookup(symbol: str, ts: int) -> float:
        history = dp.fetch_funding_rate_history(symbol, since=ts - 3_600_000, limit=5)
        matching = history[history["timestamp"] == ts]
        if not matching.empty:
            return float(matching.iloc[0]["funding_rate"])
        return dp.fetch_funding_rate(symbol)

    def price_lookup(symbol: str, ts: int) -> float:
        ohlcv = dp.fetch_ohlcv(symbol, config["data"]["timeframe"], since=ts - 3_600_000, limit=2)
        return float(ohlcv.iloc[-1]["close"])

    def benchmark_nav_provider(ts: int) -> float:
        ticker = dp.fetch_latest_snapshot(["BTC/USDT:USDT"])["BTC/USDT:USDT"]
        return btc_hold_agent.nav(price=float(ticker["last"]))

    def random_nav_provider(ts: int) -> float:
        return scheduler.simulators["random"].get_portfolio()["nav"]

    risk_check_cfg = config.get("risk_check", {}) or {}
    _recent_window_minutes = int(risk_check_cfg.get("recent_window_minutes", 300))

    def recent_price_provider(symbol: str, ts: int) -> list:
        """每小时确定性风控检查用(见 config.yaml risk_check 段 + main.py
        run_risk_check_cycle)。只拉最近一小段分钟线(默认5小时=300根1分钟K线,
        OKX单次调用上限内,不需要分页),不是2年历史那个量级。"""
        since = ts - _recent_window_minutes * 60_000
        ohlcv = dp.fetch_ohlcv(symbol, "1m", since=since, limit=_recent_window_minutes)
        return [(int(row["timestamp"]), float(row["close"])) for _, row in ohlcv.iterrows()]

    scheduler = main_module.AlphaLoopScheduler(
        config=config, clock=clock, simulators={"main": sim_main}, trader=trader,
        reflector=reflector, researcher=researcher, memory_store=memory_store, scorer=scorer,
        evolution_orchestrator=evolution_orchestrator, circuit_breaker=circuit_breaker,
        cold_start_gate=cold_start_gate, git_merge_executor=git_merge_executor,
        next_bar_provider=next_bar_provider, snapshot_provider=snapshot_provider,
        funding_rate_lookup=funding_rate_lookup, price_lookup=price_lookup,
        recent_price_provider=recent_price_provider,
        benchmark_nav_provider=benchmark_nav_provider, random_nav_provider=random_nav_provider,
        log_root=LOG_ROOT, state_path=STATE_ROOT / "scheduler_state.json",
        # main.py 默认 trader_timeout_seconds=30.0,是为"真实API几秒内应该
        # 返回"这个假设设计的。llm_client 现在是要等一个人/agent签入并手动
        # 响应的文件握手(AgentBridgeLLMClient),30秒内不可能真的等到——
        # 这里放宽到与 AgentBridgeLLMClient 自己的 timeout_seconds 一致
        # (1800s=30分钟),避免调度层自己先超时抛弃了后台还在阻塞等待的
        # 响应线程,产生"两边各自超时、互相不知道对方状态"的混乱。
        trader_timeout_seconds=1800.0,
    )

    if cold_start_gate.is_cold_start():
        run_cold_start(scheduler, dp, uf, researcher, clock, config)
        # 重新用 COLD_START 产出的真实名单构造 Simulator(替换掉上面可能是
        # None 的占位实例)。
        symbols = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))["symbols"]
        scheduler.simulators["main"] = Simulator(
            config=config, circuit_breaker=circuit_breaker, cold_start_gate=cold_start_gate.is_cold_start,
            universe_symbols=symbols, db_path=STATE_ROOT / "portfolio_main.db",
            log_root=LOG_ROOT, branch="main", resume=True,
        )

    if symbols is None:
        raise SystemExit("universe_active.json 仍不存在,COLD_START 逻辑有误,无法继续点火")

    # random 对照组:独立 Simulator 分支 + 独立 CircuitBreaker + RandomAgent
    # (seed 固定=42,可复现)。
    sim_random = Simulator(
        config=config, circuit_breaker=circuit_breaker_random, cold_start_gate=cold_start_gate.is_cold_start,
        universe_symbols=symbols, db_path=STATE_ROOT / "portfolio_random.db",
        log_root=LOG_ROOT, branch="random", resume=True,
    )
    scheduler.simulators["random"] = sim_random
    random_agent = RandomAgent(universe_symbols=symbols, seed=42, branch="random")

    # 用户要求的多分支并行(2026-07-14,快速验证阶段):每个候选分支各自独立
    # 仓位、各自走完整的 Trader/LLM 签入决策周期(与 random 对照组不同——
    # random 刻意不经过 Trader,这几个 evo 分支的存在意义就是要真的比较不同
    # 战术下 Trader 的表现,所以必须真的经过签入)。战术差异通过
    # scheduler.program_tactics 这个既有的公开属性在每次调用前手动切换实现
    # (main.py 的 _call_trader_with_timeout 本来就会把它转发给 Trader.decide(),
    # 见 main.py:344 —— 这里只是从 ignite.py 侧按分支复用同一个已有机制,
    # 不需要改 main.py 的接口)。分支名遵循 LOCKED 区"evo/YYYYMMDD-简述"命名
    # 约定,一旦注册永不重用(见 evolution_orchestrator.register_branch
    # docstring),日期写死在这里是有意的——它是这一批分支被创建的日期,不是
    # "今天"的意思,不应该随进程重启的当天日期变化。
    EVO_BRANCH_TACTICS = {
        "evo/20260714-aggressive": (
            "进取战术:在假设置信度足够(至少两条独立假设互相印证,或有真实"
            "价格/资金费率数据强支撑)时,愿意用更高杠杆、更集中的仓位捕捉"
            "机会,不要求像main分支那样保守分散;但仍必须为每笔非hold决策"
            "写出明确的falsifier_condition并严格执行,进取不等于不设止损。"
        ),
        "evo/20260714-conservative": (
            "保守战术:只在多条独立假设互相印证、且没有明显反向风险信号时"
            "才建仓;仓位规模、杠杆都应明显低于main分支的对应决策,宁可错过"
            "一部分机会也不放大不确定性下的敞口;对新标的(缺乏完整2年历史"
            "或资金费率样本不足96天)一律用最小仓位或直接观望。"
        ),
    }
    evo_simulators: dict[str, Simulator] = {}
    for evo_branch, _tactics in EVO_BRANCH_TACTICS.items():
        if evolution_orchestrator.branch_meta(evo_branch) is None:
            registered = evolution_orchestrator.register_branch(evo_branch, _utc_date_str(clock.now_ms()))
            print(f"注册候选分支: {evo_branch} (registered={registered})", flush=True)
        evo_cb = CircuitBreaker(config, log_root=LOG_ROOT)
        safe_evo = evo_branch.replace("/", "_").replace(":", "_")
        evo_sim = Simulator(
            config=config, circuit_breaker=evo_cb, cold_start_gate=cold_start_gate.is_cold_start,
            universe_symbols=symbols, db_path=STATE_ROOT / f"portfolio_{safe_evo}.db",
            log_root=LOG_ROOT, branch=evo_branch, resume=True,
        )
        evo_simulators[evo_branch] = evo_sim
        scheduler.simulators[evo_branch] = evo_sim

    reflection_interval_ms = int((24 * 3_600_000) // max(1, int(config["cycle"]["reflection_per_day"])))
    ratchet_interval_ms = int(float(config["cycle"]["ratchet_interval_hours"]) * 3_600_000)
    research_interval_ms = int(float(config["cycle"]["research_interval_hours"]) * 3_600_000)

    print("=== 进入常驻调度循环(每分钟轮询一次到期任务,Ctrl+C 停止)===", flush=True)
    last_settlement_check_ms = 0
    last_risk_check_ms = 0
    last_research_ms = 0
    risk_check_interval_ms = scheduler._risk_check_interval_hours * 3_600_000
    while True:
        now = clock.now_ms()
        try:
            scheduler.program_tactics = None  # main 分支不带任何战术偏置,重置掉上一轮evo循环可能残留的值
            result = scheduler.run_decision_cycle("main")
            if result["status"] == "decided":
                print(f"[{now}] 决策周期完成: {[d.action for d in result['decisions']]}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[{now}] run_decision_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)

        try:
            acted = run_random_branch_cycle(
                sim_random, random_agent, next_bar_provider, now, scheduler._decision_interval_hours
            )
            if acted:
                print(f"[{now}] random对照组决策周期完成", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[{now}] run_random_branch_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)

        # 多分支并行:每个 evo 候选分支各自走一遍完整的 Trader/LLM 签入决策
        # 周期,战术差异通过临时切换 scheduler.program_tactics 实现(见上面
        # EVO_BRANCH_TACTICS 定义处的说明)。循环结束后不需要额外重置——
        # 下一轮 while 顶部 main 分支调用前已经会重置一次。
        for evo_branch, evo_tactics in EVO_BRANCH_TACTICS.items():
            try:
                scheduler.program_tactics = evo_tactics
                evo_result = scheduler.run_decision_cycle(evo_branch)
                if evo_result["status"] == "decided":
                    print(f"[{now}] [{evo_branch}] 决策周期完成: {[d.action for d in evo_result['decisions']]}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] [{evo_branch}] run_decision_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)

        if now - last_settlement_check_ms >= 3_600_000:
            try:
                settlement_result = scheduler.run_settlement_catchup()
                if settlement_result["missed_instants"]:
                    print(f"[{now}] 结算补齐: {settlement_result['missed_instants']}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] run_settlement_catchup 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            last_settlement_check_ms = now

        # 用户新增护栏:每小时确定性风控检查,不经过Trader/LLM(见
        # LOCKED/position_risk_monitor.py + main.py.run_risk_check_cycle)。
        # 故意只对main分支做——random对照组的存在意义是"同样规则+零额外
        # 保护下瞎搞会怎样",给它加急停保护会不公平地美化这条对照线。
        if now - last_risk_check_ms >= risk_check_interval_ms:
            try:
                risk_result = scheduler.run_risk_check_cycle("main")
                if risk_result.get("triggered"):
                    print(f"[{now}] 紧急风控平仓触发: {risk_result['triggered']}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] run_risk_check_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)

            # 用户要求:面板净值曲线要能看小时级粒度,不是只有nav.tsv那种
            # 日线。这里不改scorer.daily_mark()/nav.tsv本身(那条线是LOCKED
            # 棘轮判定用的权威日线历史,改它的语义有风险),而是另开一份
            # 独立的、纯附加的 LOG/nav_intraday.jsonl,复用同一个已经在跑的
            # 每小时节拍顺手记一笔三线实时净值(snapshot_provider本身就是
            # 一次全universe ticker拉取,成本可忽略)。分钟级故意不做——
            # 决策4h一次、风控哨兵1h一次,真拉到分钟级只是徒增每分钟一次的
            # 交易所API调用,对这个研究系统没有实际信息增量。
            try:
                # snapshot_provider() 返回 {symbol: {"last":..,"bid":..,"ask":..,...}}
                # (dp.fetch_latest_snapshot 的原始ticker结构),而
                # Simulator.get_portfolio(snapshot=...) -> mark_to_market() 要的是
                # {symbol: price} 这种扁平结构(_unrealized_pnl 直接拿 price 做算术)——
                # 两者形状不同,这是刚才第一次重启后真实复现的bug,不是假设性边界情况。
                live_ticker_snapshot = snapshot_provider(now)
                live_price_snapshot = {
                    symbol: float(ticker["last"])
                    for symbol, ticker in live_ticker_snapshot.items()
                    if ticker.get("last") is not None
                }
                nav_agent_now = scheduler.simulators["main"].get_portfolio(snapshot=live_price_snapshot)["nav"]
                nav_random_now = scheduler.simulators["random"].get_portfolio(snapshot=live_price_snapshot)["nav"]
                nav_benchmark_now = benchmark_nav_provider(now)
                log_writer.append_jsonl(
                    "nav_intraday.jsonl",
                    {
                        "ts": now,
                        "nav_agent": nav_agent_now,
                        "nav_benchmark": nav_benchmark_now,
                        "nav_random": nav_random_now,
                    },
                    root=LOG_ROOT,
                )

                # 用户指出面板持仓表里的"未实现盈亏"一直是硬编码的"--"(webui/
                # static/index.html之前从没真正填过这一列)。webui/app.py 明确
                # 立了"零计算"的规矩(不能自己重新算NAV/盈亏),所以修法不是在
                # 面板里现算,而是让本脚本(允许调用LOCKED)复用
                # Simulator._unrealized_pnl 这个和真实结算完全同一套公式的
                # 方法,算出来直接落盘成"已经算好的数字"——面板照旧只读、不算。
                # 用 state/(而非LOG/)存,因为这是"当前最新标记价快照"而不是
                # 需要保留全部历史的append-only事件,与ignite_heartbeat.json
                # 是同一种性质。
                def _mark_positions(sim: Simulator, branch_name: str) -> dict:
                    return {
                        "ts": now,
                        "branch": branch_name,
                        "positions": [
                            {
                                "symbol": pos.symbol,
                                "mark_price": live_price_snapshot.get(pos.symbol, pos.entry_price),
                                "unrealized_pnl": Simulator._unrealized_pnl(
                                    pos, live_price_snapshot.get(pos.symbol, pos.entry_price)
                                ),
                            }
                            for pos in sim.positions.values()
                        ],
                    }

                positions_marked = _mark_positions(scheduler.simulators["main"], "main")
                (STATE_ROOT / "positions_marked_main.json").write_text(
                    json.dumps(positions_marked, ensure_ascii=False), encoding="utf-8"
                )

                # 用户要求的多分支功能:"总面板可以看他们的对比,然后每个分支
                # 又可以点进去看他们的仓位"——random对照组和每个evo候选分支
                # 都按同样的方式各自落一份标记快照 + 一条小时级净值记录。
                # nav_intraday.jsonl的既有三线schema(nav_agent/nav_benchmark/
                # nav_random)保持不动,不动它的既有读者(webui默认图表);
                # 这里新增一份独立的、long格式的 nav_intraday_branches.jsonl,
                # 覆盖random和所有evo分支,webui侧再合并展示,不影响已经在跑
                # 的旧代码路径。
                for other_branch, other_sim in {"random": sim_random, **evo_simulators}.items():
                    safe_other = other_branch.replace("/", "_").replace(":", "_")
                    marked = _mark_positions(other_sim, other_branch)
                    (STATE_ROOT / f"positions_marked_{safe_other}.json").write_text(
                        json.dumps(marked, ensure_ascii=False), encoding="utf-8"
                    )
                    nav_now = other_sim.get_portfolio(snapshot=live_price_snapshot)["nav"]
                    log_writer.append_jsonl(
                        "nav_intraday_branches.jsonl",
                        {"ts": now, "branch": other_branch, "nav": nav_now},
                        root=LOG_ROOT,
                    )
            except Exception:  # noqa: BLE001
                print(f"[{now}] nav_intraday/持仓标记 记录异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)

            last_risk_check_ms = now

        # 反思(Reflector):每天 reflection_per_day 次,均匀分布节拍。
        last_reflection_ts = schedule_state.get("last_reflection_ts")
        if last_reflection_ts is None or now - last_reflection_ts >= reflection_interval_ms:
            try:
                marks = scheduler.run_reflection_cycle("main")
                print(f"[{now}] 反思周期完成: {len(marks) if marks is not None else 0} 条", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] run_reflection_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            schedule_state["last_reflection_ts"] = now
            save_schedule_state(schedule_state)

        today = _utc_date_str(now)

        # 研究(Researcher):每 research_interval_hours 小时一次(用户要求从
        # "每UTC自然日一次"收紧到每小时一次)。标签改用日期-小时粒度(见
        # _utc_research_label),避免同一天多次调用互相覆盖研究笔记文件。
        if now - last_research_ms >= research_interval_ms:
            research_label = _utc_research_label(now)
            try:
                research_result = scheduler.run_daily_research(research_label)
                print(f"[{now}] 研究{'完成: ' + str(research_result) if research_result is not None else '本轮失败(已记录),下个节拍自然重试'}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] run_daily_research 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            last_research_ms = now

        # 每日净值记录(nav.tsv 三线)+ 熔断检查:每个UTC自然日一次。
        if schedule_state.get("last_daily_mark_date") != today:
            try:
                nav_agent = scheduler.simulators["main"].get_portfolio()["nav"]
                nav_benchmark = benchmark_nav_provider(now)
                nav_random = scheduler.simulators["random"].get_portfolio()["nav"]
                scheduler.mark_daily_nav(today, nav_agent=nav_agent, nav_benchmark=nav_benchmark, nav_random=nav_random)
                print(
                    f"[{now}] 每日净值记录: agent={nav_agent:.2f} benchmark={nav_benchmark:.2f} random={nav_random:.2f}",
                    flush=True,
                )

                nav_series = read_nav_tsv_series()
                try:
                    scheduler.run_circuit_breaker_check(_navs_to_ms_series(nav_series["main"]), now_ts=now)
                except Exception:  # noqa: BLE001
                    print(f"[{now}] circuit_breaker(main) 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
                try:
                    circuit_breaker_random.check(_navs_to_ms_series(nav_series["random"]), now_ts=now)
                except Exception:  # noqa: BLE001
                    print(f"[{now}] circuit_breaker(random) 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] 每日净值记录/熔断检查 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            schedule_state["last_daily_mark_date"] = today
            save_schedule_state(schedule_state)

        # 棘轮判定(EvolutionOrchestrator):每 ratchet_interval_hours 小时一次
        # (用户2026-07-14要求从3天收紧到2小时)。main的NAV仍然只读nav.tsv这份
        # 日粒度权威历史(LOCKED评分逻辑没变);evo候选分支的NAV来自
        # read_evo_branch_daily_nav_series()——按方案2从小时级nav_intraday_
        # branches.jsonl聚合出的日粒度序列,不是nav.tsv的一部分。
        last_ratchet_ts = schedule_state.get("last_ratchet_ts")
        if last_ratchet_ts is None or now - last_ratchet_ts >= ratchet_interval_ms:
            try:
                nav_series = read_nav_tsv_series()
                evo_nav_series = read_evo_branch_daily_nav_series()
                branch_navs = {"main": nav_series["main"]}
                branch_navs.update({b: s for b, s in evo_nav_series.items() if b in EVO_BRANCH_TACTICS})
                if len(nav_series["main"]) >= 2:
                    verdicts = scheduler.run_ratchet_judgment(
                        now_date=today,
                        branch_navs=branch_navs,
                        benchmark_navs=nav_series["benchmark"],
                    )
                    print(f"[{now}] 棘轮判定完成: {len(verdicts)} 条候选分支裁决", flush=True)
                else:
                    print(f"[{now}] 棘轮判定跳过: nav.tsv历史点数不足({len(nav_series['main'])})", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] run_ratchet_judgment 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            schedule_state["last_ratchet_ts"] = now
            save_schedule_state(schedule_state)

        write_heartbeat(clock, {"status": "running"})
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
