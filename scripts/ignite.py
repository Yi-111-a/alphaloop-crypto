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

跑法:
    cd alphaloop
    python scripts/ignite.py            # 前台运行,Ctrl+C 停止
    nohup python scripts/ignite.py &     # 后台常驻

本脚本不修改任何 LOCKED 区业务逻辑,只是把已经各自独立测试过的 LOCKED/ASSET
模块按 main.py 的 AlphaLoopScheduler 接口接起来,加一层真实时钟驱动的轮询
循环。
"""
from __future__ import annotations

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
    evolution_orchestrator = EvolutionOrchestrator(config, scorer=scorer, log_root=LOG_ROOT)
    git_merge_executor = GitMergeExecutor(repo_path=PROJECT_ROOT, log_root=LOG_ROOT)

    sim_main = Simulator(
        config=config, circuit_breaker=circuit_breaker, cold_start_gate=cold_start_gate,
        universe_symbols=None,  # COLD_START 完成前先不限制,refresh 后由 ignite 自己再传一次真实名单
        db_path=STATE_ROOT / "portfolio_main.db", log_root=LOG_ROOT, branch="main", resume=True,
    )

    def next_bar_provider(symbol: str, ts: int) -> dict:
        ohlcv = dp.fetch_ohlcv(symbol, config["data"]["timeframe"], since=ts, limit=2)
        row = ohlcv.iloc[0]
        return {"open_time": int(row["timestamp"]), "open": float(row["open"])}

    def snapshot_provider(ts: int) -> dict:
        universe_path = PROJECT_ROOT / "universe_active.json"
        symbols = json.loads(universe_path.read_text(encoding="utf-8"))["symbols"] if universe_path.exists() else []
        return dp.fetch_latest_snapshot(symbols) if symbols else {}

    def funding_rate_lookup(symbol: str, ts: int) -> float:
        history = dp.fetch_funding_rate_history(symbol, since=ts - 3_600_000, limit=5)
        matching = history[history["timestamp"] == ts]
        if not matching.empty:
            return float(matching.iloc[0]["funding_rate"])
        return dp.fetch_funding_rate(symbol)

    def price_lookup(symbol: str, ts: int) -> float:
        ohlcv = dp.fetch_ohlcv(symbol, config["data"]["timeframe"], since=ts - 3_600_000, limit=2)
        return float(ohlcv.iloc[-1]["close"])

    scheduler = main_module.AlphaLoopScheduler(
        config=config, clock=clock, simulators={"main": sim_main}, trader=trader,
        reflector=reflector, researcher=researcher, memory_store=memory_store, scorer=scorer,
        evolution_orchestrator=evolution_orchestrator, circuit_breaker=circuit_breaker,
        cold_start_gate=cold_start_gate, git_merge_executor=git_merge_executor,
        next_bar_provider=next_bar_provider, snapshot_provider=snapshot_provider,
        funding_rate_lookup=funding_rate_lookup, price_lookup=price_lookup,
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
        # 重新用 COLD_START 产出的真实名单构造 Simulator(替换掉上面 universe_symbols=None 的占位实例)。
        universe_path = PROJECT_ROOT / "universe_active.json"
        symbols = json.loads(universe_path.read_text(encoding="utf-8"))["symbols"]
        scheduler.simulators["main"] = Simulator(
            config=config, circuit_breaker=circuit_breaker, cold_start_gate=cold_start_gate,
            universe_symbols=symbols, db_path=STATE_ROOT / "portfolio_main.db",
            log_root=LOG_ROOT, branch="main", resume=True,
        )

    print("=== 进入常驻调度循环(每分钟轮询一次到期任务,Ctrl+C 停止)===", flush=True)
    last_settlement_check_ms = 0
    while True:
        now = clock.now_ms()
        try:
            result = scheduler.run_decision_cycle("main")
            if result["status"] == "decided":
                print(f"[{now}] 决策周期完成: {[d.action for d in result['decisions']]}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[{now}] run_decision_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)

        if now - last_settlement_check_ms >= 3_600_000:
            try:
                settlement_result = scheduler.run_settlement_catchup()
                if settlement_result["missed_instants"]:
                    print(f"[{now}] 结算补齐: {settlement_result['missed_instants']}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] run_settlement_catchup 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            last_settlement_check_ms = now

        write_heartbeat(clock, {"status": "running"})
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
