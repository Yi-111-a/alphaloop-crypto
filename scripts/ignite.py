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
    - 棘轮判定(EvolutionOrchestrator,LOCKED官方评分):每
      config.cycle.ratchet_interval_hours 小时一次(现为2小时,原为3天)。
      只喂main自己的nav.tsv日粒度权威历史,不喂evo候选分支——见下面"战术
      锦标赛"条目,evo分支的评估已经迁到一套独立机制,不再尝试塞进这个
      本来是为"真实git分支候选"设计的judge()里。
    - 战术锦标赛(用户2026-07-14要求"自动进化+晋升赢家"):与上面的官方
      棘轮判定完全独立的一套并行机制,同样复用 ratchet_interval_hours 节拍
      (见 evaluate_tactic_tournament())。用小时级的 nav_intraday.jsonl /
      nav_intraday_branches.jsonl 数据比较每个active evo分支相对main的
      净值表现:相对main的净值优势 >= tactic_tournament.promote_edge_pct
      → 该分支的战术文字被写入 state/main_program_tactics.json,main分支
      从下一个决策周期起就开始用这个"晋升"上来的战术(不需要重启进程,
      也不走LOCKED区的GitMergeExecutor——这几个分支本来就不是真实git分支,
      只是同一份代码下不同的提示词,没有代码可合并);分支自身净值从峰值
      回撤超过 tactic_tournament.fail_drawdown_pct → 强制平仓退场。两种
      情况都会把该分支从候选名册(state/tactic_tournament_roster.json)
      标记为非active、腾出名额。用户2026-07-14要求把"腾出名额后谁来设计
      替补战术"这一步也自动化(此前是写一个state/new_tactic_request.json
      标记文件,靠签入agent自己想起来去看、再手写一次性scratch脚本填补)——
      现在腾出名额后立刻通过同一条LLM签入通道(见generate_replacement_
      tactic())主动发起一次"设计新战术"请求,校验失败自动重试(最多
      _TACTIC_GENERATION_MAX_RETRIES次),通过后直接写回roster、立即生效,
      全过程记录进LOG/tactic_generations.jsonl。用户明确要求保留"智能来自
      agent"这一层(不换成直连模型API,见AgentBridgeLLMClient),这里自动化
      的只是"要不要问、什么时候问、问完怎么落盘"这些调度性工作,创造性的
      战术设计本身仍然经由LLM桥、由签入的agent(或它调用的子代理)完成,
      与全项目"所有AI操作都由agent发起"的既定原则一致。全部重试都失败时
      (比如agent连续几次给出无法解析的格式),名额保持空缺,不伪造一个
      占位战术填进去——下一次锦标赛节拍(ratchet_interval_hours)会自动
      重试,不需要人工干预去"救回"这个流程。
      没有达到PROMOTE/FAIL任一条件的分支保持active,不产出裁决,可以
      无限期地继续被观察——这是与LOCKED官方judge()"每次调用都对所有
      active分支强制产出终局裁决"最核心的行为差异,由evaluate_tactic_
      tournament()的docstring详细说明。
    - benchmark(BTC_HOLD)通过 LOCKED.baseline_agents.BTCHoldAgent 解析计算
      (不经过Simulator/execute()九步校验链,见该模块docstring的人类裁决)。
    - random对照组(RandomAgent独立分支,不经过Trader/LLM)已按用户要求
      (2026-07-14)下线,不再运行——nav.tsv的nav_random列是LOCKED区
      scorer.daily_mark()的必填参数,这里改成传一个冻结常量(=期初资本)
      占位,不是继续追踪一条真实的随机决策净值轨迹;webui面板侧也已经把
      "随机对照"从图表/分支选择器里去掉。

M8 晋升闸门重构(改造规格书M8,2026-07-15):
    - 诊断(规格书§0.1诊断2):本文件此前调 run_ratchet_judgment 时
      branch_navs 永远只传 {"main": ...},EvolutionOrchestrator.judge() 的
      "候选分支必须同时出现在 branch_navs 和 branch_created_dates 里"这条
      交集过滤使 verdicts 恒为空,GitMergeExecutor 这条生产链路从未被真实
      调用过——锦标赛"晋升"一个evo分支时,晋升的只是一段战术文字(见
      save_main_tactics),不是真正的代码/分支晋升。
    - 修复本体:scheduler.trader 现在不再直接是 ASSET/strategy/trader.py
      的 Trader(纯LLM),而是包一层 ASSET/strategy/policy_trader.py::
      DispatchingTrader——policy_resolver(branch) 能查到 policy_id(名册
      条目/state/main_policy.json)的分支走 M7 确定性代码路径,零LLM;
      查不到的分支(老式纯提示词evo分支、main默认状态)原样走LLM路径,
      main.py.run_decision_cycle 完全不需要感知这层分发,零改动。
    - "两级晋升":战术锦标赛(evaluate_tactic_tournament)的PROMOTE不再对
      有policy_id的分支直接调用save_main_tactics"接管"main的提示词——那
      只是"这个分支有资格去问真正的裁判"这一步,真正的晋升改走
      promote_policy_branch():register_branch(若尚未注册)→ 构造真实的
      日级branch_navs调 scheduler.run_ratchet_judgment()(内部调用
      EvolutionOrchestrator.judge() + GitMergeExecutor.attempt_merge()，
      两者都是LOCKED区已经各自测试过的真实实现,这里第一次让它们在
      "代码化策略分支"上被真实触发)→ merge成功才更新
      state/main_program_tactics.json + state/main_policy.json,main从
      下一个决策周期起自动切换成该policy的纯代码路径,不需要重启进程。
      无policy_id的老式提示词分支维持原有save_main_tactics行为不变。
    - 日级序列(§4.2):锦标赛/末位斩杀的收益率/edge计算不再直接用小时级
      nav_intraday(_branches).jsonl,而是先按UTC日边界降采样(每日取最后
      一个点,见 read_main_daily_nav_series/read_branch_daily_nav_series)。
      信噪比理由:小时级窗口下,正常的净值波动量级(尤其是高杠杆/高频
      分支)经常超过真实优势的量级,窗口太短时棘轮会被噪声填满而不是被
      真实alpha填满——回撤判定(_intraday_max_drawdown_pct)仍然用小时级
      原始序列不变,盘中回撤本来就该看盘中,不应该被日级降采样抹平。

跑法:
    cd alphaloop
    python scripts/ignite.py            # 前台运行,Ctrl+C 停止
    nohup python scripts/ignite.py &     # 后台常驻

本脚本不修改任何 LOCKED 区业务逻辑,只是把已经各自独立测试过的 LOCKED/ASSET
模块按 main.py 的 AlphaLoopScheduler 接口接起来,加一层真实时钟驱动的轮询
循环。
"""
from __future__ import annotations

import copy
import datetime
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml  # noqa: E402

from llm_bridge import AgentBridgeLLMClient  # noqa: E402

from LOCKED import log_writer  # noqa: E402
from LOCKED.baseline_agents import BTCHoldAgent  # noqa: E402
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
from ASSET.strategy.policies import load_policy  # noqa: E402
from ASSET.strategy.policy_trader import DispatchingTrader  # noqa: E402
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


def build_llm_clients(config: dict):
    """返回 (routine_llm, deep_llm) 两个 Callable[[str], str]。

    llm.mode == "bridge"(默认,本地开发/人工介入模式):两个都是同一个
    AgentBridgeLLMClient——签入的 Claude Code agent 通过 state/ 文件握手
    应答,不调用外部API。timeout_seconds 故意给得比决策周期短,超时走
    main.py 既有的失败隔离路径,不让循环无限期卡死。

    llm.mode == "api"(服务器24h无人值守模式,用户2026-07-15要求):直连
    Anthropic API。分两档控制成本——
      routine_llm(便宜模型):例行的30分钟Trader决策周期,量大(7分支x48
        周期/天),大多数输出是hold;
      deep_llm(强模型):反思摘要、每小时研究、锦标赛替补战术设计,量小
        (每天几十次)但真正需要思考质量。
    两档共用同一个每日调用预算文件(见AnthropicLLMClient),超预算当天
    全部降级为安全fallback。"""
    llm_cfg = (config.get("llm", {}) or {})
    mode = llm_cfg.get("mode", "bridge")
    if mode == "bridge":
        bridge = AgentBridgeLLMClient(poll_seconds=2.0, timeout_seconds=1800.0)
        return bridge, bridge
    if mode != "api":
        raise SystemExit(f"config llm.mode 必须是 'bridge' 或 'api',得到: {mode!r}")

    from llm_bridge import AnthropicLLMClient  # noqa: PLC0415

    api_cfg = llm_cfg.get("api", {}) or {}
    max_daily_calls = int(api_cfg.get("max_daily_calls", 600))
    base_url = api_cfg.get("base_url")  # None=官方Anthropic;可指向任何Anthropic格式兼容端点
    routine_llm = AnthropicLLMClient(
        model=api_cfg.get("trader_model", "claude-haiku-4-5-20251001"),
        max_daily_calls=max_daily_calls,
        base_url=base_url,
    )
    deep_llm = AnthropicLLMClient(
        model=api_cfg.get("deep_model", "claude-sonnet-5"),
        max_daily_calls=max_daily_calls,
        base_url=base_url,
    )
    return routine_llm, deep_llm


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
        "last_tournament_ms": None,
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


# ---------------------------------------------------------------------------
# 战术锦标赛("自动进化+晋升赢家",用户2026-07-14要求)——见模块docstring的
# 完整设计说明。核心原则:与 LOCKED.evolution_orchestrator.judge() 完全独立、
# 不共用同一套数据/裁决逻辑,因为 judge() 的"一次性裁决后永久退出候选池"
# 语义(见其docstring步骤5)不适合这里"反复观察、允许长期不产生结论"的
# 场景。这里的裁决结果不会,也不需要,写回 LOCKED 的 branch_registrations/
# ratchet_verdicts 日志。
# ---------------------------------------------------------------------------

TOURNAMENT_ROSTER_PATH = STATE_ROOT / "tactic_tournament_roster.json"
MAIN_TACTICS_PATH = STATE_ROOT / "main_program_tactics.json"
# M8新增(改造规格书M8"两级晋升"):main分支当前生效的M7确定性策略代码
# policy_id——DispatchingTrader 的 policy_resolver 对 "main" 分支查的就是
# 这个文件(见 make_policy_resolver)。与 MAIN_TACTICS_PATH 并存而不是二选
# 一:main_program_tactics.json 保留人类可读的战术描述文字(webui/人工巡检
# 用),main_policy.json 才是决定main分支下一个决策周期是否真的走零LLM代码
# 路径的权威开关。两份状态在 promote_policy_branch() 里总是同一次晋升
# 一起更新,不会出现只更新其中一份的中间态。
MAIN_POLICY_PATH = STATE_ROOT / "main_policy.json"

_DAY_MS = 86_400_000

# 锦标赛生存规则后缀(用户2026-07-14要求:生存后果必须写进每个分支的战术
# 正文,让做决策的Trader/子代理真实看到,不能只留在注释里)。M8起从main()
# 的局部变量提升为模块级常量——admit_policy_to_forward_pool()(模块级函数,
# 供内环/手动调用,不经过main())拼接policy分支的tactics字段时也需要它,
# 两处必须是同一份文字。数值(15%/0.5%)与tactic_tournament配置项保持一致,
# 如果以后改config记得同步改这里。
_TOURNAMENT_STAKES_SUFFIX = (
    "\n\n【锦标赛生存规则,请认真对待,这不是走过场】你正在和另外几个"
    "战术不同的分支实时竞争同一个位置。每隔几小时会用真实净值数据评估"
    "一次:如果你自己的净值从历史峰值回撤超过15%,或任何一笔持仓被爆仓,"
    "你会被立即强制平仓、永久淘汰出局,由全新设计的战术取代你,没有'再给"
    "一次机会'这种说法;如果你的净值持续跑赢main分支0.5个百分点以上,你的"
    "整套战术会被'扶正'成main分支新的打法,相当于赢家通吃、直接接管"
    "主账户的决策权。此外还有末位斩杀(用户2026-07-15新增):每隔约3天,"
    "所有存活满一个考核窗口的分支按窗口内净值收益排名,收益最低的"
    "那一个直接斩杀出局——哪怕它在小幅盈利、哪怕它没犯任何错。这意味着"
    "'平庸'和'亏损'一样致命:长期空仓观望零收益,大概率就是下一个被斩的。"
    "同理,仓位规模必须和信念匹配:target_notional_pct的单位是净值的百分点"
    "(15.0=拿净值的15%建仓),长期只动用个位数percent的资金,收益被锁死在"
    "噪声量级,和空仓等价,一样是斩杀头号候选。本金只有100U,系统允许单仓"
    "名义到净值100%、全组合总名义敞口到300%(即总杠杆3倍)——总敞口长期"
    "连净值的100%都不到,意味着连1倍杠杆都不敢用满,数学上注定跑不出有意义"
    "的收益,也逃不过斩杀。有真实依据就该用杠杆把仓位放大到有意义的量级,"
    "风险控制靠falsifier止损纪律和杠杆选择,不靠把仓位缩到没有意义。"
    "你的每一笔决策都真实关系到这个战术能不能活下去。"
)


def read_main_hourly_nav_series() -> list[tuple[int, float]]:
    """main分支的小时级净值序列,来自本脚本已经在维护的 nav_intraday.jsonl
    (nav_agent字段)——与锦标赛用同一套小时级数据源,口径一致才能公平比较。"""
    records = log_writer.read_jsonl("nav_intraday.jsonl", root=LOG_ROOT)
    return sorted(
        (r["ts"], float(r["nav_agent"])) for r in records if "ts" in r and "nav_agent" in r
    )


def read_branch_hourly_nav_series(branch: str) -> list[tuple[int, float]]:
    records = log_writer.read_jsonl("nav_intraday_branches.jsonl", root=LOG_ROOT)
    return sorted(
        (r["ts"], float(r["nav"])) for r in records if r.get("branch") == branch and "ts" in r and "nav" in r
    )


def _downsample_daily_last(series: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """改造规格书M8§4.2"日级序列"的核心降采样:按UTC自然日边界,每天只保留
    该日内最后一个(ts最大)点,用day_start_ms(该UTC日00:00的毫秒时间戳)
    作为这个点的key,而不是原始ts本身——这样"今天"的这一个点在多次调用
    之间是同一个key,可以被_slice_since/_return_pct这些既有的、按ms比较的
    函数直接消费,不需要改动它们的接口。

    要求输入序列已经按ts升序排列(read_main_hourly_nav_series/
    read_branch_hourly_nav_series 的返回值本身就是sorted()过的)——同一天
    内后出现的点(ts更大)覆盖同一天内先出现的点,直接覆盖写就能保证最终
    保留的是"当日最后一个点",不需要额外排序或比较。
    """
    daily: dict[int, float] = {}
    for ts, nav in series:
        day_start = (ts // _DAY_MS) * _DAY_MS
        daily[day_start] = nav
    return sorted(daily.items())


def read_main_daily_nav_series() -> list[tuple[int, float]]:
    """main分支的日级净值序列:对 read_main_hourly_nav_series() 的结果按
    UTC自然日边界降采样,每天只保留最后一个点。

    改造规格书M8§4.2信噪比修正:原来的锦标赛/末位斩杀直接用小时级序列算
    收益率/edge,窗口拉长之前(min_hours_before_judgment=4,
    cull_interval_hours=12)这个窗口本身就短到"正常的净值波动量级(尤其是
    高杠杆/高频分支)经常超过真实优势的量级"——评估噪声压过信号,棘轮/
    末位斩杀被噪声填满而不是被真实alpha填满。改用日级序列后单点噪声被
    摊薄,窗口拉长(见config.yaml tactic_tournament新数值)后信噪比进一步
    改善。回撤判定(_intraday_max_drawdown_pct)刻意不做这个降采样,仍然吃
    小时级原始序列——盘中回撤本来就该看盘中,不应该被日级降采样抹平掉
    "日内曾经跌穿阈值"这个真实发生过的事实。
    """
    return _downsample_daily_last(read_main_hourly_nav_series())


def read_branch_daily_nav_series(
    branch: str, hourly_lookup: Optional[Callable[[str], list[tuple[int, float]]]] = None
) -> list[tuple[int, float]]:
    """candidate分支版本,同 read_main_daily_nav_series 的降采样逻辑。
    hourly_lookup 可注入(测试用),默认 read_branch_hourly_nav_series。"""
    lookup = hourly_lookup or read_branch_hourly_nav_series
    return _downsample_daily_last(lookup(branch))


def _return_pct(series: list[tuple[int, float]]) -> Optional[float]:
    if len(series) < 2:
        return None
    first, last = series[0][1], series[-1][1]
    if first == 0:
        return None
    return (last - first) / first * 100.0


def _intraday_max_drawdown_pct(series: list[tuple[int, float]]) -> float:
    """相对该序列自身历史滚动高点的最大回撤(不是相对期初资本),与
    LOCKED.scorer._max_drawdown_pct 同样的定义,这里独立实现一份简单版本——
    这是锦标赛自己的淘汰逻辑,不复用LOCKED的评分函数,避免误用一个为
    "日粒度、git分支候选"场景设计的函数到"小时粒度、提示词候选"场景。"""
    if not series:
        return 0.0
    peak = series[0][1]
    max_dd = 0.0
    for _, nav in series:
        peak = max(peak, nav)
        if peak > 0:
            max_dd = max(max_dd, (peak - nav) / peak * 100.0)
    return max_dd


def _slice_since(series: list[tuple[int, float]], since_ms: int) -> list[tuple[int, float]]:
    return [(ts, nav) for ts, nav in series if ts >= since_ms]


def load_tournament_roster(now_ms: int, defaults: dict[str, str]) -> dict:
    """锦标赛的候选名册,持久化到 state/tactic_tournament_roster.json,重启后
    不丢失。首次运行(文件不存在)时用 defaults 里的初始5个战术建档,
    created_ms 记为本次首次建档的时刻——这是一个有意的简化:如果这5个分支
    此前已经通过 EvolutionOrchestrator.register_branch() 注册过(本次改造前
    的做法),它们在 LOCKED 那份注册日志里的创建日期不会跟这里的created_ms
    完全一致,但两者是两套独立的簿记,不需要对齐;这里的created_ms只用于
    锦标赛自己的min_hours_before_judgment门槛计算。"""
    if TOURNAMENT_ROSTER_PATH.exists():
        try:
            return json.loads(TOURNAMENT_ROSTER_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    # 首次建档:必须立刻落盘,不能只留在内存里——本函数在同一次循环里会被
    # 调用多次(决策周期那段、锦标赛评估那段),如果不在这里就写文件,
    # created_ms会在每次调用时都被重新赋成"现在",min_hours_before_judgment
    # 门槛永远也等不到,是刚才第一次重启后真实复现的bug,不是假设性边界情况。
    roster = {
        name: {"tactics": tactics, "status": "active", "created_ms": now_ms}
        for name, tactics in defaults.items()
    }
    save_tournament_roster(roster)
    return roster


def save_tournament_roster(roster: dict) -> None:
    TOURNAMENT_ROSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOURNAMENT_ROSTER_PATH.write_text(json.dumps(roster, ensure_ascii=False), encoding="utf-8")


def load_main_tactics() -> Optional[str]:
    """main分支目前生效的战术文字——默认None(无偏置),一旦锦标赛判定某个
    evo分支PROMOTE,这里会被更新成赢家的战术文字,并持久化到重启后依然生效。"""
    if MAIN_TACTICS_PATH.exists():
        try:
            data = json.loads(MAIN_TACTICS_PATH.read_text(encoding="utf-8"))
            return data.get("tactics")
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_main_tactics(tactics: str, source_branch: str, now_ms: int) -> None:
    MAIN_TACTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAIN_TACTICS_PATH.write_text(
        json.dumps({"tactics": tactics, "promoted_from": source_branch, "ts": now_ms}, ensure_ascii=False),
        encoding="utf-8",
    )


def load_main_policy_id() -> Optional[str]:
    """main分支当前生效的M7确定性策略代码policy_id——默认None(main默认
    走LLM路径,与load_main_tactics()默认None=无战术偏置是同一种"未晋升过
    任何东西"的初始状态)。一旦 promote_policy_branch() 真正把某个policy
    分支合并进main,这里会被更新,且持久化到重启后依然生效。"""
    if MAIN_POLICY_PATH.exists():
        try:
            data = json.loads(MAIN_POLICY_PATH.read_text(encoding="utf-8"))
            return data.get("policy_id")
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_main_policy_id(policy_id: str, source_branch: str, now_ms: int) -> None:
    MAIN_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAIN_POLICY_PATH.write_text(
        json.dumps({"policy_id": policy_id, "promoted_from": source_branch, "ts": now_ms}, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_policy_id_for_branch(branch: str, roster: dict) -> Optional[str]:
    """名册条目可选带 policy_id 字段(M8):有则返回该分支应该运行的
    policy_id,老式纯提示词分支/不在名册里的分支返回None——
    DispatchingTrader 收到None会原样委托LLM路径,老式分支的既有行为完全
    不受影响。纯函数,不碰磁盘,方便独立单测。"""
    meta = roster.get(branch)
    if meta is None:
        return None
    return meta.get("policy_id")


def make_policy_resolver(clock, defaults: dict[str, str]) -> Callable[[str], Optional[str]]:
    """构造一个 branch -> Optional[policy_id] 的可调用对象,供
    ASSET/strategy/policy_trader.py::DispatchingTrader 在每次 decide() 调用
    时查询。

    "main" 分支查 state/main_policy.json(load_main_policy_id);其它分支
    查当前锦标赛名册(每次调用都从磁盘重新读取,不缓存——名册可能在上一次
    锦标赛节拍被改写或有新分支通过 admit_policy_to_forward_pool 加进来,
    这里必须总是看到最新版本,与主循环里"scheduler.program_tactics =
    load_main_tactics()"每轮重读同一个理由)。
    """

    def _resolver(branch: str) -> Optional[str]:
        if branch == "main":
            return load_main_policy_id()
        roster = load_tournament_roster(clock.now_ms(), defaults)
        return resolve_policy_id_for_branch(branch, roster)

    return _resolver


def admit_policy_to_forward_pool(policy_id: str, roster: dict, now_ms: int) -> tuple[dict, str]:
    """把一个M7确定性策略代码(ASSET/strategy/policies/{policy_id}.py)真正
    接入前向锦标赛池(改造规格书M8)。供未来的M6内环(自动生成/迭代policy
    代码之后,把回测里表现最好的一个送进前向验证)调用,也可以在开发/
    验证阶段手动调用。

    做两件事:
      1. 往 roster 里新增一条 {"policy_id":..., "tactics": policy.DESCRIPTION
         + 锦标赛生存规则后缀, "status":"active", "created_ms": now_ms},
         立刻落盘(与 load_tournament_roster 首次建档同一条纪律:不能只留
         在内存里)。tactics字段仍然携带人类可读的战术描述+生存后果
         后缀——不是给 DispatchingTrader 看的(它走 policy_resolver ->
         load_policy 的纯代码路径,压根不读这个字段),而是为了让这个
         policy分支在名册schema上与老式纯提示词分支保持同构(webui展示/
         人工巡检统一读同一个字段)。
      2. 用 subprocess 在仓库里创建一个同名 git 分支(当前HEAD),已存在则
         跳过——这是让 EvolutionOrchestrator.register_branch()/
         GitMergeExecutor.attempt_merge() 后续操作成立的前提:如果这个
         分支名在git里根本不存在,promote_policy_branch() 真正走到
         attempt_merge() 时,会在第一步"分支是否存在"校验上就直接拒绝,
         不管评分多好——这不是本函数负责判定的事,但本函数负责让这个
         前提提前成立。

    返回 (更新后的roster, 新分支名) 二元组。分支名格式
    evo/{YYYYMMDD}-{policy_id},YYYYMMDD 来自 now_ms 对应的 UTC 日期。
    """
    date_compact = datetime.datetime.utcfromtimestamp(now_ms / 1000).strftime("%Y%m%d")
    branch_name = f"evo/{date_compact}-{policy_id}"
    if branch_name in roster:
        raise ValueError(
            f"branch {branch_name!r} already exists in the roster (policy already admitted today?)"
        )

    policy = load_policy(policy_id)
    roster = dict(roster)
    roster[branch_name] = {
        "policy_id": policy_id,
        "tactics": policy.DESCRIPTION + _TOURNAMENT_STAKES_SUFFIX,
        "status": "active",
        "created_ms": now_ms,
    }
    save_tournament_roster(roster)

    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "branch", branch_name],
            capture_output=True, text=True, timeout=30, shell=False,
        )
        if result.returncode != 0 and "already exists" not in (result.stderr or ""):
            print(
                f"admit_policy_to_forward_pool: git branch {branch_name!r} 创建失败"
                f"(非'已存在'原因,已忽略,继续): {result.stderr.strip()}",
                flush=True,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        print(
            f"admit_policy_to_forward_pool: git branch {branch_name!r} 创建异常"
            f"(已忽略,继续): {exc!r}",
            flush=True,
        )

    return roster, branch_name


def promote_policy_branch(
    branch: str,
    policy_id: str,
    roster: dict,
    evolution_orchestrator: EvolutionOrchestrator,
    scheduler,
    today: str,
    daily_branch_lookup: Optional[Callable[[str], list[tuple[int, float]]]] = None,
) -> dict:
    """M8"两级晋升"的核心(修复改造规格书§0.1诊断2)。战术锦标赛
    (evaluate_tactic_tournament)对一个挂着policy_id的分支判定PROMOTE,
    只代表"这个分支有资格去问真正的裁判",不等于真的晋升——这里才是真正
    问裁判、真正合并代码的地方:

      1. register_branch:如果这个分支在_ensure_evo_simulator()把它加进
         名册、构造它自己的Simulator时就已经注册过(主循环里对所有名册
         分支无条件注册,不分是否有policy_id),这里跳过,不重复注册；
         如果尚未注册且orchestrator已达到max_concurrent_branches上限,
         register_branch()返回False——这种情况下"真正的裁决"这一步做不
         了,如实返回一个未合并的结果,不假装晋升成功(锦标赛自己那个
         PROMOTE事件依然会让分支离开锦标赛active池、腾出名额,这是"两级"
         架构里一个诚实的权衡:锦标赛的判定权与EvolutionOrchestrator的
         准入上限是两回事,不因为这里注册失败就回滚锦标赛自己的判定)。
      2. 构造真实的日级 branch_navs——main用 LOG/nav.tsv 的官方日级历史
         (scorer.daily_mark()维护的权威三线序列,长期存在、覆盖这个候选
         分支整个生命周期),candidate分支用
         read_branch_daily_nav_series(branch)(从nav_intraday_branches.jsonl
         降采样,转成与nav.tsv同构的ISO日期字符串key)——调用
         scheduler.run_ratchet_judgment(),这是"侵入最小"的选择:该方法
         已经完整实现了"调judge() -> PROMOTE时调
         git_merge_executor.attempt_merge() -> merge成功才推进
         effective_main_branch"这一整套逻辑并且已经被
         tests/test_main_scheduler.py验证过,这里不需要、也不应该绕开它
         直接调 orchestrator.judge()/git_merge_executor.attempt_merge()。
      3. 判定的verdict如果不是PROMOTE(比如日级窗口下edge不够、或触发了
         死刑回撤条款),真正的裁判否决了锦标赛自己的PROMOTE判断——这是
         完全合法的结果,不触碰main任何状态。
      4. verdict是PROMOTE但GitMergeExecutor拒绝了真实merge(测试红/分支
         非法)——scheduler.run_ratchet_judgment() 内部已经把这种情况记成
         severity=critical的promotion_veto,effective_main_branch保持
         不变,这里只需要读 scheduler.effective_main_branch 是否等于
         branch 就能判断merge到底有没有真的发生。
      5. merge成功 -> 用该policy的DESCRIPTION更新
         state/main_program_tactics.json(human/webui可读的文字描述)+
         state/main_policy.json(DispatchingTrader.policy_resolver对"main"
         分支实际查的文件)。main从下一个决策周期起自动切换成该policy的
         纯代码路径,不需要重启进程。

    返回一个summary dict(至少含"summary"这个可打印的人类可读字符串),
    调用方(主循环)只需要打印,不需要进一步处理。
    """
    if evolution_orchestrator.branch_meta(branch) is None:
        created_date = roster[branch].get("created_date") or _utc_date_str(roster[branch]["created_ms"])
        if not evolution_orchestrator.register_branch(branch, created_date):
            return {
                "registered": False,
                "verdict": None,
                "merged": False,
                "summary": (
                    f"evolution_orchestrator at max_concurrent_branches capacity, cannot "
                    f"register {branch!r} -- real code-level judgment/merge did not run this "
                    f"round (tournament's own PROMOTE nomination still stands and the branch "
                    f"still vacates its tournament slot)"
                ),
            }

    branch_lookup = daily_branch_lookup or read_branch_daily_nav_series
    nav_series = read_nav_tsv_series()  # 官方日级三线历史(main/benchmark),权威来源
    branch_date_series = [(_utc_date_str(ts), nav) for ts, nav in branch_lookup(branch)]

    if len(nav_series["main"]) < 2 or len(branch_date_series) < 2:
        return {
            "registered": True,
            "verdict": None,
            "merged": False,
            "summary": (
                "insufficient daily NAV history for main or branch -- skipping the real "
                "code-level judgment this round, will retry next tournament tick"
            ),
        }

    verdicts = scheduler.run_ratchet_judgment(
        now_date=today,
        branch_navs={"main": nav_series["main"], branch: branch_date_series},
        benchmark_navs=nav_series["benchmark"],
    )
    verdict = verdicts.get(branch)
    if verdict is None:
        return {
            "registered": True,
            "verdict": None,
            "merged": False,
            "summary": "evolution_orchestrator.judge() produced no verdict for this branch this round",
        }

    if verdict.decision != "PROMOTE":
        return {
            "registered": True,
            "verdict": verdict.decision,
            "merged": False,
            "summary": (
                f"real judge() verdict={verdict.decision!r} (reason={verdict.reason!r}) -- the "
                f"tournament's own PROMOTE nomination was overridden by the official gate"
            ),
        }

    if scheduler.effective_main_branch != branch:
        return {
            "registered": True,
            "verdict": "PROMOTE",
            "merged": False,
            "summary": (
                "judge()==PROMOTE but GitMergeExecutor vetoed the real git merge (see "
                "LOG/scheduler_errors.jsonl event=promotion_veto for the reason)"
            ),
        }

    policy = load_policy(policy_id)
    save_main_tactics(policy.DESCRIPTION, branch, scheduler.clock.now_ms())
    save_main_policy_id(policy_id, branch, scheduler.clock.now_ms())
    return {
        "registered": True,
        "verdict": "PROMOTE",
        "merged": True,
        "summary": f"merged into main -- main now runs policy_id={policy_id!r} (no restart needed)",
    }


def evaluate_tactic_tournament(
    roster: dict,
    now_ms: int,
    tournament_cfg: dict,
    daily_main_lookup: Optional[Callable[[], list[tuple[int, float]]]] = None,
    daily_branch_lookup: Optional[Callable[[str], list[tuple[int, float]]]] = None,
    hourly_branch_lookup: Optional[Callable[[str], list[tuple[int, float]]]] = None,
) -> list[dict]:
    """对名册里每个status=='active'的分支做一次评估,返回本次新产生的裁决
    事件列表(可能是空列表——这是正常情况,大多数分支大多数时候应该处于
    "还没有明确结论,继续观察"这个中间状态,不是每次调用都必须产生裁决,
    这正是与LOCKED.judge()"每次调用都对所有active分支强制产出裁决"最核心
    的行为差异)。不修改roster的落盘,调用方负责在收到非空事件列表后落盘。

    改造规格书M8§4.2:收益率/edge的分母改用日级序列(daily_main_lookup/
    daily_branch_lookup,默认 read_main_daily_nav_series/
    read_branch_daily_nav_series)——信噪比理由见这两个函数的docstring。
    回撤判定(_intraday_max_drawdown_pct)刻意维持小时级序列不变
    (hourly_branch_lookup,默认read_branch_hourly_nav_series)——盘中回撤
    本来就该看盘中,不应该被日级降采样抹平掉"日内曾经跌穿阈值"这个事实。
    三个lookup均可注入,供测试构造合成日级/小时级数据。"""
    min_age_ms = int(float(tournament_cfg.get("min_hours_before_judgment", 4)) * 3_600_000)
    promote_edge = float(tournament_cfg.get("promote_edge_pct", 0.5))
    fail_dd = float(tournament_cfg.get("fail_drawdown_pct", 15))

    main_daily_lookup = daily_main_lookup or read_main_daily_nav_series
    branch_daily_lookup = daily_branch_lookup or read_branch_daily_nav_series
    branch_hourly_lookup = hourly_branch_lookup or read_branch_hourly_nav_series

    main_daily_series = main_daily_lookup()
    events: list[dict] = []

    for branch, meta in roster.items():
        if meta.get("status") != "active":
            continue
        created_ms = meta.get("created_ms", now_ms)
        if now_ms - created_ms < min_age_ms:
            continue

        branch_daily_series = _slice_since(branch_daily_lookup(branch), created_ms)
        if len(branch_daily_series) < 2:
            continue
        branch_return = _return_pct(branch_daily_series)
        if branch_return is None:
            continue

        main_window = _slice_since(main_daily_series, created_ms)
        main_return = _return_pct(main_window)
        if main_return is None:
            continue

        edge = branch_return - main_return
        # 回撤仍然看小时级原始序列,不是上面用于算收益的日级降采样序列——
        # 两者衡量的是不同的东西(收益率关心"起点到终点",回撤关心"中途
        # 曾经跌到多深"),日级降采样会把日内的最深回撤点抹掉。
        branch_hourly_window = _slice_since(branch_hourly_lookup(branch), created_ms)
        drawdown = _intraday_max_drawdown_pct(branch_hourly_window)

        if drawdown > fail_dd:
            events.append({
                "branch": branch, "decision": "FAIL", "edge_vs_main_pct": edge,
                "max_drawdown_pct": drawdown,
                "reason": f"intraday drawdown {drawdown:.2f}% > fail_drawdown_pct {fail_dd:.2f}%",
            })
        elif edge >= promote_edge:
            events.append({
                "branch": branch, "decision": "PROMOTE", "edge_vs_main_pct": edge,
                "max_drawdown_pct": drawdown,
                "reason": f"edge_vs_main {edge:.2f}% >= promote_edge_pct {promote_edge:.2f}%",
            })
        # 否则:既没有触发死刑条款也没有达到晋升门槛 -> 保持active,不产出
        # 裁决,下一次锦标赛节拍再评估——这是有意的"允许长期没有结论"。

    return events


def evaluate_cull(
    roster: dict, now_ms: int, tournament_cfg: dict, nav_series_lookup=None
) -> Optional[dict]:
    """末位斩杀(用户2026-07-15要求):每 cull_interval_hours 一次,把所有
    "存活满一个考核窗口"的active分支按窗口内净值收益排名,收益最低的那个
    直接判CULLED——哪怕它在盈利。与 evaluate_tactic_tournament 的两条绝对
    红线(回撤15%/晋升0.5%)互补:那两条管的是"犯错"和"卓越",这条管的是
    "平庸"——保证种群哪怕在风平浪静的行情里也持续换血。

    保护规则(防止斩杀机制退化成斩杀噪声/斩杀新人):
      - 不满 cull_min_age_hours 的新分支不参与排名(否则新补位分支带着
        全新空仓账户必然垫底,进化变成死循环);
      - 可排名的分支不足2个时不斩(排名无意义);
      - main分支不参与(它是擂主,由晋升机制换血,不由斩杀机制清退)。

    返回一个与 evaluate_tactic_tournament 事件同构的dict(decision='CULLED'),
    或 None(本轮没有可斩对象)。调用方负责节拍控制(cull_interval_hours)
    和后续处理(强平/落盘/生成替补)。nav_series_lookup 可注入,测试用,
    默认改造规格书M8§4.2的日级序列(read_branch_daily_nav_series)——同
    evaluate_tactic_tournament的信噪比理由,末位斩杀比的是"窗口内收益排名"
    而不是"窗口内曾经跌到多深"(那是evaluate_tactic_tournament的死刑回撤
    条款负责的事),所以这里也改用日级、不需要额外的小时级回撤分支。"""
    interval_h = float(tournament_cfg.get("cull_interval_hours", 12))
    min_age_ms = int(float(tournament_cfg.get("cull_min_age_hours", interval_h)) * 3_600_000)
    window_ms = int(interval_h * 3_600_000)
    if nav_series_lookup is None:
        nav_series_lookup = read_branch_daily_nav_series

    candidates: list[tuple[str, float]] = []
    for branch, meta in roster.items():
        if meta.get("status") != "active":
            continue
        if now_ms - meta.get("created_ms", now_ms) < min_age_ms:
            continue
        series = _slice_since(nav_series_lookup(branch), now_ms - window_ms)
        if len(series) < 2:
            continue
        ret = _return_pct(series)
        if ret is None:
            continue
        candidates.append((branch, ret))

    if len(candidates) < 2:
        return None

    worst_branch, worst_ret = min(candidates, key=lambda pair: pair[1])
    return {
        "branch": worst_branch,
        "decision": "CULLED",
        "edge_vs_main_pct": worst_ret,
        "max_drawdown_pct": 0.0,
        "reason": (
            f"cull: lowest return {worst_ret:.2f}% over last {interval_h:.0f}h "
            f"among {len(candidates)} eligible branches (末位斩杀,与对错无关)"
        ),
    }


_TACTIC_GENERATION_MAX_RETRIES = 3
_MIN_GENERATED_TACTICS_LEN = 20


def _build_tactic_generation_prompt(
    event: dict, roster: dict, universe_symbols: list[str], retry_feedback: Optional[str] = None
) -> str:
    """用户要求(2026-07-14)补上锦标赛闭环里唯一还依赖人工的一步:分支被
    淘汰/晋升腾出名额后,不再只是写一面旗子(state/new_tactic_request.json)
    等签入agent自己想起来去手写一个scratch脚本,而是立刻通过同一条LLM签入
    通道主动发起一次"设计新战术"的请求——用户明确选择保留"智能来自agent"
    这一层(不换成直连模型API),这里只是把"要不要问、什么时候问、问完怎么
    落盘"这些调度性工作自动化,创造性的战术设计本身仍然经由LLM桥完成。"""
    active_tactics_lines = [
        f"- {b}: {m['tactics'][:200]}..."
        for b, m in roster.items()
        if m.get("status") == "active"
    ]
    lines = [
        "You are designing ONE new candidate trading-tactic branch for AlphaLoop-Crypto's "
        "tactic tournament (paper-trading, no real money). A slot just opened because an "
        "existing branch was resolved:",
        f"  resolved_branch={event['branch']!r}, decision={event['decision']!r}, "
        f"edge_vs_main_pct={event['edge_vs_main_pct']:.2f}%, max_drawdown_pct={event['max_drawdown_pct']:.2f}%, "
        f"reason={event['reason']!r}",
        "",
        "Currently active tactics in the tournament (design something genuinely DIFFERENT "
        "from all of these -- not a minor variation, a distinct trading idea):",
        "\n".join(active_tactics_lines) if active_tactics_lines else "(none currently active)",
        "",
        f"Tradeable universe this cycle: {universe_symbols}",
        "",
        "Respond with ONLY a JSON object (no markdown fences, no prose outside the JSON): "
        '{"branch_id": "evo/YYYYMMDD-<short-english-slug>", "tactics": "<tactic description in '
        "Chinese, 2-5 sentences, genuinely differentiated from the active list above, grounded in "
        "a concrete trading idea (momentum / funding-rate carry / mean-reversion / volatility-"
        'targeting / event-driven / cross-asset correlation / etc, not a vague restatement)>"}. '
        "Do NOT write any survival-stakes/tournament-rules text yourself -- the system appends "
        "that automatically to every branch's tactics. branch_id must start with 'evo/' and use "
        "today's UTC date.",
    ]
    if retry_feedback:
        lines.append("")
        lines.append(
            f"Your previous response failed validation: {retry_feedback}. "
            "Fix the issue and resubmit strictly as a JSON object."
        )
    return "\n".join(lines)


def _validate_tactic_generation_response(raw: str, roster: dict) -> tuple[Optional[dict], Optional[str]]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "response_not_valid_json"
    if not isinstance(parsed, dict):
        return None, "response_not_a_json_object"

    branch_id = parsed.get("branch_id")
    if not isinstance(branch_id, str) or not branch_id.strip().startswith("evo/") or len(branch_id.strip()) < len("evo/x"):
        return None, "branch_id_invalid: must be a non-empty string starting with 'evo/'"
    branch_id = branch_id.strip()
    if branch_id in roster:
        return None, f"branch_id_collision: {branch_id!r} already exists in the roster, pick a different slug"

    tactics = parsed.get("tactics")
    if not isinstance(tactics, str) or len(tactics.strip()) < _MIN_GENERATED_TACTICS_LEN:
        return None, f"tactics_invalid: must be a string with len>={_MIN_GENERATED_TACTICS_LEN} after strip"

    return {"branch_id": branch_id, "tactics": tactics.strip()}, None


def generate_replacement_tactic(
    llm_client, event: dict, roster: dict, universe_symbols: list[str],
    max_retries: int = _TACTIC_GENERATION_MAX_RETRIES,
) -> Optional[dict]:
    """向签入的LLM/agent请求为刚腾出的名额设计一个全新战术,校验失败重试,
    全部失败则返回None——调用方应该跳过这次自动生成、把名额留到下一次锦标赛
    节拍再重试,而不是伪造一个占位战术填进去:一个"看起来合法但没有真实
    差异化思考"的战术会污染锦标赛的比较意义,比"暂时空一个名额"更糟。"""
    retry_feedback: Optional[str] = None
    for _attempt in range(max_retries):
        prompt = _build_tactic_generation_prompt(event, roster, universe_symbols, retry_feedback)
        raw = llm_client(prompt)
        result, error = _validate_tactic_generation_response(raw, roster)
        if error is None:
            return result
        retry_feedback = error
    return None


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
    # routine_llm:量大的例行决策周期;deep_llm:反思/研究/战术设计。
    # bridge模式下两者是同一个对象,api模式下分别对应便宜/强两档模型。
    routine_llm, deep_llm = build_llm_clients(config)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    # 启动即写心跳(2026-07-15修复的真实bug):此前只在每轮循环结尾写,
    # 思考模式API下一轮循环要十几分钟,清零后watchdog以"心跳文件不存在"
    # 为由每5分钟杀一次进程,进程永远活不到第一次写心跳,无限重启循环。
    write_heartbeat(clock, {"status": "starting"})

    dp = DataPipeline(exchange_id=config["data"]["exchange"], clock=clock)
    uf = UniverseFilter(config, clock=clock)
    memory_store = MemoryStore(db_path=PROJECT_ROOT / "ASSET" / "memory" / "memory.db")
    # 真实外部检索(2026-07-15,用户选定Tavily):TAVILY_API_KEY 存在时研究
    # 周期带真实新闻/宏观检索(每小时1次≈720次/月,落在免费层1000次内,
    # 客户端自带950次/月保险丝);没有key时 search_client=None,Researcher
    # 走既有的"无搜索源"降级路径,行为与此前完全一致。
    _tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if _tavily_key:
        from search_client import TavilySearchClient  # noqa: PLC0415
        search_client = TavilySearchClient(api_key=_tavily_key)
        print("Tavily搜索已启用(月度保险丝950次)", flush=True)
    else:
        search_client = None
    researcher = Researcher(
        llm_client=deep_llm, memory_store=memory_store, genesis_path=GENESIS_PATH,
        search_client=search_client,
    )
    trader = Trader(
        llm_client=routine_llm,
        memory_store=memory_store,
        max_leverage=int((config.get("leverage", {}) or {}).get("max", 10)),
    )
    reflector = Reflector(llm_client=deep_llm, memory_store=memory_store, log_root=LOG_ROOT)
    scorer = Scorer(config, log_root=LOG_ROOT)
    cold_start_gate = ColdStartGate(genesis_path=GENESIS_PATH, min_hypothesis_count=10, log_root=LOG_ROOT)
    circuit_breaker = CircuitBreaker(config, log_root=LOG_ROOT)
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

    # 趋势上下文缓存:{symbol: (computed_at_ms, chg_24h_pct, chg_7d_pct)}。
    # 2026-07-15用户指出的真实盲区:API模式下Trader是无状态的,快照里只有
    # 当前价,模型根本不知道"LAB刚跌了26%"、更看不到"全场都在跌"的大势,
    # 于是全员逆势做多。这里给每个symbol补上24h/7d涨跌幅(从4h K线算,
    # 24h=6根、7d=42根),trader._format_prompt再基于这些字段生成大势
    # 判断行。10分钟memo缓存:snapshot_provider每轮决策会被7个分支各调
    # 一次,不缓存的话13个币x7次=91次K线拉取/30分钟,纯浪费。
    _trend_cache: dict = {}
    _TREND_CACHE_MS = 10 * 60_000

    def _trend_fields(symbol: str, now_ms: int) -> dict:
        cached = _trend_cache.get(symbol)
        if cached and now_ms - cached[0] < _TREND_CACHE_MS:
            return {"chg_24h_pct": cached[1], "chg_7d_pct": cached[2]}
        try:
            bars = dp.fetch_ohlcv(symbol, "4h", limit=43)
            closes = bars["close"]
            last = float(closes.iloc[-1])
            chg24 = round((last / float(closes.iloc[-7]) - 1) * 100, 2) if len(closes) >= 7 else None
            chg7d = round((last / float(closes.iloc[-43]) - 1) * 100, 2) if len(closes) >= 43 else None
            _trend_cache[symbol] = (now_ms, chg24, chg7d)
            return {"chg_24h_pct": chg24, "chg_7d_pct": chg7d}
        except Exception:  # noqa: BLE001 -- 趋势字段是增强信息,拉不到不该毁掉整个快照
            return {}

    def snapshot_provider(ts: int) -> dict:
        symbols_now = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))["symbols"] if UNIVERSE_PATH.exists() else []
        if not symbols_now:
            return {}
        snap = dp.fetch_latest_snapshot(symbols_now)
        for symbol, ticker in snap.items():
            if isinstance(ticker, dict):
                ticker.update(_trend_fields(symbol, ts))
        return snap

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
        benchmark_nav_provider=benchmark_nav_provider,  # random_nav_provider 已按用户要求(2026-07-14)不再注入,random对照组下线
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

    # 用户要求的多分支并行 + 战术锦标赛自动进化(2026-07-14,快速验证阶段,
    # random对照组同时下线):每个候选分支各自独立仓位、各自走完整的
    # Trader/LLM 签入决策周期。战术差异通过 scheduler.program_tactics 这个
    # 既有的公开属性在每次调用前手动切换实现(main.py 的
    # _call_trader_with_timeout 本来就会把它转发给 Trader.decide(),见
    # main.py:344 —— 这里只是从 ignite.py 侧按分支复用同一个已有机制,不需要
    # 改 main.py 的接口)。分支名遵循 LOCKED 区"evo/YYYYMMDD-简述"命名约定。
    # 名册现在持久化在 state/tactic_tournament_roster.json(见
    # load_tournament_roster),不再是这里的固定字典——锦标赛判定PROMOTE/
    # FAIL后会把对应分支标记为非active、腾出名额,由签入的agent(我)之后
    # 设计新战术填补空缺,详见下面主循环里"战术锦标赛"那一段。这里只是
    # 提供"如果名册文件还不存在,首次建档用什么初始战术"的默认值。
    # 用户明确要求(2026-07-14):战术文字本身必须写清楚锦标赛的生存后果
    # ("投资差的代理就会死"),不能只在ignite.py/config.yaml的注释里描述
    # 规则——如果实际做决策的Trader/子代理看不到这段文字,它就不知道自己
    # 真的在被淘汰机制盯着,不会产生"赢家通吃"该有的真实竞争压力。这段
    # 后缀直接写进每个分支的战术正文,拼进program_tactics,每次决策请求都
    # 会带给Trader/子代理看到。数值(15%/0.5%)与tactic_tournament配置项
    # 保持一致,如果以后改config记得同步改这里。
    LIANGXI_BRANCH = "evo/20260714-liangxi-style"

    _DEFAULT_EVO_TACTICS = {
        "evo/20260714-aggressive": (
            "进取战术:在假设置信度足够(至少两条独立假设互相印证,或有真实"
            "价格/资金费率数据强支撑)时,愿意用更高杠杆、更集中的仓位捕捉"
            "机会,不要求像main分支那样保守分散;但仍必须为每笔非hold决策"
            "写出明确的falsifier_condition并严格执行,进取不等于不设止损。"
            + _TOURNAMENT_STAKES_SUFFIX
        ),
        "evo/20260714-conservative": (
            "保守战术:只在多条独立假设互相印证、且没有明显反向风险信号时"
            "才建仓;仓位规模、杠杆都应明显低于main分支的对应决策,宁可错过"
            "一部分机会也不放大不确定性下的敞口;对新标的(缺乏完整2年历史"
            "或资金费率样本不足96天)一律用最小仓位或直接观望。"
            + _TOURNAMENT_STAKES_SUFFIX
        ),
        "evo/20260714-momentum": (
            "动量战术:优先关注最近若干个决策周期内价格单方向变动幅度最大的"
            "标的,顺势而非逆势——如果某标的近期出现明显的单边趋势(结合"
            "genesis.md记录的趋势/波动率画像),倾向于跟随该趋势方向建仓,"
            "而不是像main分支那样只押H1的BTC低波动锚定逻辑;必须为每笔"
            "决策写清楚'趋势可能反转'的falsifier_condition,顺势不等于追高。"
            + _TOURNAMENT_STAKES_SUFFIX
        ),
        "evo/20260714-carry": (
            "资金费率carry战术:优先寻找资金费率长期偏离零、且价格没有极端"
            "单边趋势的标的做反向持仓吃资金费率(如H3描述的BEAT多头付费"
            "场景);对资金费率样本不足96天或标准差明显偏高(如H4描述的LAB)"
            "的标的保持谨慎小仓位;这个分支的核心假设是carry收益本身、不是"
            "赌方向,仓位方向应该跟资金费率符号相反(资金费率为正->偏空吃"
            "carry,为负->需要额外警惕H2描述的挤压反弹风险)。"
            + _TOURNAMENT_STAKES_SUFFIX
        ),
        "evo/20260714-diversified": (
            "分散战术:每个决策周期尽量在universe内多个不相关标的上分别建立"
            "小仓位,而不是像main分支那样长期只集中在BTC一个标的上;单个"
            "标的的仓位规模应该明显小于main/aggressive分支的对应决策,用"
            "'广撒网、小额验证多个假设'的方式积累各个标的的真实表现数据,"
            "为后续Reflector反思和棘轮判定提供更丰富的样本。"
            + _TOURNAMENT_STAKES_SUFFIX
        ),
        # 用户2026-07-14要求新增,明确点名模仿"凉兮"——真实报道的币圈交易员
        # (知乎、非小号等多方独立报道):2021年5月用父亲银行卡里1000元本金、
        # 100倍杠杆做空BTC,"滚仓"策略平均每5分钟操作一次、单周交易1454次,
        # 一个月内做到数千万,巅峰资产超4000万;但2021下半年BTC反弹,他继续
        # 用100倍杠杆判断失误,资产从4000多万迅速缩水、最终倒欠数百万。这是
        # 一个有据可查的真实"高杠杆滚仓最终爆仓"案例,不是虚构的稳健策略——
        # 用户在得知这个真实结局后,明确选择保留这个教训、仍然要求加入这个
        # 风格作为候选分支(而不是要求我发明一个更保守的替代版本)。
        # 用户2026-07-14二次反馈:实盘观察到第一版战术文字虽然提了"可以用
        # 远高于其他分支的杠杆",但从没明确要求仓位规模也同样激进,导致
        # 实际决策出现"杠杆写得很高、但只压极小一部分保证金"的自相矛盾
        # 组合(首笔交易两个仓位合计保证金只占总资金约1.2%)——用户原话
        # "感觉凉兮这个不敢梭哈啊,不像本尊的风格"。改写为明确要求仓位
        # 集中度必须和杠杆倍数匹配,"梭哈"式集中下注才是这个人设的核心,
        # 不是单纯堆杠杆数字;止损纪律(falsifier_condition)保持不变,这是
        # 用户明确认可保留的、唯一刻意偏离真实凉兮的地方。
        LIANGXI_BRANCH: (
            "高频高杠杆滚仓战术(模仿公开报道的'凉兮'风格):高杠杆的数字本身"
            "不是重点,真正的核心是'梭哈'式的仓位集中度——当方向性判断成型"
            "时,应该把可用保证金的绝大部分甚至接近全部压在这一个方向上,"
            "而不是像其他分支那样为了分散风险把小额仓位铺在多个标的上;"
            "宁可只做一个高确信度的方向,也不要为了'看起来更稳健'而故意"
            "缩小仓位规模——缩小仓位规模是conservative分支该做的事,不是"
            "这个人设该有的行为。杠杆可以远高于其他分支(系统硬上限已提到"
            "100倍,真实报道的凉兮本人就是常年使用100倍杠杆),仓位规模也"
            "应该同样激进、与杠杆倍数相匹配,不能出现'杠杆很高但只压极小"
            "一部分保证金'这种自相矛盾的组合。同时高频操作,尽量每个决策"
            "周期都重新评估仓位;方向性博弈优先于传统技术分析,震荡行情下"
            "可以同时在同一标的开多空双向仓位('多空双撸')捕捉双向波动。"
            "但必须吸取凉兮真实爆仓的教训:2021年他在BTC反转后仍然死扛"
            "100倍杠杆的方向性判断、拒绝止损,最终从4000多万倒欠数百万——"
            "本分支每笔非hold决策仍然必须写出真实的falsifier_condition并"
            "严格执行,不能因为'全仓高杠杆'就省略止损纪律;一旦某个方向的"
            "判断被falsifier_condition证伪,应该立即反手或平仓,而不是像"
            "凉兮当年那样加仓死扛。仓位集中+严格止损两者同时成立,才是这个"
            "分支真正要验证的假设。"
            + _TOURNAMENT_STAKES_SUFFIX
        ),
    }

    # M8晋升闸门重构:scheduler.trader 从这里起不再直接是上面构造的裸
    # Trader(纯LLM)——包一层 DispatchingTrader,policy_resolver 能查到
    # policy_id 的分支(main通过state/main_policy.json,evo分支通过当前
    # 名册)走M7确定性代码路径,零LLM;查不到的分支(老式纯提示词分支、
    # 尚未被任何policy接管的main)原样委托给原始trader,现有LLM路径行为
    # 完全不变。main.py.AlphaLoopScheduler._call_trader_with_timeout()
    # 只认 self.trader.decide(...) 这一个签名,不需要知道、也不需要改动
    # 来适配这层分发——这正是main.py"每个外部依赖都通过构造函数注入"这条
    # 既定纪律的自然延伸。放在这里(_DEFAULT_EVO_TACTICS 定义之后)而不是
    # 紧跟着 scheduler 构造的地方,是因为 make_policy_resolver 需要
    # _DEFAULT_EVO_TACTICS 作为"名册文件不存在时的首次建档默认值"——两者
    # 必须是同一份 defaults,否则 policy_resolver 在名册还没被主循环第一次
    # 写盘之前查到的可能是一份不同的初始名册。
    scheduler.trader = DispatchingTrader(
        llm_trader=trader,
        policy_resolver=make_policy_resolver(clock, _DEFAULT_EVO_TACTICS),
        data_pipeline=dp,
        memory_store=memory_store,
        timeframe=config["data"]["timeframe"],
    )

    tournament_roster = load_tournament_roster(clock.now_ms(), _DEFAULT_EVO_TACTICS)
    evo_simulators: dict[str, Simulator] = {}

    def _ensure_evo_simulator(evo_branch: str) -> Simulator:
        """按需为名册里的分支(不论active/promoted/failed)构造/复用一个
        Simulator——promoted/failed分支也需要能读回它们的历史持仓(比如生成
        强制平仓决策、或面板上还能看到它们最后的仓位状态),不是构造完就
        不再关心了。"""
        if evo_branch in evo_simulators:
            return evo_simulators[evo_branch]
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
        return evo_sim

    for evo_branch in tournament_roster:
        _ensure_evo_simulator(evo_branch)

    # 凉兮分支曾经有一个专属5分钟高频scheduler(2026-07-14加的)——用户
    # 2026-07-15决定下线:24h无人值守API模式下,统一所有分支为30分钟节拍,
    # 控制调用成本、简化调度结构。凉兮的"人设"(高杠杆、滚仓、确认即加码)
    # 完整保留在它的战术文字里,与调用频率无关。

    reflection_interval_ms = int((24 * 3_600_000) // max(1, int(config["cycle"]["reflection_per_day"])))
    ratchet_interval_ms = int(float(config["cycle"]["ratchet_interval_hours"]) * 3_600_000)
    research_interval_ms = int(float(config["cycle"]["research_interval_hours"]) * 3_600_000)

    print("=== 进入常驻调度循环(每分钟轮询一次到期任务,Ctrl+C 停止)===", flush=True)
    last_settlement_check_ms = 0
    last_risk_check_ms = 0
    last_mark_ms = 0
    last_research_ms = 0
    risk_check_interval_ms = scheduler._risk_check_interval_hours * 3_600_000
    # 用户要求(2026-07-14)面板"未实现盈亏"更实时:标记价/浮盈浮亏的刷新
    # 节拍从原来跟哨兵共用的risk_check_interval_ms拆出来,独立走
    # risk_check.mark_interval_minutes(默认5分钟),不影响哨兵判定本身
    # 仍然是每小时一次。
    mark_interval_ms = int(float((config.get("risk_check", {}) or {}).get("mark_interval_minutes", 5)) * 60_000)
    while True:
        now = clock.now_ms()
        try:
            # main 分支的战术偏置默认是None(无偏置),但战术锦标赛一旦判定
            # 某个evo分支PROMOTE,这里会改成从state/main_program_tactics.json
            # 读回的赢家战术文字——每次循环都重新读一次文件(而不是缓存在
            # 内存变量里),这样锦标赛在本轮循环后面判定出PROMOTE时,下一轮
            # main决策就能立刻用上,不需要重启进程。
            scheduler.program_tactics = load_main_tactics()
            result = scheduler.run_decision_cycle("main")
            if result["status"] == "decided":
                print(f"[{now}] 决策周期完成: {[d.action for d in result['decisions']]}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[{now}] run_decision_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)

        # 多分支并行:每个 evo 候选分支各自走一遍完整的 Trader/LLM 签入决策
        # 周期,战术差异通过临时切换 scheduler.program_tactics 实现(见上面
        # _DEFAULT_EVO_TACTICS 定义处的说明)。每轮循环都重新从磁盘加载名册
        # (而不是只在main()启动时读一次),这样我(签入agent)在名册文件里
        # 添加一个新战术、或战术锦标赛把某个分支标记为非active之后,不需要
        # 重启ignite.py进程就能生效——只处理status=='active'的分支,循环
        # 结束后不需要额外重置program_tactics,下一轮while顶部main分支调用
        # 前已经会重新读一次。
        tournament_roster = load_tournament_roster(now, _DEFAULT_EVO_TACTICS)
        for evo_branch, meta in tournament_roster.items():
            if meta.get("status") != "active":
                continue
            _ensure_evo_simulator(evo_branch)
            try:
                scheduler.program_tactics = meta["tactics"]
                evo_result = scheduler.run_decision_cycle(evo_branch)
                if evo_result["status"] == "decided":
                    print(f"[{now}] [{evo_branch}] 决策周期完成: {[d.action for d in evo_result['decisions']]}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] [{evo_branch}] run_decision_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            # 每处理完一个分支就刷新心跳:思考模式API下单次决策可达1-3分钟,
            # 一轮7个分支远超watchdog的stale阈值,只在循环结尾写会被误杀。
            write_heartbeat(clock, {"status": "running"})

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
        # 故意只对main分支做,不覆盖evo候选分支——每个分支自己的风险偏好
        # 本身就是要被比较的对象之一,统一加急停保护会削弱这种比较意义。
        if now - last_risk_check_ms >= risk_check_interval_ms:
            try:
                risk_result = scheduler.run_risk_check_cycle("main")
                if risk_result.get("triggered"):
                    print(f"[{now}] 紧急风控平仓触发: {risk_result['triggered']}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] run_risk_check_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            last_risk_check_ms = now

        # 用户要求(2026-07-14):面板"未实现盈亏"要更实时,不要跟风控哨兵
        # 共用60分钟节拍——这里独立走 risk_check.mark_interval_minutes
        # (默认5分钟)。同时也顺手记一笔净值曲线用的小时级(现在其实是
        # 5分钟级)历史点:不改scorer.daily_mark()/nav.tsv本身(那条线是
        # LOCKED棘轮判定用的权威日线历史,改它的语义有风险),而是另开一份
        # 独立的、纯附加的 LOG/nav_intraday.jsonl。
        if now - last_mark_ms >= mark_interval_ms:
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
                nav_benchmark_now = benchmark_nav_provider(now)
                log_writer.append_jsonl(
                    "nav_intraday.jsonl",
                    {
                        "ts": now,
                        "nav_agent": nav_agent_now,
                        "nav_benchmark": nav_benchmark_now,
                        # random对照组已按用户要求(2026-07-14)下线,这个字段
                        # 冻结在期初资本,不再追踪一条真实的随机决策净值——
                        # 保留字段本身是为了不破坏nav_intraday.jsonl现有的三线
                        # schema(webui侧已经把这条线从图表里去掉,不会渲染它)。
                        "nav_random": config["capital_usdt"],
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
                # 又可以点进去看他们的仓位"——每个evo候选分支都按同样的方式
                # 各自落一份标记快照 + 一条小时级净值记录。nav_intraday.jsonl
                # 的既有三线schema(nav_agent/nav_benchmark/nav_random)保持
                # 不动,不动它的既有读者(webui默认图表);这里新增一份独立的、
                # long格式的 nav_intraday_branches.jsonl,覆盖所有evo分支,
                # webui侧再合并展示,不影响已经在跑的旧代码路径。
                for other_branch, other_sim in evo_simulators.items():
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
            last_mark_ms = now

        # 反思(Reflector):每天 reflection_per_day 次,均匀分布节拍。
        # 2026-07-15起覆盖main+所有active evo分支——每个分支的反思写入自己
        # 分支标签下的记忆(L2/L3按分支隔离,见ASSET/memory/engine.py),
        # 进化路线互不污染,这是用户对24h无人值守模式的明确要求。
        last_reflection_ts = schedule_state.get("last_reflection_ts")
        if last_reflection_ts is None or now - last_reflection_ts >= reflection_interval_ms:
            reflection_branches = ["main"] + [
                b for b, m in load_tournament_roster(now, _DEFAULT_EVO_TACTICS).items()
                if m.get("status") == "active"
            ]
            for reflection_branch in reflection_branches:
                try:
                    marks = scheduler.run_reflection_cycle(reflection_branch)
                    print(f"[{now}] [{reflection_branch}] 反思周期完成: {len(marks) if marks is not None else 0} 条", flush=True)
                except Exception:  # noqa: BLE001
                    print(f"[{now}] [{reflection_branch}] run_reflection_cycle 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
                write_heartbeat(clock, {"status": "running"})  # 同上:长反思轮不饿死心跳
            schedule_state["last_reflection_ts"] = now
            save_schedule_state(schedule_state)

        today = _utc_date_str(now)

        # 研究(Researcher):每 research_interval_hours 小时一次。2026-07-15起
        # 改为分支视角轮换制(用户要求):每个研究节拍轮到一个分支,用它自己
        # 的人设视角做研究,产出写进它的私有记忆(L1+branch标签),其他分支
        # 看不到——此前全局共读一份研究,实测导致7个分支全员同向押注同两个
        # 币,种群观点趋同,锦标赛退化成比仓位大小。轮换制下调用量与原来
        # 完全一致(每小时1次),每个分支约N小时轮到一次(N=活跃分支数)。
        if now - last_research_ms >= research_interval_ms:
            research_label = _utc_research_label(now)
            research_roster = load_tournament_roster(now, _DEFAULT_EVO_TACTICS)
            research_branches = ["main"] + [
                b for b, m in research_roster.items() if m.get("status") == "active"
            ]
            research_idx = int(schedule_state.get("research_rotation_idx", 0)) % len(research_branches)
            research_branch = research_branches[research_idx]
            if research_branch == "main":
                research_persona = load_main_tactics() or (
                    "main主账户的稳健基线风格:低杠杆、分散、以可证伪假设驱动"
                )
            else:
                research_persona = research_roster[research_branch]["tactics"]
            # 检索query带上当前市场状态(涨跌最猛的币),让Tavily返回的是
            # "此刻正在发生什么、为什么"的新闻,而不是泛泛的行业文章;
            # 没配Tavily时queries无效果(search_client=None直接返回空)。
            movers = sorted(
                ((s.split("/")[0], v[1]) for s, v in _trend_cache.items() if v[1] is not None),
                key=lambda kv: abs(kv[1]), reverse=True,
            )[:3]
            movers_txt = " ".join(f"{sym}" for sym, _ in movers)
            research_queries = [
                f"crypto market today why moving {movers_txt} news macro fed stocks gold sentiment"
            ]
            try:
                research_result = researcher.daily_research(
                    ts=now, date_str=research_label,
                    queries=research_queries,
                    branch=research_branch, persona=research_persona,
                )
                print(f"[{now}] [{research_branch}] 分支视角研究完成: {research_result}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] [{research_branch}] daily_research 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            schedule_state["research_rotation_idx"] = research_idx + 1
            save_schedule_state(schedule_state)
            last_research_ms = now

        # 每日净值记录(nav.tsv 三线)+ 熔断检查:每个UTC自然日一次。
        if schedule_state.get("last_daily_mark_date") != today:
            try:
                nav_agent = scheduler.simulators["main"].get_portfolio()["nav"]
                nav_benchmark = benchmark_nav_provider(now)
                # random对照组已按用户要求(2026-07-14)下线;nav_random是
                # LOCKED区scorer.daily_mark()的必填参数,这里传期初资本常量
                # 占位,不再追踪一条真实的随机决策净值轨迹。
                nav_random = config["capital_usdt"]
                scheduler.mark_daily_nav(today, nav_agent=nav_agent, nav_benchmark=nav_benchmark, nav_random=nav_random)
                print(
                    f"[{now}] 每日净值记录: agent={nav_agent:.2f} benchmark={nav_benchmark:.2f}",
                    flush=True,
                )

                nav_series = read_nav_tsv_series()
                try:
                    scheduler.run_circuit_breaker_check(_navs_to_ms_series(nav_series["main"]), now_ts=now)
                except Exception:  # noqa: BLE001
                    print(f"[{now}] circuit_breaker(main) 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] 每日净值记录/熔断检查 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            schedule_state["last_daily_mark_date"] = today
            save_schedule_state(schedule_state)

        # 棘轮判定(EvolutionOrchestrator,LOCKED官方评分):每 ratchet_interval_
        # hours 小时一次(用户2026-07-14要求从3天收紧到2小时)。故意只喂main
        # 自己的nav.tsv历史,不再把evo分支塞进来——见下面"战术锦标赛"那段
        # 的说明,evo分支现在完全由一套独立的、不会"一次性裁决后永久退场"的
        # 机制来评估,不适合LOCKED这套judge()语义。
        last_ratchet_ts = schedule_state.get("last_ratchet_ts")
        if last_ratchet_ts is None or now - last_ratchet_ts >= ratchet_interval_ms:
            try:
                nav_series = read_nav_tsv_series()
                if len(nav_series["main"]) >= 2:
                    verdicts = scheduler.run_ratchet_judgment(
                        now_date=today,
                        branch_navs={"main": nav_series["main"]},
                        benchmark_navs=nav_series["benchmark"],
                    )
                    print(f"[{now}] 棘轮判定完成: {len(verdicts)} 条候选分支裁决", flush=True)
                else:
                    print(f"[{now}] 棘轮判定跳过: nav.tsv历史点数不足({len(nav_series['main'])})", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[{now}] run_ratchet_judgment 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            schedule_state["last_ratchet_ts"] = now
            save_schedule_state(schedule_state)

        # 战术锦标赛(用户2026-07-14要求"自动进化+晋升赢家"):复用同一个
        # ratchet_interval_ms节拍,评估每个active分支相对main的小时级净值
        # 表现。与上面的官方棘轮判定完全独立、互不干扰——见
        # evaluate_tactic_tournament()的docstring对两者语义差异的说明。
        last_tournament_ms = schedule_state.get("last_tournament_ms")
        if last_tournament_ms is None or now - last_tournament_ms >= ratchet_interval_ms:
            try:
                tournament_cfg = config.get("tactic_tournament", {}) or {}
                roster = load_tournament_roster(now, _DEFAULT_EVO_TACTICS)
                events = evaluate_tactic_tournament(roster, now, tournament_cfg)

                # 末位斩杀(用户2026-07-15要求,见evaluate_cull的docstring):
                # 独立于上面两条绝对红线的定期强制换血,按自己的节拍
                # (cull_interval_hours)触发,不跟随每次锦标赛评估。
                cull_interval_ms = int(float(tournament_cfg.get("cull_interval_hours", 12)) * 3_600_000)
                last_cull_ms = schedule_state.get("last_cull_ms")
                if last_cull_ms is None:
                    # 首轮不斩,先记时间起点——刚清零/刚部署时所有分支都没有
                    # 满窗口的数据,立刻评估只会空转。
                    schedule_state["last_cull_ms"] = now
                    save_schedule_state(schedule_state)
                elif now - last_cull_ms >= cull_interval_ms:
                    cull_event = evaluate_cull(roster, now, tournament_cfg)
                    if cull_event is not None:
                        events.append(cull_event)
                    schedule_state["last_cull_ms"] = now
                    save_schedule_state(schedule_state)

                for ev in events:
                    branch = ev["branch"]
                    log_writer.append_jsonl("tactic_promotions.jsonl", {**ev, "ts": now}, root=LOG_ROOT)
                    print(
                        f"[{now}] 战术锦标赛裁决: {branch} -> {ev['decision']} "
                        f"(edge={ev['edge_vs_main_pct']:.2f}%, drawdown={ev['max_drawdown_pct']:.2f}%)",
                        flush=True,
                    )
                    roster[branch]["status"] = ev["decision"].lower()  # "promoted" / "failed" / "culled"
                    roster[branch]["resolved_ms"] = now

                    if ev["decision"] == "PROMOTE":
                        # M8两级晋升(见 promote_policy_branch 模块docstring):
                        # 锦标赛自己的PROMOTE只是"有资格去问真正的裁判"这一步,
                        # 这个分支离开锦标赛active池、腾出名额(上面两行已经
                        # 做了)不代表main真的被改变——是否真的合并代码/切换
                        # policy,完全由 promote_policy_branch() 里对
                        # EvolutionOrchestrator.judge() + GitMergeExecutor 的
                        # 真实调用结果决定。挂着policy_id的分支(M7代码化策略)
                        # 走这条真实链路;没有policy_id的老式纯提示词分支
                        # 维持原有的"战术文字接管main"行为不变(它们没有真实
                        # 代码可以合并,GitMergeExecutor对它们没有意义)。
                        policy_id = roster[branch].get("policy_id")
                        if policy_id is None:
                            winning_tactics = roster[branch]["tactics"]
                            save_main_tactics(winning_tactics, branch, now)
                            print(f"[{now}] main分支战术已更新为 {branch} 的战术(立即生效,不需要重启)", flush=True)
                        else:
                            try:
                                outcome = promote_policy_branch(
                                    branch=branch, policy_id=policy_id, roster=roster,
                                    evolution_orchestrator=evolution_orchestrator,
                                    scheduler=scheduler, today=today,
                                )
                                print(
                                    f"[{now}] [{branch}] M8两级晋升(policy_id={policy_id}): "
                                    f"{outcome['summary']}", flush=True,
                                )
                            except Exception:  # noqa: BLE001 -- 真正的裁决/merge出问题
                                # 不应该拖垮整轮锦标赛处理(与本文件其它 try/except
                                # 段落同一纪律),异常记录后本轮就当没有真的晋升,
                                # main状态保持不变,下一次锦标赛节拍如果再次判定
                                # 该policy分支(或它的替补)PROMOTE会自然重试。
                                print(
                                    f"[{now}] [{branch}] promote_policy_branch 异常"
                                    f"(已记录,继续循环,main状态未改变):\n{traceback.format_exc()}",
                                    flush=True,
                                )
                    elif ev["decision"] in ("FAIL", "CULLED"):
                        # 确定性强制平仓,不经过Trader/子代理判断——这是锦标赛
                        # 淘汰机制本身的动作,与main.py.run_risk_check_cycle
                        # (LOCKED区确定性风控哨兵)同一种"不等agent推理"的设计
                        # 理念,但这里的裁决逻辑是本脚本自己的锦标赛规则,不是
                        # LOCKED区代码。FAIL=回撤红线;CULLED=末位斩杀,处理
                        # 流程完全相同(强平退场+自动生成替补),只是死因不同。
                        failed_sim = evo_simulators.get(branch)
                        if failed_sim is not None:
                            for symbol, pos in list(failed_sim.positions.items()):
                                close_decision = main_module.Decision(
                                    ts=now, symbol=symbol, action="close", target_notional_pct=0.0,
                                    leverage=pos.leverage,
                                    thesis=(
                                        f"锦标赛出局强制平仓({ev['decision']}: {ev['reason'][:80]}),"
                                        f"确定性机制触发,不是Trader/子代理的主观判断。"
                                    ),
                                    falsifier="本决策为锦标赛淘汰机制触发,不设新的可证伪主张。",
                                    horizon="0h", branch=branch,
                                )
                                failed_sim.log_decision(close_decision)
                                next_bar = next_bar_provider(symbol, now)
                                failed_sim.execute(close_decision, next_bar)
                                print(f"[{now}] [{branch}] 锦标赛出局强制平仓: {symbol}", flush=True)

                if events:
                    save_tournament_roster(roster)
                    universe_symbols_now = (
                        json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))["symbols"]
                        if UNIVERSE_PATH.exists() else []
                    )
                    for ev in events:
                        resolved_branch = ev["branch"]
                        try:
                            generated = generate_replacement_tactic(
                                deep_llm, ev, roster, universe_symbols_now
                            )
                        except Exception:  # noqa: BLE001
                            generated = None
                            print(
                                f"[{now}] [{resolved_branch}] generate_replacement_tactic 异常"
                                f"(已记录,继续循环):\n{traceback.format_exc()}",
                                flush=True,
                            )

                        if generated is None:
                            log_writer.append_jsonl(
                                "tactic_generations.jsonl",
                                {
                                    "ts": now, "resolved_branch": resolved_branch,
                                    "resolved_decision": ev["decision"],
                                    "status": "generation_failed", "new_branch_id": None,
                                },
                                root=LOG_ROOT,
                            )
                            print(
                                f"[{now}] [{resolved_branch}] 自动生成替补战术失败(连续"
                                f"{_TACTIC_GENERATION_MAX_RETRIES}次校验不通过),名额暂时"
                                f"空缺,下一次锦标赛节拍会自动重试",
                                flush=True,
                            )
                            continue

                        new_branch_id = generated["branch_id"]
                        roster[new_branch_id] = {
                            "tactics": generated["tactics"] + _TOURNAMENT_STAKES_SUFFIX,
                            "status": "active",
                            "created_ms": now,
                        }
                        save_tournament_roster(roster)
                        log_writer.append_jsonl(
                            "tactic_generations.jsonl",
                            {
                                "ts": now, "resolved_branch": resolved_branch,
                                "resolved_decision": ev["decision"],
                                "status": "generated", "new_branch_id": new_branch_id,
                            },
                            root=LOG_ROOT,
                        )
                        print(
                            f"[{now}] 自动生成替补战术分支: {new_branch_id}"
                            f"(替补 {resolved_branch} 腾出的名额,立即生效,不需要重启)",
                            flush=True,
                        )
            except Exception:  # noqa: BLE001
                print(f"[{now}] evaluate_tactic_tournament 异常(已记录,继续循环):\n{traceback.format_exc()}", flush=True)
            schedule_state["last_tournament_ms"] = now
            save_schedule_state(schedule_state)

        write_heartbeat(clock, {"status": "running"})
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
