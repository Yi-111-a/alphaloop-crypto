"""
scripts/research_loop.py —— M7 内环研究循环(改造规格书§3.4/§3.5)。

职责:一个可独立运行的调度脚本,每一轮:挑一个研究想法 -> 先提交"协议"
(假设+计划,git commit,早于任何结果落地)-> 让LLM(deep_llm,研究员角色)
写/改一个 ASSET/strategy/policies/{policy_id}.py 确定性策略模块 -> AST静态
lint -> 加载并跑 LOCKED.backtest_engine.BacktestEngine 回测 -> 对比该
policy_id的历史最优分数,决定keep(git commit)还是revert(git checkout恢复
文件)-> 把这轮结果连同holdout表现(仅记录,不参与判定)追加进
ASSET/strategy/experiments/ledger.jsonl。

跑法:
    python scripts/research_loop.py --max-experiments 20
    python scripts/research_loop.py --max-experiments 5 --dry-run   # 只看想法/prompt/lint,不动磁盘/git

设计上的几条硬边界(§3.4"结构性保证,不是纪律性保证"同款精神):
  1. keep/revert 判定函数 decide_keep_or_revert(new_score, previous_best_score)
     只接受两个已经算好的浮点数,签名里没有 results/holdout 这样的参数——
     holdout 结果不可能通过这个函数的参数表被判定逻辑看到,这是结构性保证,
     不是"我们保证调用时不传"这种约定层面的自律。holdout 的表现只在
     run_single_experiment() 里单独抽出来、原样写进 ledger 供日后晋升评审用。
  2. git 操作(git_commit_protocol/git_commit_result/git_revert_file)全部是
     接受 repo_path 参数的纯 subprocess 封装,不隐式假设"当前工作目录就是
     本仓库"——测试永远传一个 tmp_path 下 `git init` 出来的临时仓库,本脚本
     开发/测试过程中不对本仓库执行任何真实 git 写操作。
  3. LLM 生成的策略源码在写入磁盘前必须先过 policy_lint.lint_policy_source;
     写入目标只能是 ASSET/strategy/policies/{policy_id}.py,policy_id 必须先
     通过 ^[a-z][a-z0-9_]{2,40}$ 白名单校验(validate_policy_id/
     safe_policy_path),防止路径穿越(比如 "../evil" 这种输入必须在写盘之前
     就被拒绝,而不是拼出路径之后才发现越界)。
  4. 本文件不复用 ASSET.strategy.policies.load_policy() 本身去加载回测用的
     策略模块——那个函数的 POLICIES_DIR 是硬编码为"自己所在目录"的模块级
     常量,没有办法参数化指向一个临时目录,而测试要求"绝不对本仓库执行真实
     git 写操作"(自然也包括不能真的写主目录下的策略文件再删掉)。这里改用
     load_policy_from_dir(policy_id, policies_dir),是 load_policy() 校验
     契约的一个可参数化镜像实现(同样的"每次从磁盘重新编译执行,不复用
     字节码缓存"设计),生产环境调用时传入的就是真实的
     ASSET.strategy.policies.POLICIES_DIR,行为与 load_policy() 完全一致;
     测试则可以传任意 tmp_path 下的目录,不接触真实策略文件。这是本次实现
     里"规格书未明确"因此需要自行拍板的一项设计决定(任务列表里明确要求
     如实标注)。
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 与 scripts/ignite.py 同样的 sys.path 处理:允许本文件既能被
# `python scripts/research_loop.py` 直接运行,也能被测试以
# `import scripts.research_loop` 的方式导入(pytest.ini pythonpath=. 已经把
# 项目根目录放上 sys.path,这里的 insert 对该场景是幂等的无害重复)。
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ASSET.strategy.policies import POLICIES_DIR, PolicyLoadError  # noqa: E402
from ASSET.strategy.policy_lint import lint_policy_source  # noqa: E402
from LOCKED.backtest_engine import BacktestEngine, BacktestWindow  # noqa: E402
from scripts.ignite import build_llm_clients  # noqa: E402
from scripts.llm_bridge import _strip_code_fences as strip_code_fences  # noqa: E402

DEFAULT_UNIVERSE_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
_BENCHMARK_SYMBOL = "BTC/USDT:USDT"

_POLICY_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
_DATE_NOTE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(-\d{2})?\.md$")
_HYPOTHESIS_RE = re.compile(r"^##\s*H\d+:\s*(.+)$", re.MULTILINE)

_SEED_POLICY_IDS = ["aggressive_v1", "carry_v1", "conservative_v1", "diversified_v1", "momentum_v1"]
_DEFAULT_REFERENCE_POLICY_ID = "diversified_v1"  # research_note来源"新建型"时的默认参照源码

MAX_LINT_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# 1. 窗口切分(纯函数,可独立测试)
# ---------------------------------------------------------------------------


def build_default_windows(config: dict, data_end_ts: int) -> list[BacktestWindow]:
    """按 config.backtest.windows 从 data_end_ts 往前切:
    train(365天,最老)-> val_1 -> val_2(各90天)-> holdout(90天,is_holdout=True,
    时间上最新、与 data_end_ts 对齐)。四段时间上首尾相接,不留缝隙也不重叠。"""
    windows_cfg = (config.get("backtest", {}) or {}).get("windows", {}) or {}
    train_days = int(windows_cfg.get("train_days", 365))
    val_window_days = int(windows_cfg.get("val_window_days", 90))
    val_window_count = int(windows_cfg.get("val_window_count", 2))
    holdout_days = int(windows_cfg.get("holdout_days", 90))

    day_ms = 86_400_000
    end = data_end_ts
    holdout_start = end - holdout_days * day_ms

    val_windows: list[BacktestWindow] = []
    cursor_end = holdout_start
    for i in range(val_window_count, 0, -1):
        cursor_start = cursor_end - val_window_days * day_ms
        val_windows.append(BacktestWindow(label=f"val_{i}", start_ts=cursor_start, end_ts=cursor_end))
        cursor_end = cursor_start
    val_windows.reverse()  # val_1(最老)... val_n(最新,紧邻holdout)

    train_end = cursor_end
    train_start = train_end - train_days * day_ms

    windows = [BacktestWindow(label="train", start_ts=train_start, end_ts=train_end)]
    windows.extend(val_windows)
    windows.append(BacktestWindow(label="holdout", start_ts=holdout_start, end_ts=end, is_holdout=True))
    return windows


def load_universe_symbols(project_root: Path) -> list[str]:
    """读 universe_active.json 的 symbols 列表;文件不存在/格式不对/为空时
    回退到规格书指定的默认三标的。"""
    path = project_root / "universe_active.json"
    if not path.exists():
        return list(DEFAULT_UNIVERSE_SYMBOLS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return list(DEFAULT_UNIVERSE_SYMBOLS)
    symbols = data.get("symbols")
    if not symbols:
        return list(DEFAULT_UNIVERSE_SYMBOLS)
    return list(symbols)


def determine_data_end_ts(config: dict, cache_dir: Path, benchmark_symbol: str = _BENCHMARK_SYMBOL) -> int:
    """用 data_cache 里 BTC 缓存的 OHLCV parquet 的最大时间戳推出回测窗口的
    终点(data_end_ts)。文件命名规则与 LOCKED.data_pipeline.DataPipeline.
    _symbol_cache_path 完全一致(safe_symbol = symbol替换'/'->'-'、':'->'_' ),
    这里没有依赖那个私有方法,是就地复刻同样几行的小工具,与
    LOCKED/backtest_engine.py 对 _timeframe_to_ms 的处理方式同源。"""
    timeframe = (config.get("data", {}) or {}).get("timeframe", "4h")
    safe_symbol = benchmark_symbol.replace("/", "-").replace(":", "_")
    cache_path = Path(cache_dir) / f"ohlcv_{safe_symbol}_{timeframe}.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"determine_data_end_ts: BTC OHLCV cache not found at {cache_path}; "
            "run scripts/backfill_history.py first"
        )
    df = pd.read_parquet(cache_path)
    if df.empty:
        raise ValueError(f"determine_data_end_ts: {cache_path} is empty")
    return int(df["timestamp"].max())


# ---------------------------------------------------------------------------
# 2. ledger 读写(append-only jsonl)
# ---------------------------------------------------------------------------


def read_ledger(ledger_path: Path) -> list[dict]:
    if not ledger_path.exists():
        return []
    records: list[dict] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 容忍读到半行(另一个进程正在追加写入中途读到)
    return records


def append_ledger(ledger_path: Path, record: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_ledger_record(
    *,
    experiment_id: str,
    idea: "Idea",
    commit_sha_protocol: Optional[str],
    commit_sha_result: Optional[str],
    status: str,
    val_edge_vs_benchmark_pct: Optional[float],
    val_max_drawdown_pct: Optional[float],
    holdout_edge_vs_benchmark_pct: Optional[float],
    wall_time_seconds: float,
    revert_reason: Optional[str] = None,
    holdout_trade_count: Optional[int] = None,
    holdout_return_pct: Optional[float] = None,
) -> dict:
    """schema 逐字遵守任务说明里列出的字段集合,另加四个向后兼容的新字段:

    - revert_reason:kept时为None;status=="reverted"时按触发原因区分——
      "insufficient_trades"(任务2:验证窗口trade_count总和不足门槛,直接
      revert,分数再高也不算数)或"score_not_better"(正常棘轮判定没打赢
      历史最优)。lint_failed/backtest_failed这两种status与"revert原因"
      的概念无关,统一为None。
    - strategy_class:source=="director"时来自研究总监备忘录的direction.
      class,其余来源为None。写进ledger是为了下次
      collect_known_strategy_classes()能认出"这个类别已经出现过"，不需要
      额外的状态文件。
    - holdout_trade_count / holdout_return_pct(本次改动新增,修"holdout
      躺赢"缺陷):分别来自 results['holdout'].trade_count 和
      results['holdout'].return_pct(绝对收益,不是edge)。动机——实测发现
      holdout窗口BTC大跌期间,8个在holdout里零交易的策略全部得到完全相同
      的 holdout_edge_vs_benchmark_pct(因为它们的持仓/净值曲线在holdout
      全程都是"没有任何操作",edge纯粹等于"跑赢/跑输BTC基准"的镜像,不是
      策略自己的能力)。这两个字段让下游(人工review/研究总监prompt)能
      识别"这个holdout_edge到底有没有参考价值"——holdout_trade_count==0时
      holdout_edge_vs_benchmark_pct不代表任何真实的策略能力,只是基准
      涨跌的镜像。lint_failed/backtest_failed状态下(回测从未跑起来)
      两个字段均为None,与holdout_edge_vs_benchmark_pct同一处理口径。

    读旧行(不含这四个key)的调用方一律用 .get(...) 取值,缺字段不会报错——
    向后兼容是字段新增的硬要求,不是可选项。"""
    return {
        "experiment_id": experiment_id,
        "policy_id": idea.policy_id,
        "parent_policy_id": idea.parent_policy_id,
        "hypothesis": idea.hypothesis,
        "commit_sha_protocol": commit_sha_protocol,
        "commit_sha_result": commit_sha_result,
        "status": status,
        "val_edge_vs_benchmark_pct": val_edge_vs_benchmark_pct,
        "val_max_drawdown_pct": val_max_drawdown_pct,
        "holdout_edge_vs_benchmark_pct": holdout_edge_vs_benchmark_pct,
        "wall_time_seconds": wall_time_seconds,
        "revert_reason": revert_reason,
        "strategy_class": idea.strategy_class,
        "holdout_trade_count": holdout_trade_count,
        "holdout_return_pct": holdout_return_pct,
    }


# ---------------------------------------------------------------------------
# 3. 想法挑选(§3.5三个来源,按序轮转;冷启动从种子策略开始)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Idea:
    source: str  # "kept_variant" | "research_note" | "seed_variant" | "director" | "novelty"
    policy_id: str  # 目标文件名(已通过路径安全校验的候选)
    parent_policy_id: Optional[str]
    hypothesis: str
    reference_policy_id: Optional[str]  # "新建型"时用来给LLM参照的源码来自哪个policy_id
    # source=="director"时由研究总监备忘录里的direction.class带过来,用于
    # (a) 给LLM研究员prompt里点名要实现的类别 (b) 写进ledger后供下一次
    # collect_known_strategy_classes()识别"这个类别已经出现过"。其余来源
    # (kept_variant/research_note/seed_variant/novelty)一律为None——不是
    # 每条idea都天然有一个"类别名"这个概念,novelty来源的类别名要等LLM真的
    # 写出策略之后才知道,这里不去猜测/伪造一个。
    strategy_class: Optional[str] = None


def select_idea_source(experiment_index: int, ledger_has_kept: bool) -> str:
    """三个来源按序轮转:kept_variant(a) -> research_note(b) -> seed_variant(c)。

    冷启动(ledger里还没有任何status=kept的记录)时,不管experiment_index是
    多少,一律强制从(c)种子策略变体开始——这是任务里明确写的规则,不是"轮转
    到哪个算哪个"的自然结果:来源(a)在没有任何kept策略时根本没有素材可用,
    与其去回退到(b),这里显式遵照规格书选择直接跳到(c)。"""
    if not ledger_has_kept:
        return "seed_variant"
    order = ("kept_variant", "research_note", "seed_variant")
    return order[experiment_index % 3]


def _strip_variant_suffix(policy_id: str) -> str:
    """去掉 policy_id 已有的 "_vN" 后缀(如果有),得到它的"策略家族名"——
    momentum_v1/momentum_v2 都属于家族名"momentum"。_next_policy_variant_id
    与"策略类别"相关的启发式(collect_known_strategy_classes/
    _pick_reference_for_class)共用这一条规则,避免同一个概念散落成两份
    正则。"""
    return re.sub(r"_v\d+$", "", policy_id) or policy_id


def _next_policy_variant_id(base_policy_id: str, policies_dir: Path) -> str:
    """给 base_policy_id 生成一个尚未存在于 policies_dir 下、且满足
    ^[a-z][a-z0-9_]{2,40}$ 白名单的新 policy_id,形如 "{stem}_v{n}"。
    先去掉 base_policy_id 已有的 "_vN" 后缀(如果有),避免变体的变体越叠
    越长(momentum_v1 -> momentum_v2,而不是 momentum_v1_v2)。"""
    stem = _strip_variant_suffix(base_policy_id)
    n = 1
    while True:
        candidate = f"{stem}_v{n}"
        if _POLICY_ID_RE.match(candidate) and not (policies_dir / f"{candidate}.py").exists():
            return candidate
        n += 1


def latest_research_note(notes_dir: Path) -> Optional[Path]:
    """挑"最新"研究笔记:优先按文件名里的日期(YYYY-MM-DD[-HH].md)降序取最新
    一份;完全没有按日期命名的研究产物时(比如刚冷启动、只有genesis.md),
    回退到 genesis.md。"""
    if not notes_dir.exists():
        return None
    dated = sorted(
        (p for p in notes_dir.glob("*.md") if _DATE_NOTE_RE.match(p.name)),
        key=lambda p: p.name,
        reverse=True,
    )
    if dated:
        return dated[0]
    genesis = notes_dir / "genesis.md"
    return genesis if genesis.exists() else None


def _extract_hypotheses(note_text: str) -> list[str]:
    """从研究笔记 markdown 里取出 "## H<n>: <正文>" 形式的假设段落文本
    (与 ASSET/research_notes/genesis.md 的既有格式一致)。"""
    return [m.strip() for m in _HYPOTHESIS_RE.findall(note_text) if m.strip()]


def _load_direction_memo(experiments_dir: Optional[Path]) -> Optional[dict]:
    """读 experiments_dir/direction_memo.json;文件不存在、不是合法JSON、或
    顶层不是一个JSON对象,一律返回None(冷启动/总监从未成功运行过时的
    正常状态,不是错误)。"""
    if experiments_dir is None:
        return None
    path = Path(experiments_dir) / "direction_memo.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _pick_reference_for_class(class_name: str, ledger_entries: list[dict], policies_dir: Path) -> str:
    """给研究总监指定的策略类别"就近"挑一个参照源码:优先在磁盘上真实存在
    的已保留(kept)策略里找家族名与class_name有词面重叠的,其次退回同样
    规则下的种子策略,都找不到就退回默认参照(diversified_v1)。

    这是一个尽力而为的启发式,不是精确的语义分类器——class_name完全可能是
    总监起的纯中文类别名(比如"资金费率carry的时段增强版"),这种情况下
    大概率找不到任何词面重叠,直接落到默认参照分支。规格书没有规定"就近"
    具体怎么算,这是本次实现自行拍板的一项设计决定。"""
    needle = class_name.strip().lower()
    kept_ids = [e["policy_id"] for e in ledger_entries if e.get("status") == "kept" and e.get("policy_id")]
    seen: set[str] = set()
    for candidate in list(dict.fromkeys(kept_ids)) + list(_SEED_POLICY_IDS):
        if candidate in seen:
            continue
        seen.add(candidate)
        family = _strip_variant_suffix(candidate).lower()
        if family and (family in needle or needle in family) and (policies_dir / f"{candidate}.py").exists():
            return candidate
    return _DEFAULT_REFERENCE_POLICY_ID


_NOVELTY_HYPOTHESIS = (
    "新颖性槽位:设计一个当前所有已知策略类别之外的全新策略类别,而不是对"
    "已有策略做参数/信号微调。可参考经典类别菜单启发(不限于此):配对/"
    "价差回归、横截面多空排名、波动率突破、资金费率carry、时段效应、"
    "相关性regime切换;也允许发明菜单之外的类别。"
)


def _select_director_idea(experiment_index: int, directions: list[dict], ledger_entries: list[dict], policies_dir: Path) -> Idea:
    """消费研究总监备忘录里的一条研究方向(轮转:experiment_index % len(directions))。"""
    direction = directions[experiment_index % len(directions)]
    class_name = str(direction.get("class") or "").strip() or "unspecified"
    hypothesis = str(direction.get("hypothesis") or "").strip() or "研究总监未提供具体假设文本。"
    policy_id = _next_policy_variant_id("director_idea", policies_dir)
    reference_policy_id = _pick_reference_for_class(class_name, ledger_entries, policies_dir)
    return Idea(
        source="director", policy_id=policy_id, parent_policy_id=None,
        hypothesis=hypothesis, reference_policy_id=reference_policy_id,
        strategy_class=class_name,
    )


def _select_novelty_idea(policies_dir: Path) -> Idea:
    """新颖性槽位(每第5个实验强制,见select_idea):固定的探索性假设文本,
    真正"发明一个新类别"的创造性工作交给build_researcher_prompt里对
    source=="novelty"专门构造的prompt(带上已存在类别清单+经典类别菜单)。
    strategy_class刻意留空——novelty产出的策略具体归属哪个新类别,要等LLM
    真的写出DESCRIPTION之后才知道,这里不预先猜测/伪造一个类别名。"""
    policy_id = _next_policy_variant_id("novelty_idea", policies_dir)
    return Idea(
        source="novelty", policy_id=policy_id, parent_policy_id=None,
        hypothesis=_NOVELTY_HYPOTHESIS, reference_policy_id=_DEFAULT_REFERENCE_POLICY_ID,
        strategy_class=None,
    )


def select_idea(
    experiment_index: int,
    ledger_entries: list[dict],
    policies_dir: Path,
    notes_dir: Path,
    experiments_dir: Optional[Path] = None,
) -> Idea:
    """想法来源轮转优先级(本次改动新增了两层,凌驾于原有三来源轮转之上):

    1. 新颖性槽位:每第5个实验(experiment_index % 5 == 4)无条件强制走
       source="novelty"——不管备忘录存不存在,这条槽位专门用来对抗"内环
       只会顺着已有类别做局部变体、永远发明不出全新类别"这个问题。
    2. 研究总监备忘录:experiments_dir/direction_memo.json 存在且有非空
       directions 时,优先消费备忘录方向(source="director"),按
       experiment_index % len(directions) 轮转取一条。
    3. 备忘录不存在(典型的"首夜",总监还从未成功运行过一次)时,退回
       原有的 kept_variant/research_note/seed_variant 三来源轮转。

    experiments_dir 是新增的可选参数(默认None——省略即视为"不查备忘录",
    向后兼容既有调用/测试)。"""
    if experiment_index % 5 == 4:
        return _select_novelty_idea(policies_dir)

    memo = _load_direction_memo(experiments_dir)
    directions = memo.get("directions") if memo else None
    if directions:
        return _select_director_idea(experiment_index, directions, ledger_entries, policies_dir)

    kept_entries = [e for e in ledger_entries if e.get("status") == "kept"]
    source = select_idea_source(experiment_index, ledger_has_kept=bool(kept_entries))

    if source == "kept_variant":
        parent_entry = kept_entries[experiment_index % len(kept_entries)]
        parent_id = parent_entry["policy_id"]
        policy_id = _next_policy_variant_id(parent_id, policies_dir)
        hypothesis = (
            f"对已保留策略 {parent_id} 做局部参数/信号变体:在其现有逻辑框架基础上"
            f"调整关键参数或信号组合的取值区间,验证能否在保持验证集稳健性的前提下"
            f"进一步提升相对基准的edge。"
        )
        return Idea(
            source=source, policy_id=policy_id, parent_policy_id=parent_id,
            hypothesis=hypothesis, reference_policy_id=parent_id,
        )

    if source == "research_note":
        note_path = latest_research_note(notes_dir)
        hypotheses = _extract_hypotheses(note_path.read_text(encoding="utf-8")) if note_path else []
        if hypotheses:
            note_text = hypotheses[experiment_index % len(hypotheses)]
        else:
            note_text = "研究笔记中未发现可提取的假设文本,退化为通用探索性假设。"
        policy_id = _next_policy_variant_id("research_idea", policies_dir)
        hypothesis = f"基于研究笔记假设构造新策略:{note_text}"
        return Idea(
            source=source, policy_id=policy_id, parent_policy_id=None,
            hypothesis=hypothesis, reference_policy_id=_DEFAULT_REFERENCE_POLICY_ID,
        )

    # source == "seed_variant"
    base = _SEED_POLICY_IDS[experiment_index % len(_SEED_POLICY_IDS)]
    policy_id = _next_policy_variant_id(base, policies_dir)
    hypothesis = (
        f"探索种子策略 {base} 的其他参数区间/信号组合(不同阈值、杠杆、回看窗口"
        f"长度等),验证是否存在比原始种子参数更稳健的变体。"
    )
    return Idea(
        source="seed_variant", policy_id=policy_id, parent_policy_id=base,
        hypothesis=hypothesis, reference_policy_id=base,
    )


# ---------------------------------------------------------------------------
# 4. policy_id 路径安全校验
# ---------------------------------------------------------------------------


def validate_policy_id(policy_id: str) -> None:
    if not isinstance(policy_id, str) or not _POLICY_ID_RE.match(policy_id):
        raise ValueError(
            f"policy_id fails path-safety whitelist ^[a-z][a-z0-9_]{{2,40}}$: {policy_id!r}"
        )


def safe_policy_path(policies_dir: Path, policy_id: str) -> Path:
    """先校验 policy_id 本身(白名单正则,拒绝 "../evil" 这类穿越尝试),再
    额外确认拼出来的绝对路径确实落在 policies_dir 内部——双重防御,即使正则
    本身出现意外漏洞,第二层解析路径校验依然能挡住越界写入。"""
    validate_policy_id(policy_id)
    resolved_dir = Path(policies_dir).resolve()
    path = (resolved_dir / f"{policy_id}.py").resolve()
    if path.parent != resolved_dir:
        raise ValueError(f"policy_id resolves outside policies_dir: {policy_id!r} -> {path}")
    return path


# ---------------------------------------------------------------------------
# 5. 加载策略模块(可参数化目录版本,见模块docstring第4条设计说明)
# ---------------------------------------------------------------------------

_load_counter = itertools.count()


def load_policy_from_dir(policy_id: str, policies_dir: Path):
    """镜像 ASSET.strategy.policies.load_policy() 的校验契约与"每次都从磁盘
    重新编译执行、不复用字节码缓存"的实现方式,但 policies_dir 可注入——原函数
    把 POLICIES_DIR 写死成自己所在的目录,没有参数化的余地。生产运行时这里
    传入的就是真实的 ASSET.strategy.policies.POLICIES_DIR,行为与
    load_policy() 完全一致;测试传入 tmp_path 下的目录,不接触真实策略文件。"""
    import importlib.util
    import sys as _sys

    policies_dir = Path(policies_dir)
    file_path = policies_dir / f"{policy_id}.py"
    if not file_path.exists():
        raise PolicyLoadError(f"policy file not found: {file_path} (policy_id={policy_id!r})")

    internal_name = f"_alphaloop_research_loop_policy_{policy_id}_{next(_load_counter)}"
    source = file_path.read_text(encoding="utf-8")
    try:
        code = compile(source, str(file_path), "exec")
    except SyntaxError as exc:
        raise PolicyLoadError(f"failed to compile policy module {file_path}: {exc!r}") from exc

    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(internal_name, loader=None))
    module.__file__ = str(file_path)
    _sys.modules[internal_name] = module
    try:
        exec(code, module.__dict__)
    except Exception as exc:
        _sys.modules.pop(internal_name, None)
        raise PolicyLoadError(f"failed to execute policy module {file_path}: {exc!r}") from exc

    missing: list[str] = []
    if not hasattr(module, "decide") or not callable(getattr(module, "decide")):
        missing.append("decide (callable)")
    hist_bars = getattr(module, "REQUIRED_HISTORY_BARS", None)
    if not isinstance(hist_bars, int) or isinstance(hist_bars, bool):
        missing.append("REQUIRED_HISTORY_BARS (int)")
    if not isinstance(getattr(module, "DESCRIPTION", None), str):
        missing.append("DESCRIPTION (str)")
    if missing:
        _sys.modules.pop(internal_name, None)
        raise PolicyLoadError(
            f"policy module {file_path} is missing required export(s): {', '.join(missing)}"
        )
    return module


# ---------------------------------------------------------------------------
# 6. LLM prompt 构造 + lint 重写循环
# ---------------------------------------------------------------------------


_FORBIDDEN_SUMMARY = (
    "- 禁止任何网络调用/网络库导入(requests/httpx/urllib/socket/ccxt/anthropic/aiohttp/http/websocket/grpc等)\n"
    "- 禁止读墙钟(time.time()/time.time_ns()/datetime.now()/utcnow()/date.today()等)——"
    "\"现在几点\"只能来自 ctx.ts\n"
    "- 禁止随机数(random模块、numpy.random)\n"
    "- 禁止 os.environ / os.getenv / subprocess / exec / eval\n"
    "- open() 只允许只读模式(默认或显式 'r'/'rb'/'rt'),禁止写/追加/独占模式('w'/'a'/'x'/'+')\n"
)


def resolve_reference(policies_dir: Path, idea: Idea) -> tuple[str, str]:
    """决定给LLM看的参照源码:目标文件已存在 -> "改进型"(给它自己当前的源码);
    目标文件不存在(通常情况,因为 idea 的 policy_id 由
    _next_policy_variant_id 生成,保证是全新文件名)-> "新建型"(给最相近的
    种子/父策略源码作参照)。"""
    target_path = policies_dir / f"{idea.policy_id}.py"
    if target_path.exists():
        return "improve", target_path.read_text(encoding="utf-8")
    ref_id = idea.reference_policy_id or _DEFAULT_REFERENCE_POLICY_ID
    ref_path = policies_dir / f"{ref_id}.py"
    ref_source = ref_path.read_text(encoding="utf-8") if ref_path.exists() else ""
    return "new", ref_source


def _build_source_specific_prompt_section(idea: Idea, known_classes: Optional[list[str]]) -> str:
    """director/novelty 两种来源在"你是研究员,写代码"这个通用外壳之外,各自
    需要一段额外的定向要求段落;kept_variant/research_note/seed_variant
    维持原有prompt形态,不受影响(返回空字符串)。"""
    if idea.source == "director":
        return (
            "\n## 研究总监指定方向\n"
            f"- 目标策略类别:{idea.strategy_class or '(未指定)'}\n"
            "这条假设来自外环研究总监对全部历史实验轨迹的复盘,请围绕这个类别"
            "具体落地,不要偏离到无关的策略类型。\n"
        )
    if idea.source == "novelty":
        classes_text = "、".join(known_classes) if known_classes else "(暂无记录)"
        return (
            "\n## 新颖性槽位要求\n"
            f"- 已存在的策略类别清单(必须避开,不要与其重复或只做参数微调):{classes_text}\n"
            "- 请设计一个上述清单中不存在的全新策略类别。经典类别菜单仅作启发"
            "(不限于此):配对/价差回归、横截面多空排名、波动率突破、资金"
            "费率carry、时段效应、相关性regime切换;也允许发明菜单之外的类别。\n"
        )
    return ""


def build_researcher_prompt(
    idea: Idea,
    target_policy_id: str,
    reference_kind: str,
    reference_source: str,
    known_classes: Optional[list[str]] = None,
) -> str:
    reference_note = (
        "改进型:下面是目标文件当前的源码,请在此基础上按假设修改"
        if reference_kind == "improve"
        else "新建型:下面是最相近的参考源码(种子策略或血缘父策略),请仿照其结构新建"
    )
    source_specific = _build_source_specific_prompt_section(idea, known_classes)
    return (
        "你是AlphaLoop-Crypto的内环策略研究员。任务:为下面这条假设写/改一个"
        "确定性的Python策略模块(ASSET/strategy/policies 目录下的policy)。\n\n"
        f"## 目标 policy_id\n{target_policy_id}\n\n"
        f"## 假设\n{idea.hypothesis}\n"
        f"{source_specific}\n"
        f"## 参考源码({reference_note})\n"
        f"```python\n{reference_source}\n```\n\n"
        "## StrategyContext 协议(decide(ctx)的唯一输入)\n"
        "- ctx.ts: int,当前bar时间戳(UTC毫秒),策略判断\"现在几点\"只能用这个字段\n"
        "- ctx.positions: dict[str, PerpPosition],当前持仓快照\n"
        "- ctx.snapshot: dict[str, dict],{symbol: {'last': 最新收盘价}}\n"
        "- ctx.recent_bars: dict[str, pd.DataFrame],{symbol: 最近K线,列为"
        "[timestamp,open,high,low,close,volume],按timestamp升序,最后一行是最新已收盘K线}\n"
        "- ctx.memory_context: list[str],记忆检索文本,可以是空列表\n\n"
        "## 模块必须导出\n"
        "- decide(ctx) -> list[Decision]\n"
        "- REQUIRED_HISTORY_BARS: int\n"
        "- DESCRIPTION: str\n"
        "无信号/数据不足时返回空列表 []。不要构造 action='hold' 的Decision去代替空列表。\n\n"
        "## 硬性确定性/沙箱禁止事项(违反任何一条都会被静态检查拒绝并要求重写)\n"
        f"{_FORBIDDEN_SUMMARY}\n"
        "## 输出要求\n"
        "输出完整的python源码(需要的import自己补全,包括"
        "`from ASSET.strategy.policies import StrategyContext` 与 "
        "`from LOCKED.schemas import Decision`),不要在代码之外添加任何解释文字,"
        "也不要省略任何部分。"
    )


def build_retry_prompt(original_prompt: str, previous_source: str, violations: list[str]) -> str:
    violation_lines = "\n".join(f"- {v}" for v in violations)
    return (
        f"{original_prompt}\n\n"
        "---\n"
        f"你上一次的输出未通过静态确定性检查(policy_lint),违规清单:\n{violation_lines}\n\n"
        f"你上一次输出的源码:\n```python\n{previous_source}\n```\n\n"
        "请修正以上全部违规后,重新输出完整的、通过检查的python源码。"
        "仍然只输出源码本身,不要在代码之外添加任何解释文字。"
    )


def generate_and_lint_policy(
    llm_client: Callable[[str], str], prompt: str, max_attempts: int = MAX_LINT_ATTEMPTS
) -> tuple[Optional[str], list[list[str]]]:
    """调用 llm_client 生成 policy 源码并跑 policy_lint.lint_policy_source,
    有违规就把违规清单喂回去要求重写,最多 max_attempts 次。返回
    (source_or_None, attempts_log) —— attempts_log 是每次尝试的违规清单
    (某次为空列表代表那次通过);全部失败则第一个返回值是 None。"""
    attempts_log: list[list[str]] = []
    current_prompt = prompt
    for _ in range(max_attempts):
        raw = llm_client(current_prompt)
        source = strip_code_fences(raw)
        violations = lint_policy_source(source)
        attempts_log.append(violations)
        if not violations:
            return source, attempts_log
        current_prompt = build_retry_prompt(prompt, source, violations)
    return None, attempts_log


def build_protocol_content(experiment_id: str, idea: Idea, reference_kind: str) -> str:
    plan = (
        f"改进 {idea.policy_id}(基于其当前源码)"
        if reference_kind == "improve"
        else f"新建 {idea.policy_id}(参考 {idea.reference_policy_id or _DEFAULT_REFERENCE_POLICY_ID} 的源码)"
    )
    return (
        f"# Experiment Protocol: {experiment_id}\n\n"
        f"- policy_id: {idea.policy_id}\n"
        f"- parent_policy_id: {idea.parent_policy_id}\n"
        f"- idea_source: {idea.source}\n"
        f"- plan: {plan}\n\n"
        f"## Hypothesis\n\n{idea.hypothesis}\n"
    )


# ---------------------------------------------------------------------------
# 7. git 协议封装(全部接受 repo_path 参数,测试永远指向临时仓库)
# ---------------------------------------------------------------------------


def _run_git(args: list[str], repo_path: Path, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True, text=True, encoding="utf-8", timeout=timeout, shell=False,
        # 显式指定utf-8解码——commit message/假设文本含中文,Windows下
        # subprocess.run(text=True)默认按系统locale(GBK)解码会在真实中文
        # commit信息上直接抛UnicodeDecodeError(已实测复现),不是防御性写法。
    )


def _current_head_sha(repo_path: Path) -> str:
    result = _run_git(["rev-parse", "HEAD"], repo_path)
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {result.stderr}")
    return result.stdout.strip()


def git_commit_protocol(repo_path: Path, protocol_file: Path, message: str) -> str:
    """协议提交:先把假设+计划落盘,再commit——必须先于任何回测结果产生,
    这是autoresearch可信度的根基(见模块docstring)。返回该commit的sha。"""
    add = _run_git(["add", str(protocol_file)], repo_path)
    if add.returncode != 0:
        raise RuntimeError(f"git add (protocol) failed: {add.stderr}")
    commit = _run_git(["commit", "-m", message], repo_path)
    if commit.returncode != 0:
        # git的'nothing to commit'走stdout而非stderr——两者都带上,否则报错
        # 信息是一个空字符串,现场无法定位(服务器首次冒烟真实发生过)。
        raise RuntimeError(
            f"git commit (protocol) failed: stderr={commit.stderr!r} stdout={commit.stdout!r}"
        )
    return _current_head_sha(repo_path)


def git_commit_result(repo_path: Path, policy_file: Path, message: str) -> str:
    add = _run_git(["add", str(policy_file)], repo_path)
    if add.returncode != 0:
        raise RuntimeError(f"git add (result) failed: {add.stderr}")
    commit = _run_git(["commit", "-m", message], repo_path)
    if commit.returncode != 0:
        raise RuntimeError(f"git commit (result) failed: {commit.stderr}")
    return _current_head_sha(repo_path)


def git_revert_file(repo_path: Path, file_path: Path) -> None:
    """把 file_path 恢复到revert前的状态。两种情况:
    - 该文件此前已经被提交过(改进型/曾经kept过的旧文件):`git checkout --`
      精确恢复到上一次提交的内容。
    - 该文件是本轮全新创建、从未被提交过(新建型且这轮判定revert):
      `git checkout -- <untracked file>` 在真实git下会直接报错("did not
      match any file(s) known to git"),因为没有"上一个版本"可以恢复——
      这种情况下"恢复原状"就是"删掉这个从未存在过的文件",与"这个策略变体
      本来就不该存在"语义一致。用 `git status --porcelain` 先判断文件是否
      是 untracked("??"前缀)来分流这两条路径。"""
    status = _run_git(["status", "--porcelain", "--", str(file_path)], repo_path)
    if status.stdout.strip().startswith("??"):
        Path(file_path).unlink(missing_ok=True)
        return
    checkout = _run_git(["checkout", "--", str(file_path)], repo_path)
    if checkout.returncode != 0:
        raise RuntimeError(f"git checkout -- failed: {checkout.stderr}")


# ---------------------------------------------------------------------------
# 8. keep/revert 判定(结构性保证:签名里没有 results/holdout)
# ---------------------------------------------------------------------------


def compute_val_stats(results: dict) -> tuple[float, float]:
    """从完整的窗口结果字典里抽出"验证窗口"(排除train和is_holdout)的
    最差edge与最大回撤——与 BacktestEngine.score() 的取值口径一致,单独暴露
    出来是为了把这两个数字原样写进 ledger(供人工/日后晋升评审读,不是给
    decide_keep_or_revert 用)。"""
    val_results = [r for label, r in results.items() if label != "train" and not r.window.is_holdout]
    if not val_results:
        raise ValueError("compute_val_stats: no validation window results (excluding train/holdout)")
    edge = min(r.edge_vs_benchmark_pct for r in val_results)
    dd = max(r.max_drawdown_pct for r in val_results)
    return edge, dd


def compute_val_trade_count(results: dict) -> int:
    """验证窗口(排除train和holdout,口径与compute_val_stats完全一致)的
    trade_count总和——最小样本量门槛(config.backtest.min_val_trades)的
    判定依据。任务2的核心动机:2-3笔交易就能靠运气拿到很高的edge分数,
    量化的统计优势前提是samplesize——52%胜率×1万次才是大数定律生效的场景,
    ×3次只是抛硬币,不该被允许晋级。"""
    val_results = [r for label, r in results.items() if label != "train" and not r.window.is_holdout]
    return sum(int(r.trade_count) for r in val_results)


def historical_best_score(ledger_entries: list[dict], policy_id: str) -> float:
    """该 policy_id 迄今为止所有 status=kept 记录里的最高分数(用
    val_edge_vs_benchmark_pct - 0.5*val_max_drawdown_pct 重算,与
    BacktestEngine.score() 的公式分量完全对应)。没有任何历史kept记录时
    (含首次出现的policy_id)返回0.0——"首轮与0比"。"""
    scores = [
        e["val_edge_vs_benchmark_pct"] - 0.5 * e["val_max_drawdown_pct"]
        for e in ledger_entries
        if e.get("policy_id") == policy_id and e.get("status") == "kept"
        and isinstance(e.get("val_edge_vs_benchmark_pct"), (int, float))
        and isinstance(e.get("val_max_drawdown_pct"), (int, float))
    ]
    return max(scores) if scores else 0.0


def decide_keep_or_revert(new_score: float, previous_best_score: float) -> str:
    """纯函数,只接受两个已经算好的分数——签名里没有 results/holdout 这样的
    参数,holdout结果结构性地不可能进入这个判定函数(不是"调用时不传"这种
    约定层面的自律,而是这个函数根本没有能接收它的形参位置)。

    严格好于历史最优才 keep,持平或更差一律 revert(棘轮式:不允许"打平就
    保留"悄悄放宽晋升门槛)。"""
    return "kept" if new_score > previous_best_score else "reverted"


# ---------------------------------------------------------------------------
# 9. 单轮实验编排
# ---------------------------------------------------------------------------


@dataclass
class LoopContext:
    config: dict
    repo_path: Path
    policies_dir: Path
    experiments_dir: Path
    ledger_path: Path
    protocols_dir: Path
    notes_dir: Path
    deep_llm: Callable[[str], str]
    data_pipeline: Any
    symbols: list[str]
    data_end_ts: int
    scratch_root: Path


def build_loop_context(config: dict, project_root: Path, deep_llm: Callable[[str], str], data_pipeline: Any) -> LoopContext:
    backtest_cfg = config.get("backtest", {}) or {}
    scratch_root = project_root / backtest_cfg.get("scratch_root", "ASSET/strategy/experiments")
    return LoopContext(
        config=config,
        repo_path=project_root,
        policies_dir=POLICIES_DIR,
        experiments_dir=scratch_root,
        ledger_path=scratch_root / "ledger.jsonl",
        protocols_dir=scratch_root / "protocols",
        notes_dir=project_root / "ASSET" / "research_notes",
        deep_llm=deep_llm,
        data_pipeline=data_pipeline,
        symbols=load_universe_symbols(project_root),
        data_end_ts=determine_data_end_ts(config, project_root / "data_cache"),
        scratch_root=scratch_root,
    )


def run_single_experiment(
    ctx: LoopContext, experiment_index: int, dry_run: bool = False, print_fn: Callable[[str], None] = print
) -> dict:
    start_time = time.time()
    ledger_entries = read_ledger(ctx.ledger_path)
    idea = select_idea(experiment_index, ledger_entries, ctx.policies_dir, ctx.notes_dir, experiments_dir=ctx.experiments_dir)
    experiment_id = f"{idea.policy_id}_{experiment_index:04d}"

    reference_kind, reference_source = resolve_reference(ctx.policies_dir, idea)
    # known_classes 只在 director/novelty 两种来源的prompt分支里真正用到
    # (见 _build_source_specific_prompt_section),但计算成本很低、且对
    # 其余来源无副作用,统一算好传下去,不必按source分叉两条调用路径。
    policy_descriptions = collect_existing_strategy_descriptions(ctx.policies_dir)
    known_classes = collect_known_strategy_classes(ledger_entries, policy_descriptions)
    prompt = build_researcher_prompt(idea, idea.policy_id, reference_kind, reference_source, known_classes=known_classes)

    if dry_run:
        print_fn(f"[dry-run] experiment_id={experiment_id} policy_id={idea.policy_id} source={idea.source}")
        print_fn(f"[dry-run] hypothesis: {idea.hypothesis}")
        print_fn(f"[dry-run] prompt:\n{prompt}")
        preview_source, attempts_log = generate_and_lint_policy(ctx.deep_llm, prompt)
        print_fn(f"[dry-run] lint_passed={preview_source is not None} attempts={len(attempts_log)}")
        return {
            "experiment_id": experiment_id,
            "policy_id": idea.policy_id,
            "status": "dry_run",
            "lint_passed": preview_source is not None,
            "attempts": len(attempts_log),
        }

    # ---- 步骤2:协议锁定——先落盘假设+计划,再git commit,早于任何结果 ----
    protocol_path = ctx.protocols_dir / f"{experiment_id}.md"
    ctx.protocols_dir.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(build_protocol_content(experiment_id, idea, reference_kind), encoding="utf-8")
    commit_sha_protocol = git_commit_protocol(
        ctx.repo_path, protocol_path,
        f"research(protocol): {idea.policy_id} — {idea.hypothesis[:80]}",
    )

    # ---- 步骤3-4:LLM生成 + lint重写(最多MAX_LINT_ATTEMPTS次) ----
    policy_path = safe_policy_path(ctx.policies_dir, idea.policy_id)
    source, attempts_log = generate_and_lint_policy(ctx.deep_llm, prompt)

    if source is None:
        wall_time_seconds = time.time() - start_time
        record = build_ledger_record(
            experiment_id=experiment_id, idea=idea,
            commit_sha_protocol=commit_sha_protocol, commit_sha_result=None,
            status="lint_failed",
            val_edge_vs_benchmark_pct=None, val_max_drawdown_pct=None,
            holdout_edge_vs_benchmark_pct=None, wall_time_seconds=wall_time_seconds,
        )
        append_ledger(ctx.ledger_path, record)
        print_fn(
            f"experiment_id={experiment_id} policy_id={idea.policy_id} score=N/A "
            f"status=lint_failed time={wall_time_seconds:.1f}s"
        )
        return record

    policy_path.write_text(source, encoding="utf-8")

    # ---- 步骤5:加载 + 回测 ----
    try:
        module = load_policy_from_dir(idea.policy_id, ctx.policies_dir)
        engine = BacktestEngine(config=ctx.config, data_pipeline=ctx.data_pipeline, scratch_root=ctx.scratch_root)
        windows = build_default_windows(ctx.config, ctx.data_end_ts)
        results = engine.run(module.decide, ctx.symbols, windows, experiment_id=experiment_id)
    except Exception as exc:  # noqa: BLE001 - 任何加载/回测异常都视为本轮失败,revert
        git_revert_file(ctx.repo_path, policy_path)
        wall_time_seconds = time.time() - start_time
        record = build_ledger_record(
            experiment_id=experiment_id, idea=idea,
            commit_sha_protocol=commit_sha_protocol, commit_sha_result=None,
            status="backtest_failed",
            val_edge_vs_benchmark_pct=None, val_max_drawdown_pct=None,
            holdout_edge_vs_benchmark_pct=None, wall_time_seconds=wall_time_seconds,
        )
        append_ledger(ctx.ledger_path, record)
        print_fn(
            f"experiment_id={experiment_id} policy_id={idea.policy_id} score=N/A "
            f"status=backtest_failed time={wall_time_seconds:.1f}s error={exc!r}"
        )
        return record

    # ---- 步骤6:keep/revert判定——holdout严格不参与,见decide_keep_or_revert ----
    scoring_results = {label: r for label, r in results.items() if label != "holdout"}
    new_score = engine.score(scoring_results)
    val_edge, val_dd = compute_val_stats(results)
    val_trade_count = compute_val_trade_count(results)
    holdout_result = results.get("holdout")
    holdout_edge = holdout_result.edge_vs_benchmark_pct if holdout_result is not None else None
    # 修"holdout躺赢"缺陷:同时记下holdout窗口的真实交易笔数与绝对收益,
    # 供下游识别"这个holdout_edge是不是零交易的镜像分数"(见
    # build_ledger_record docstring)。
    holdout_trade_count = int(holdout_result.trade_count) if holdout_result is not None else None
    holdout_return_pct = float(holdout_result.return_pct) if holdout_result is not None else None

    # ---- 任务2:最小样本量门槛——比分数判定更前置,总交易笔数不够直接
    # revert,分数再好也不算数(见 config.yaml backtest.min_val_trades 注释:
    # 2-3笔交易靠运气拿高分是抛硬币,不是统计优势)。
    min_val_trades = int((ctx.config.get("backtest", {}) or {}).get("min_val_trades", 10))
    if val_trade_count < min_val_trades:
        status = "reverted"
        revert_reason = "insufficient_trades"
    else:
        previous_best = historical_best_score(ledger_entries, idea.policy_id)
        status = decide_keep_or_revert(new_score, previous_best)
        revert_reason = None if status == "kept" else "score_not_better"

    if status == "kept":
        commit_sha_result = git_commit_result(
            ctx.repo_path, policy_path, f"research(results): {idea.policy_id} — score={new_score:.3f}",
        )
    else:
        git_revert_file(ctx.repo_path, policy_path)
        commit_sha_result = None

    wall_time_seconds = time.time() - start_time
    record = build_ledger_record(
        experiment_id=experiment_id, idea=idea,
        commit_sha_protocol=commit_sha_protocol, commit_sha_result=commit_sha_result,
        status=status,
        val_edge_vs_benchmark_pct=val_edge, val_max_drawdown_pct=val_dd,
        holdout_edge_vs_benchmark_pct=holdout_edge, wall_time_seconds=wall_time_seconds,
        revert_reason=revert_reason,
        holdout_trade_count=holdout_trade_count, holdout_return_pct=holdout_return_pct,
    )
    append_ledger(ctx.ledger_path, record)
    print_fn(
        f"experiment_id={experiment_id} policy_id={idea.policy_id} "
        f"score={new_score:.3f} status={status} time={wall_time_seconds:.1f}s"
    )
    return record


def existing_experiment_count(protocols_dir: Path) -> int:
    """已存在的协议文件数,作为本次运行experiment_index的起始偏移。

    真实踩过的坑(服务器首次冒烟):上一次运行在协议commit之后、写ledger之前
    崩溃(LLM调用失败),重跑时index从0重来->生成同名协议文件+相同内容->
    git 'nothing to commit'->整个循环报错退出。协议文件本身就是最可靠的
    "已经消耗掉的实验序号"记录(它先于一切结果落盘),用它做偏移比用ledger
    行数更能覆盖"半途崩溃"的场景。"""
    if not protocols_dir.exists():
        return 0
    return len(list(protocols_dir.glob("*.md")))


def run_research_loop(
    ctx: LoopContext, max_experiments: int, dry_run: bool = False, print_fn: Callable[[str], None] = print
) -> list[dict]:
    max_per_night = int((ctx.config.get("backtest", {}) or {}).get("max_experiments_per_night", 50))
    limit = max(0, min(max_experiments, max_per_night))
    base = existing_experiment_count(ctx.protocols_dir)
    records = []
    for i in range(limit):
        records.append(run_single_experiment(ctx, base + i, dry_run=dry_run, print_fn=print_fn))
    # 所有实验跑完之后自动跑一次外环研究总监,产出给下一晚select_idea()消费
    # 的方向备忘录——dry_run时跳过(dry_run本身就是"只看想法/prompt/lint,
    # 不动磁盘/git",总监落盘memo文件属于"动磁盘",与dry_run的语义矛盾)。
    if not dry_run:
        run_research_director(ctx, print_fn=print_fn)
    return records


# ---------------------------------------------------------------------------
# 9b. 外环研究总监(§任务1:让内环从"无记忆爬山"升级为"真正做研究")
# ---------------------------------------------------------------------------

_DIRECTOR_MAX_ATTEMPTS = 3
_DIRECTOR_MIN_DIRECTIONS = 3
_DIRECTOR_MAX_DIRECTIONS = 5


def _read_tactic_promotions(repo_path: Path) -> list[dict]:
    """LOG/tactic_promotions.jsonl——战术锦标赛的分支死亡/晋升档案(见
    scripts/ignite.py evaluate_tactic_tournament/evaluate_cull 写入处,
    字段:branch/decision[PROMOTE|FAIL|CULLED]/edge_vs_main_pct/
    max_drawdown_pct/reason/ts)。复用read_ledger:纯jsonl读取,文件不存在
    -> 空列表,与ledger.jsonl的读取方式完全对称,不需要为一个不同目录下的
    同构jsonl文件另写一遍读取逻辑。"""
    return read_ledger(repo_path / "LOG" / "tactic_promotions.jsonl")


def collect_existing_strategy_descriptions(policies_dir: Path) -> dict[str, str]:
    """遍历 policies_dir 下全部 *.py(排除 __init__.py),用
    load_policy_from_dir 尝试加载并取出 DESCRIPTION 导出;加载失败(语法
    错误/契约缺失/AST层面就有问题)的文件直接跳过——研究总监需要的是"现有
    策略类别一览"这个粗粒度输入,不因为某一个坏文件让整条总监流程崩溃。"""
    result: dict[str, str] = {}
    if not policies_dir.exists():
        return result
    for path in sorted(policies_dir.glob("*.py")):
        if path.stem == "__init__":
            continue
        try:
            module = load_policy_from_dir(path.stem, policies_dir)
        except Exception:  # noqa: BLE001 - 任何加载失败都跳过,不中断总监流程
            continue
        desc = getattr(module, "DESCRIPTION", None)
        if isinstance(desc, str) and desc.strip():
            result[path.stem] = desc.strip()
    return result


def collect_known_strategy_classes(ledger_entries: list[dict], policy_descriptions: dict[str, str]) -> list[str]:
    """"已出现过的策略类别"清单,两个来源合并去重:
      (a) 现有策略文件的"家族名"(policy_id去掉_vN后缀,如 momentum_v2 ->
          momentum)——代表"已经有可运行实现"的类别;
      (b) ledger里带 strategy_class 标记的历史实验(director来源产生的
          类别,见Idea.strategy_class/build_ledger_record)——代表"总监已经
          明确指派过"的类别,哪怕对应实验最终被revert也算"出现过",避免
          总监反复重复建议同一个已经试过的方向。

    规格书没有强制规定"策略类别"这个概念该有一个独立的分类字段/标签体系,
    这是本次实现自行拍板的一项设计决定:不引入额外的人工打标签流程,只是
    复用已经存在的policy_id命名家族 + 总监自己历史上分配过的class标签,
    对"避免和已有类别重复"这个实际需求已经足够。"""
    classes = {_strip_variant_suffix(pid) for pid in policy_descriptions}
    for e in ledger_entries:
        cls = e.get("strategy_class")
        if isinstance(cls, str) and cls.strip():
            classes.add(cls.strip())
    return sorted(classes)


def _summarize_trajectory(ledger_entries: list[dict]) -> list[str]:
    # holdout_trade_count/holdout_return_pct(本次改动新增,修"holdout躺赢"
    # 缺陷)一并带上,让总监能看到"这条holdout_edge是不是零交易的镜像分数"——
    # 见 build_ledger_record docstring 与 build_director_prompt 里的专门提示。
    return [
        f"- policy_id={e.get('policy_id')} hypothesis={(e.get('hypothesis') or '')[:60]!r} "
        f"status={e.get('status')} val_edge={e.get('val_edge_vs_benchmark_pct')} "
        f"val_dd={e.get('val_max_drawdown_pct')} holdout_edge={e.get('holdout_edge_vs_benchmark_pct')} "
        f"holdout_trade_count={e.get('holdout_trade_count')} holdout_return_pct={e.get('holdout_return_pct')}"
        for e in ledger_entries
    ]


def _summarize_death_archive(events: list[dict]) -> list[str]:
    return [
        f"- branch={e.get('branch')} decision={e.get('decision')} "
        f"edge_vs_main_pct={e.get('edge_vs_main_pct')} max_drawdown_pct={e.get('max_drawdown_pct')} "
        f"reason={e.get('reason')} ts={e.get('ts')}"
        for e in events
    ]


def build_director_prompt(
    ledger_entries: list[dict],
    death_events: list[dict],
    policy_descriptions: dict[str, str],
    known_classes: list[str],
    retry_feedback: Optional[str] = None,
) -> str:
    trajectory_lines = _summarize_trajectory(ledger_entries)
    death_lines = _summarize_death_archive(death_events)
    strategy_lines = [f"- {pid}: {desc[:120]}" for pid, desc in policy_descriptions.items()]
    lines = [
        "你是AlphaLoop-Crypto的外环研究总监。你不写代码,你的职责是回顾内环"
        "全部实验轨迹,给出下一批研究方向,引导内环从'无记忆爬山'升级为"
        "'真正做研究'。",
        "",
        "## 实验轨迹(全部ledger记录:policy_id/hypothesis/status/"
        "val_edge/val_dd/holdout_edge/holdout_trade_count/holdout_return_pct)",
        "\n".join(trajectory_lines) if trajectory_lines else "(尚无实验记录,冷启动)",
        "注意:holdout_trade_count=0 的策略,其holdout edge只是基准涨跌的"
        "镜像,不代表策略能力——评估轨迹/挑选参照对象时请忽略这类记录的"
        "holdout表现,不要把它当作该策略在holdout窗口上真实验证过的信号。",
        "",
        "## 分支死亡/晋升档案(LOG/tactic_promotions.jsonl)",
        "\n".join(death_lines) if death_lines else "(尚无记录)",
        "",
        "## 现有策略类别清单(以下类别已经有实现或已经被指派过,新方向不应与其重复)",
        "\n".join(strategy_lines) if strategy_lines else "(尚无策略文件)",
        f"已出现过的策略类别名:{', '.join(known_classes) if known_classes else '(无)'}",
        "",
        "## 输出要求",
        "严格输出一个JSON对象(不要markdown代码块,不要任何JSON之外的文字):",
        '{"trajectory_insights": ["洞察1", "洞察2", ...], '
        '"directions": [{"class": "策略类别名", "hypothesis": "具体可实施的研究方向一句话", '
        '"rationale": "基于轨迹的理由"}, ...]}',
        f"directions 需要{_DIRECTOR_MIN_DIRECTIONS}到{_DIRECTOR_MAX_DIRECTIONS}条,"
        "每条都必须有非空的class与hypothesis字段。其中至少1条的class必须是"
        "上面'现有策略类别清单'里完全没有出现过的全新类别——不要全部都是对"
        "已有类别的参数微调式重复。",
    ]
    if retry_feedback:
        lines.append("")
        lines.append(f"你上一次的输出未通过校验:{retry_feedback}。请修正后重新严格输出JSON对象。")
    return "\n".join(lines)


def _validate_director_response(raw: str, known_classes: list[str]) -> tuple[Optional[dict], Optional[str]]:
    try:
        parsed = json.loads(strip_code_fences(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "response_not_valid_json"
    if not isinstance(parsed, dict):
        return None, "response_not_a_json_object"

    insights = parsed.get("trajectory_insights")
    if not isinstance(insights, list):
        return None, "trajectory_insights_must_be_a_list"

    directions = parsed.get("directions")
    if not isinstance(directions, list) or not (_DIRECTOR_MIN_DIRECTIONS <= len(directions) <= _DIRECTOR_MAX_DIRECTIONS):
        return None, f"directions_must_be_a_list_of_{_DIRECTOR_MIN_DIRECTIONS}_to_{_DIRECTOR_MAX_DIRECTIONS}_items"

    known_lower = {c.strip().lower() for c in known_classes if isinstance(c, str) and c.strip()}
    has_novel_class = False
    for d in directions:
        if not isinstance(d, dict):
            return None, "each_direction_must_be_an_object"
        cls = d.get("class")
        hyp = d.get("hypothesis")
        if not isinstance(cls, str) or not cls.strip():
            return None, "direction_missing_non_empty_class"
        if not isinstance(hyp, str) or not hyp.strip():
            return None, "direction_missing_non_empty_hypothesis"
        if cls.strip().lower() not in known_lower:
            has_novel_class = True
    if not has_novel_class:
        return None, "directions_must_include_at_least_one_class_not_in_known_classes"

    return parsed, None


def _render_direction_memo_markdown(memo: dict) -> str:
    lines = [f"# Research Direction Memo (ts={memo.get('ts')})", ""]
    lines.append("## Trajectory Insights")
    for insight in memo.get("trajectory_insights") or []:
        lines.append(f"- {insight}")
    lines.append("")
    lines.append("## Directions")
    for d in memo.get("directions") or []:
        lines.append(f"### {d.get('class')}")
        lines.append(f"- hypothesis: {d.get('hypothesis')}")
        lines.append(f"- rationale: {d.get('rationale', '')}")
        lines.append("")
    return "\n".join(lines)


def run_research_director(ctx: LoopContext, print_fn: Callable[[str], None] = print) -> Optional[dict]:
    """外环研究总监:回顾ledger全部实验轨迹 + LOG/tactic_promotions.jsonl
    死亡/晋升档案 + 现有策略DESCRIPTION清单,向 ctx.deep_llm 请求下一批研究
    方向,校验通过后落盘 direction_memo.json/.md + 历史存档
    experiments/memos/memo_{ts}.json,供下一次 select_idea() 消费。

    最多重试_DIRECTOR_MAX_ATTEMPTS次,全部未通过校验则打印一行日志并返回
    None,不抛异常——总监是"锦上添花"的外环指导,它这一轮失败不应该让
    已经跑完的内环实验结果无法落盘、也不应该让整个调用方进程崩溃退出。

    ts 直接复用 ctx.data_end_ts(已经是"由缓存数据推出的确定性时间戳",不是
    墙钟),而不是另外调用 time.time()——与本模块其余部分"不读墙钟"的精神
    一致,顺带也让测试不需要额外 monkeypatch 时间模块。"""
    ledger_entries = read_ledger(ctx.ledger_path)
    death_events = _read_tactic_promotions(ctx.repo_path)
    policy_descriptions = collect_existing_strategy_descriptions(ctx.policies_dir)
    known_classes = collect_known_strategy_classes(ledger_entries, policy_descriptions)

    parsed: Optional[dict] = None
    retry_feedback: Optional[str] = None
    for _attempt in range(_DIRECTOR_MAX_ATTEMPTS):
        prompt = build_director_prompt(ledger_entries, death_events, policy_descriptions, known_classes, retry_feedback)
        raw = ctx.deep_llm(prompt)
        parsed, error = _validate_director_response(raw, known_classes)
        if error is None:
            break
        retry_feedback = error
        parsed = None

    if parsed is None:
        print_fn(f"[research_director] 连续{_DIRECTOR_MAX_ATTEMPTS}次未通过校验,放弃本轮,last_error={retry_feedback}")
        return None

    ts = ctx.data_end_ts
    memo = dict(parsed)
    memo["ts"] = ts

    ctx.experiments_dir.mkdir(parents=True, exist_ok=True)
    memo_json = json.dumps(memo, ensure_ascii=False, indent=2)
    (ctx.experiments_dir / "direction_memo.json").write_text(memo_json, encoding="utf-8")
    (ctx.experiments_dir / "direction_memo.md").write_text(_render_direction_memo_markdown(memo), encoding="utf-8")

    memos_dir = ctx.experiments_dir / "memos"
    memos_dir.mkdir(parents=True, exist_ok=True)
    (memos_dir / f"memo_{ts}.json").write_text(memo_json, encoding="utf-8")

    print_fn(f"[research_director] memo written: {len(memo.get('directions') or [])} directions, ts={ts}")
    return memo


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------


def load_config(project_root: Path) -> dict:
    with open(project_root / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaLoop-Crypto M7 内环研究循环")
    parser.add_argument("--max-experiments", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--director-only", action="store_true",
        help="只跑一次外环研究总监(run_research_director),不跑任何内环实验——"
        "用于单独刷新 direction_memo,不消耗实验预算/不产生新的policy变体。",
    )
    args = parser.parse_args()

    from LOCKED.data_pipeline import DataPipeline

    config = load_config(PROJECT_ROOT)
    _routine_llm, deep_llm = build_llm_clients(config)
    data_pipeline = DataPipeline(exchange_id=config["data"]["exchange"])
    ctx = build_loop_context(config, PROJECT_ROOT, deep_llm, data_pipeline)
    if args.director_only:
        run_research_director(ctx, print_fn=print)
    else:
        run_research_loop(ctx, max_experiments=args.max_experiments, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
