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
) -> dict:
    """schema 逐字遵守任务说明里列出的字段集合。"""
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
    }


# ---------------------------------------------------------------------------
# 3. 想法挑选(§3.5三个来源,按序轮转;冷启动从种子策略开始)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Idea:
    source: str  # "kept_variant" | "research_note" | "seed_variant"
    policy_id: str  # 目标文件名(已通过路径安全校验的候选)
    parent_policy_id: Optional[str]
    hypothesis: str
    reference_policy_id: Optional[str]  # "新建型"时用来给LLM参照的源码来自哪个policy_id


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


def _next_policy_variant_id(base_policy_id: str, policies_dir: Path) -> str:
    """给 base_policy_id 生成一个尚未存在于 policies_dir 下、且满足
    ^[a-z][a-z0-9_]{2,40}$ 白名单的新 policy_id,形如 "{stem}_v{n}"。
    先去掉 base_policy_id 已有的 "_vN" 后缀(如果有),避免变体的变体越叠
    越长(momentum_v1 -> momentum_v2,而不是 momentum_v1_v2)。"""
    stem = re.sub(r"_v\d+$", "", base_policy_id) or base_policy_id
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


def select_idea(
    experiment_index: int,
    ledger_entries: list[dict],
    policies_dir: Path,
    notes_dir: Path,
) -> Idea:
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


def build_researcher_prompt(idea: Idea, target_policy_id: str, reference_kind: str, reference_source: str) -> str:
    reference_note = (
        "改进型:下面是目标文件当前的源码,请在此基础上按假设修改"
        if reference_kind == "improve"
        else "新建型:下面是最相近的参考源码(种子策略或血缘父策略),请仿照其结构新建"
    )
    return (
        "你是AlphaLoop-Crypto的内环策略研究员。任务:为下面这条假设写/改一个"
        "确定性的Python策略模块(ASSET/strategy/policies 目录下的policy)。\n\n"
        f"## 目标 policy_id\n{target_policy_id}\n\n"
        f"## 假设\n{idea.hypothesis}\n\n"
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
    idea = select_idea(experiment_index, ledger_entries, ctx.policies_dir, ctx.notes_dir)
    experiment_id = f"{idea.policy_id}_{experiment_index:04d}"

    reference_kind, reference_source = resolve_reference(ctx.policies_dir, idea)
    prompt = build_researcher_prompt(idea, idea.policy_id, reference_kind, reference_source)

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
    holdout_result = results.get("holdout")
    holdout_edge = holdout_result.edge_vs_benchmark_pct if holdout_result is not None else None

    previous_best = historical_best_score(ledger_entries, idea.policy_id)
    status = decide_keep_or_revert(new_score, previous_best)

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
    return records


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
    args = parser.parse_args()

    from LOCKED.data_pipeline import DataPipeline

    config = load_config(PROJECT_ROOT)
    _routine_llm, deep_llm = build_llm_clients(config)
    data_pipeline = DataPipeline(exchange_id=config["data"]["exchange"])
    ctx = build_loop_context(config, PROJECT_ROOT, deep_llm, data_pipeline)
    run_research_loop(ctx, max_experiments=args.max_experiments, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
